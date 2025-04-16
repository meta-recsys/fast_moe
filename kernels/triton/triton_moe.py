#!/usr/bin/env python3

# pyre-unsafe

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

from fast_moe.kernels.triton.utils import (
    get_bmm_configs,
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from torch.autograd.profiler import record_function


@torch.fx.wrap
def triton_index_select_jagged_bmm(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return IndexSelectJaggedBmm.apply(max_seq_len, offsets, index, jagged, weight, bias)


def triton_index_select_jagged_bmm_wrapper(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return triton_index_select_jagged_bmm(
        max_seq_len=max_seq_len,
        offsets=offsets,
        index=index,
        jagged=jagged,
        weight=weight,
        bias=bias,
    )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _index_select_jagged_bmm(
    seq_offsets,  # [B+1]
    Index,  # [Sum_B(M)], jagged indices in range [0, L * A)
    Jagged,  # [L, K]
    Dense,  # [B, K, N]
    Bias,  # [B, N]
    Out,  # [Sum_B(M), N]
    M,
    N,
    K,
    A,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute Batched Matrix Multiplication (BMM) of the form Out = Jagged @ Dense + Bias.
    Specifically for b in range(B):
        Jagged_b = Jagged[Index[seq_offsets[b]:seq_offsets[b+1]] // A]  # [M, K]
        Dense_b = Dense[b]  # [K, N]
        Bias_b = Bias[b]  # [N]
        Out_b = Jagged_b @ Dense_b + Bias_b  # [M, N]
    Split the kernel into (b, m, n) grid, each program processes [BLOCK_M, BLOCK_N] output elements for specific b.
    """
    off_b = tl.program_id(0)
    off_m = tl.program_id(1)
    off_n = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b, eviction_policy="evict_last")
    seq_end = tl.load(seq_offsets + off_b + 1, eviction_policy="evict_last")
    seq_len = seq_end - seq_start  # Matrix size in M dimension for b-th matmul
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Index += seq_start
    Dense += off_b * stride_db
    Out += seq_start.to(tl.int64) * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    # load index for all rows to be processed by this block
    idx_ptrs = Index + offs_m
    idx = tl.load(idx_ptrs, mask=offs_m < seq_len, other=0)  # [BLOCK_M]
    idx = idx // A

    # [BLOCK_M, BLOCK_K]
    jg_ptrs = Jagged + idx[:, None] * stride_jm + offs_k[None, :]
    # [BLOCK_K, BLOCK_N]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )  # [BLOCK_K, BLOCK_N]

        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        # load bias
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n  # [BLOCK_N]
        bias = tl.load(
            bias_ptrs, mask=offs_n < N, eviction_policy="evict_last"
        )  # [BLOCK_N]
        # add bias to accumulator [BLOCK_M, BLOCK_N]
        out = (accumulator + bias[None, :].to(tl.float32)).to(Out.dtype.element_ty)
    else:
        out = accumulator.to(Out.dtype.element_ty)

    # write back [BLOCK_M, BLOCK_N]
    out_ptrs = Out + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _jagged_bmm_index_add(
    seq_offsets,  # [B+1]
    Index,  # [Sum_B(M)], jagged indices in range [0, L * A)
    Jagged,  # [Sum_B(M), K]
    Dense,  # [B, K, N]
    Out,  # [Sum_B(M), N]
    M,
    N,
    K,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_om,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute Batched Matrix Multiplication (BMM) of the form Out = Jagged @ Dense.
    Specifically for b in range(B):
        Jagged_b = Jagged[seq_offsets[b]:seq_offsets[b+1]]  # [M, K]
        Dense_b = Dense[b]  # [K, N]
        Out_b[Index[seq_offsets[b]:seq_offsets[b+1]]] = Jagged_b @ Dense_b  # [M, N]
    Split the kernel into (m, b, n) grid, each program processes [BLOCK_M, BLOCK_N] output elements for specific b.
    """
    off_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_n = tl.program_id(2)

    seq_start = tl.load(
        seq_offsets + off_b,
        eviction_policy="evict_last",
    )
    seq_end = tl.load(
        seq_offsets + off_b + 1,
        eviction_policy="evict_last",
    )
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += seq_start.to(tl.int64) * stride_jm
    Dense += off_b * stride_db

    offs_m = start_m + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    jg_ptrs = (
        Jagged + offs_m[:, None].to(tl.int64) * stride_jm + offs_k[None, :]
    )  # [BLOCK_M, BLOCK_K]
    dn_ptrs = (
        Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn
    )  # [BLOCK_K, BLOCK_N]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )  # [BLOCK_K, BLOCK_N]
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    # load index for all rows to be processed by this block
    Index += seq_start
    idx_ptrs = Index + offs_m
    idx = tl.load(idx_ptrs, mask=offs_m < seq_len, other=0)  # [BLOCK_M]

    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = (
        Out + idx[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    )  # [BLOCK_M, BLOCK_N]

    # write back [BLOCK_M, BLOCK_N]
    tl.store(
        out_ptrs,
        accumulator.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _indexed_jagged_jagged_bmm_reduce_sum(
    seq_offsets,  # [B+1]
    Index,  # [Sum_B(M)], jagged indices in range [0, L)
    JaggedA,  # [M, L]
    JaggedB,  # [N, Sum_B(M)]
    Out,  # [B, M, N]
    ReduceOut,  # [B, N]
    M,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
    stride_orb,
    stride_orn,
    REDUCE_JAGGEDB: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute Batched Matrix Multiplication (BMM) of the form:
    Out = JaggedA @ JaggedB.T  # [B, M, N]
    , specifically for b in range(B):
        JaggedA_b = JaggedA[:, Index[seq_offsets[b]:seq_offsets[b+1]]]  # [M, K]
        JaggedB_b = JaggedB[:, seq_offsets[b]:seq_offsets[b+1]]  # [N, K]
        Out_b = JaggedA_b @ JaggedB_b.T  # [M, N]
    ReduceOut = sum(JaggedB, axis=1), specifically for b in range(B):
        ReduceOut_b = sum(JaggedB[:, seq_offsets[b]:seq_offsets[b+1]], axis=1)  # [N]
    Split the kernel into (b, m, n) grid, each program processes [BLOCK_M, BLOCK_N] output elements for specific b.
    """
    off_b = tl.program_id(0)  # expert index
    off_m = tl.program_id(1)  # output M dimension index
    off_n = tl.program_id(2)  # output N dimension index

    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start  # Matrix size in K dimension for b-th matmul

    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
    Out += off_b * stride_ob
    offs_m = start_m + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    out_ptrs = (
        Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )  # [BLOCK_M, BLOCK_N]

    if REDUCE_JAGGEDB:
        out_reduce_ptrs = (
            ReduceOut + off_b * stride_orb + offs_n * stride_orn
        )  # [BLOCK_N]
        acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)  # [BLOCK_N]

    if seq_len == 0:
        out = accumulator.to(Out.dtype.element_ty)
        tl.store(
            out_ptrs,
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )
        if REDUCE_JAGGEDB:
            if off_m == 0:
                tl.store(
                    out_reduce_ptrs,  # pyre-ignore [61]
                    acc_reduce.to(ReduceOut.dtype.element_ty),
                    mask=(offs_n < N),
                )
        return

    Index += seq_start
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    idx_ptrs = Index + offs_k  # [BLOCK_K]

    JaggedB += seq_start.to(tl.int64) * stride_bk

    jg_b_ptrs = (
        JaggedB + offs_k[:, None].to(tl.int64) * stride_bk + offs_n[None, :]
    )  # [BLOCK_K, BLOCK_N]

    for k in range(0, seq_len, BLOCK_K):
        idx = tl.load(idx_ptrs, mask=(k + offs_k) < seq_len, other=0)  # [BLOCK_K]
        jg_a_ptrs = (
            JaggedA + idx[None, :] * stride_ak + offs_m[:, None]
        )  # [BLOCK_M, BLOCK_K]
        jg_a = tl.load(
            jg_a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < M) and ((k + offs_k)[None, :] < seq_len),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        jg_b = tl.load(
            jg_b_ptrs,
            mask=(offs_n[None, :] < N) and ((k + offs_k)[:, None] < seq_len),
            other=0.0,
        )  # [BLOCK_K, BLOCK_N]

        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if REDUCE_JAGGEDB:
            if off_m == 0:
                acc_reduce += tl.sum(jg_b, axis=0)

        idx_ptrs += BLOCK_K
        jg_b_ptrs += BLOCK_K * stride_bk

    # write back [BLOCK_M, BLOCK_N]
    out = accumulator.to(Out.dtype.element_ty)
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    if REDUCE_JAGGEDB:
        if off_m == 0:
            # write back [BLOCK_N]
            tl.store(
                out_reduce_ptrs,  # pyre-ignore [61]
                acc_reduce.to(ReduceOut.dtype.element_ty),
                mask=(offs_n < N),
            )


class IndexSelectJaggedBmm(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        offsets: torch.Tensor,
        index: torch.Tensor,
        jagged: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            max_seq_len (int): Maximum number of input tokens for any expert.
            offsets (torch.Tensor): A tensor of shape [E + 1] representing the cumulative number of tokens dispatched to each expert.
            index (torch.Tensor): A tensor of shape [L, A] that is flattened and sorted by expert.
            jagged (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
            weight (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
            bias (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
        Returns:
            torch.Tensor: A tensor of shape [L * A, N] containing the output after applying the linear transformation.
        """
        index = switch_to_contiguous_if_needed(index)
        jagged = switch_to_contiguous_if_needed(jagged)
        if bias is not None:
            bias = switch_to_contiguous_if_needed(bias)

        # L: number of input tokens
        # A: number of activated experts
        # E: number of total experts
        # K: input dimension
        # N: output dimension
        L, A = index.shape
        _, K = jagged.shape
        E, _, N = weight.shape
        output = torch.empty(
            (L * A, N), dtype=jagged.dtype, device=jagged.device
        )  # [L * A, N]

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        _index_select_jagged_bmm[grid](
            seq_offsets=offsets,
            Index=index,
            Jagged=jagged,
            Dense=weight,
            Bias=bias,
            Out=output,
            # M is only used for trigger autotune
            M=triton.next_power_of_2(max_seq_len),
            N=N,
            K=K,
            A=A,
            stride_jm=jagged.stride(0),
            stride_db=weight.stride(0),
            stride_dk=weight.stride(1),
            stride_dn=weight.stride(2),
            stride_bias_b=bias.stride(0) if bias is not None else 0,
            stride_om=output.stride(0),
            HAS_BIAS=bias is not None,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        ctx.save_for_backward(offsets, index, jagged, weight, bias)
        ctx.E = E
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.N = N

        return output

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx,
        d_out: torch.Tensor,
    ) -> Tuple[
        None,
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Args:
            d_out (torch.Tensor): A tensor of shape [L * A, N] representing the gradient of the output.
        Returns:
            Tuple[None, None, None, torch.Tensor, torch.Tensor, torch.Tensor]:
                - None: No gradient is computed for `max_seq_len`.
                - None: No gradient is computed for `offsets`.
                - None: No gradient is computed for `index`.
                - torch.Tensor: Gradient with respect to `jagged`, of shape [L, K].
                - torch.Tensor: Gradient with respect to `weight`, of shape [E, K, N].
                - torch.Tensor: Gradient with respect to `bias`, of shape [E, N].
        """
        # offsets: [E + 1]
        # index: [L, A]
        # jagged: [L, K]
        # weight: [E, K, N]
        # bias: [E, N]
        offsets, index, jagged, weight, bias = ctx.saved_tensors
        E, K, N = ctx.E, ctx.K, ctx.N
        max_seq_len = ctx.max_seq_len

        with record_function("#### d_jagged ####"):
            d_jagged = IndexSelectJaggedBmm._calc_d_jagged(
                jagged, max_seq_len, index, E, K, N, weight, d_out, offsets
            )

        with record_function("#### d_weight ####"):
            d_weight, d_bias = IndexSelectJaggedBmm._calc_d_weight_bias(
                weight, bias, offsets, index, d_out, jagged, E, K, N, max_seq_len
            )

        return (
            None,
            None,
            None,
            d_jagged,
            d_weight,
            d_bias,
        )

    @staticmethod
    def _calc_d_jagged(
        jagged: torch.Tensor,
        max_seq_len: int,
        index: torch.Tensor,
        E,
        K,
        N,
        weight: torch.Tensor,
        d_out: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        d_jagged_expanded = torch.empty(
            (jagged.shape[0], index.shape[1], jagged.shape[1]),
            device=jagged.device,
            dtype=torch.float32,
        )  # [L, A, K]
        grid = lambda meta: (  # noqa E731
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            E,
            triton.cdiv(K, meta["BLOCK_N"]),
        )
        _jagged_bmm_index_add[grid](
            seq_offsets=offsets,
            Index=index,
            Jagged=d_out,
            Dense=weight,
            Out=d_jagged_expanded,
            # M is only used for triggering autotune
            M=triton.next_power_of_2(max_seq_len),
            N=K,
            K=N,
            stride_jm=d_out.stride(0),
            stride_db=weight.stride(0),
            stride_dk=weight.stride(2),
            stride_dn=weight.stride(1),
            stride_om=d_jagged_expanded.stride(1),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )
        return d_jagged_expanded.sum(dim=1).to(jagged.dtype)  # sum over A dimension

    @staticmethod
    def _calc_d_weight_bias(
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        offsets: torch.Tensor,
        index: torch.Tensor,
        d_out: torch.Tensor,
        jagged: torch.Tensor,
        E: int,
        K: int,
        N: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        has_bias = False
        # tensors below needs to be initialized with zeros as there could be unused
        # rows in the weight and bias
        d_weight = torch.zeros_like(weight)  # [E, K, N]
        if bias is not None:
            has_bias = True
            d_bias = torch.zeros_like(bias)  # [E, N]
            stride_orb, stride_orn = d_bias.stride(0), d_bias.stride(1)
        else:
            d_bias = None
            stride_orb, stride_orn = 0, 0
        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(K, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _indexed_jagged_jagged_bmm_reduce_sum[grid](
            seq_offsets=offsets,
            Index=index.view(-1) // index.shape[1],
            JaggedA=jagged,
            JaggedB=d_out,
            Out=d_weight,
            ReduceOut=d_bias,
            M=K,
            N=N,
            # K is only used for triggering autotune
            K=triton.next_power_of_2(max_seq_len),
            stride_ak=jagged.stride(0),
            stride_bk=d_out.stride(0),
            stride_ob=d_weight.stride(0),
            stride_om=d_weight.stride(1),
            stride_on=d_weight.stride(2),
            stride_orb=stride_orb,
            stride_orn=stride_orn,
            REDUCE_JAGGEDB=has_bias,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )
        return d_weight, d_bias


@torch.fx.wrap
def triton_mul_merge_k_add(
    index: torch.Tensor,  # Used for pytorch replaced op for aoti
    reverse_index: torch.Tensor,
    x: torch.Tensor,
    k: int,
    weight: Optional[torch.Tensor] = None,
    weight_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return MulMergeKAddFunction.apply(reverse_index, x, k, weight, weight_index)


def triton_mul_merge_k_add_wrapper(
    index: torch.Tensor,
    reverse_index: torch.Tensor,
    x: torch.Tensor,
    k: int,
    weight: Optional[torch.Tensor] = None,
    weight_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return triton_mul_merge_k_add(
        index, reverse_index, x, k, weight=weight, weight_index=weight_index
    )


def _get_mul_merge_k_add_fwd_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [64, 128]:
        for BLOCK_D in [64, 128]:
            for num_warps in [8, 16]:
                configs.append(
                    triton.Config(
                        {"BLOCK_N": BLOCK_N, "BLOCK_D": BLOCK_D},
                        num_warps=num_warps,
                    )
                )
    return configs


def _get_mul_merge_k_add_bwd_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [64, 128]:
        for num_warps in [8, 16]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton_autotune(
    configs=_get_mul_merge_k_add_fwd_configs(),
    key=["dim_D"],
)
@triton.jit
def _mul_merge_k_add_fwd(
    IDX,  # [dim_N, K]: for each row in Y which K indices from X to add
    X,  # [dim_N * K, dim_D]
    W,  # [dim_N, K]
    W_INDEX,  # [dim_N * K]
    Y,  # [dim_N, dim_D]
    dim_N,
    dim_D,
    W_STRIDE: tl.constexpr,
    K: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    r, c = tl.program_id(1), tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_D)

    dim_Nx = dim_N * K
    off_idx = (r * BLOCK_N + rows) * K
    mask_idx = off_idx < dim_Nx
    y = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # loop over all K elements from X to add to Y
    for i in range(K):
        row_idx = tl.load(
            IDX + off_idx + i, mask=mask_idx, other=dim_Nx, eviction_policy="evict_last"
        )
        off_x = row_idx.to(tl.int64)[:, None] * dim_D + c * BLOCK_D + cols[None, :]
        mask_x = (row_idx[:, None] < dim_Nx) & (c * BLOCK_D + cols[None, :] < dim_D)
        x = tl.load(X + off_x, mask=mask_x)

        if HAS_WEIGHT:
            w_index = tl.load(
                W_INDEX + row_idx, mask=row_idx < dim_Nx, eviction_policy="evict_last"
            )
            offs_w = w_index // K * W_STRIDE + w_index % K
            w = tl.load(W + offs_w, mask=row_idx < dim_Nx, eviction_policy="evict_last")
            x = x * w[:, None]

        y += x

    off_y = (
        (r.to(tl.int64) * BLOCK_N + rows[:, None]) * dim_D + c * BLOCK_D + cols[None, :]
    )
    mask_y = (r * BLOCK_N + rows[:, None] < dim_N) & (
        c * BLOCK_D + cols[None, :] < dim_D
    )
    tl.store(
        Y + off_y, y.to(Y.dtype.element_ty), mask=mask_y, eviction_policy="evict_first"
    )


@triton_autotune(
    configs=_get_mul_merge_k_add_bwd_configs(),
    key=["dim_D"],
)
@triton.jit
def _mul_merge_k_add_bwd(
    IDX,  # [dim_N, K]: for each row in DY which K indices from DX to broadcast
    X,  # [dim_N * K, dim_D]
    W,  # [dim_N, K]
    W_INDEX,  # [dim_N * K]
    DY,  # [dim_N, dim_D]
    DX,  # [dim_N * K, dim_D]
    DW,  # [dim_N * K, dim_D]
    dim_N,
    dim_D,
    W_STRIDE: tl.constexpr,
    K: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    r, c, num_c = tl.program_id(1), tl.program_id(0), tl.num_programs(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_D)

    off_y = (
        (r.to(tl.int64) * BLOCK_N + rows[:, None]) * dim_D + c * BLOCK_D + cols[None, :]
    )
    mask_y = (r * BLOCK_N + rows[:, None] < dim_N) & (
        c * BLOCK_D + cols[None, :] < dim_D
    )
    dy = tl.load(DY + off_y, mask=mask_y)

    dim_Nx = dim_N * K
    off_idx = (r * BLOCK_N + rows) * K
    mask_idx = off_idx < dim_Nx

    # broadcast dy to all K elements in dx
    for i in range(K):
        row_idx = tl.load(
            IDX + off_idx + i, mask=mask_idx, other=dim_Nx, eviction_policy="evict_last"
        )
        off_x = row_idx.to(tl.int64)[:, None] * dim_D + c * BLOCK_D + cols[None, :]
        mask_x = (row_idx[:, None] < dim_Nx) & (c * BLOCK_D + cols[None, :] < dim_D)

        if HAS_WEIGHT:
            x = tl.load(X + off_x, mask=mask_x)
            w_index = tl.load(
                W_INDEX + row_idx, mask=row_idx < dim_Nx, eviction_policy="evict_last"
            )
            offs_w = w_index // K * W_STRIDE + w_index % K
            w = tl.load(W + offs_w, mask=row_idx < dim_Nx, eviction_policy="evict_last")

            dx = dy * w[:, None]
            dw = dy * x

            tl.store(
                DW + w_index * num_c + c,
                # accumulate in FP32
                tl.sum(dw.to(tl.float32), axis=1).to(DW.dtype.element_ty),
                mask=row_idx < dim_Nx,
                eviction_policy="evict_first",
            )
        else:
            dx = dy

        tl.store(DX + off_x, dx, mask=mask_x, eviction_policy="evict_first")


class MulMergeKAddFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        index: torch.Tensor,
        x: torch.Tensor,
        k: int,
        weight: Optional[torch.Tensor],
        weight_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = switch_to_contiguous_if_needed(x)
        index = switch_to_contiguous_if_needed(index)

        if weight is not None:
            assert weight_index is not None
            assert x.shape[0] == weight.numel() == weight_index.numel()
            w = switch_to_contiguous_if_needed(weight)
            w_index = switch_to_contiguous_if_needed(weight_index)
            w_stride = weight.stride(dim=0)
        else:
            w = torch.empty((), dtype=x.dtype, device=x.device)
            w_index = torch.empty((), dtype=index.dtype, device=index.device)
            w_stride = 0

        dim_N, dim_D = x.shape[0] // k, x.shape[1]
        has_weight = weight is not None

        y = torch.empty((dim_N, dim_D), dtype=x.dtype, device=x.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(dim_D, meta["BLOCK_D"]),
            triton.cdiv(dim_N, meta["BLOCK_N"]),
        )

        _mul_merge_k_add_fwd[grid](
            index,
            x,
            w,
            w_index,
            y,
            dim_N,
            dim_D,
            w_stride,
            k,
            has_weight,
        )

        ctx.save_for_backward(x, index, w, w_index)
        ctx.k = k
        ctx.has_weight = has_weight
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[
        None,
        torch.Tensor,
        None,
        Optional[torch.Tensor],
        None,
    ]:
        dy = switch_to_contiguous_if_needed(dy)

        x, index, w, w_index = ctx.saved_tensors
        k, has_weight = ctx.k, ctx.has_weight
        if w is not None:
            w_stride = w.stride(dim=0)
        else:
            w_stride = 0

        dim_N, dim_D = dy.shape[0], dy.shape[1]
        # TODO: avoid harding-coding BLOCK_D size
        BLOCK_D = 128

        dx: torch.Tensor = torch.empty_like(x)
        dw_expanded: torch.Tensor = torch.empty(
            (w.numel(), triton.cdiv(dim_D, BLOCK_D)), dtype=w.dtype, device=w.device
        )

        grid = lambda meta: (  # noqa E731
            triton.cdiv(dim_D, BLOCK_D),
            triton.cdiv(dim_N, meta["BLOCK_N"]),
        )
        _mul_merge_k_add_bwd[grid](
            index,
            x,
            w,
            w_index,
            dy,
            dx,
            dw_expanded,
            dim_N,
            dim_D,
            w_stride,
            k,
            has_weight,
            BLOCK_D=BLOCK_D,
        )
        return (
            None,  # index
            dx,
            None,  # k
            (
                dw_expanded.to(dtype=torch.float)
                .sum(dim=1, keepdim=True)
                .to(dtype=w.dtype)
                .view(w.shape)
                if has_weight
                else None
            ),
            None,  # weight_index
        )


@torch.fx.wrap
def triton_index_select_jagged_bmm_3D(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return IndexSelectJaggedBmm3D.apply(
        max_seq_len, offsets, index, jagged, weight, bias
    )


def triton_index_select_jagged_bmm_3D_wrapper(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return triton_index_select_jagged_bmm_3D(
        max_seq_len=max_seq_len,
        offsets=offsets,
        index=index,
        jagged=jagged,
        weight=weight,
        bias=bias,
    )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _jagged_bmm_index_add_3D(
    seq_offsets,
    Index,
    Jagged,
    Dense,
    Out,
    M,
    N,
    K,
    stride_jm,
    stride_jlk,
    stride_je,
    stride_db,
    stride_dk,
    stride_dn,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_n = tl.program_id(2)

    seq_start = tl.load(
        seq_offsets + off_b,
        eviction_policy="evict_last",
    )
    seq_end = tl.load(
        seq_offsets + off_b + 1,
        eviction_policy="evict_last",
    )
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += seq_start.to(tl.int64) * stride_jm
    Dense += off_b * stride_db

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None].to(tl.int64) * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    Index += seq_start
    idx_ptrs = Index + offs_m
    idx = tl.load(idx_ptrs, mask=offs_m < seq_len, other=0)

    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = (
        Out
        + off_b * stride_je
        + idx[:, None].to(tl.int64) * stride_jlk
        + offs_n[None, :]
    )

    tl.store(
        out_ptrs,
        accumulator.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _indexed_jagged_jagged_bmm_reduce_sum_3D(
    seq_offsets,
    Index,
    JaggedA,
    JaggedB,
    Out,
    ReduceOut,
    M,
    N,
    K,
    stride_jl,
    stride_jlk,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
    stride_orb,
    stride_orn,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_m = tl.program_id(1)
    off_n = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Out += off_b * stride_ob
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    out_reduce_ptrs = ReduceOut + off_b * stride_orb + offs_n * stride_orn
    acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if seq_len == 0:
        out = accumulator.to(Out.dtype.element_ty)
        tl.store(
            out_ptrs,
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )
        if off_m == 0:
            tl.store(
                out_reduce_ptrs,  # pyre-ignore [61]
                acc_reduce.to(ReduceOut.dtype.element_ty),
                mask=(offs_n < N),
            )
        return

    Index += seq_start
    offs_k = tl.arange(0, BLOCK_K)
    idx_ptrs = Index + offs_k

    JaggedB += seq_start.to(tl.int64) * stride_bk

    jg_b_ptrs = JaggedB + offs_k[:, None].to(tl.int64) * stride_bk + offs_n[None, :]

    for k in range(0, seq_len, BLOCK_K):
        idx = tl.load(idx_ptrs, mask=(k + offs_k) < seq_len, other=0)
        jg_a_ptrs = (
            JaggedA + idx[None, :] * stride_jl + offs_m[:, None] + off_b * stride_jlk
        )
        jg_a = tl.load(
            jg_a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < M) and ((k + offs_k)[None, :] < seq_len),
            other=0.0,
        )
        jg_b = tl.load(
            jg_b_ptrs,
            mask=(offs_n[None, :] < N) and ((k + offs_k)[:, None] < seq_len),
            other=0.0,
        )

        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if off_m == 0:
            acc_reduce += tl.sum(jg_b, axis=0)

        idx_ptrs += BLOCK_K
        jg_b_ptrs += BLOCK_K * stride_bk

    out = accumulator.to(Out.dtype.element_ty)
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    if off_m == 0:
        tl.store(
            out_reduce_ptrs,  # pyre-ignore [61]
            acc_reduce.to(ReduceOut.dtype.element_ty),
            mask=(offs_n < N),
        )


class IndexSelectJaggedBmm3D(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        offsets: torch.Tensor,
        index: torch.Tensor,
        jagged: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        index = switch_to_contiguous_if_needed(index)
        jagged = switch_to_contiguous_if_needed(jagged)
        if bias is not None:
            bias = switch_to_contiguous_if_needed(bias)

        # L: number of input tokens
        # A: number of activated experts
        # E: number of total experts
        # K: input dimension
        # N: output dimension
        L, A = index.shape
        L, E, K = jagged.shape
        E, _, N = weight.shape
        output = torch.empty((L * A, N), dtype=jagged.dtype, device=jagged.device)

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        _index_select_jagged_bmm_3D[grid](
            seq_offsets=offsets,
            Index=index,
            Jagged=jagged,
            Dense=weight,
            Bias=bias,
            Out=output,
            # M is only used for trigger autotune
            M=triton.next_power_of_2(max_seq_len),
            N=N,
            K=K,
            A=A,
            stride_jl=jagged.stride(0),
            stride_je=jagged.stride(1),
            stride_db=weight.stride(0),
            stride_dk=weight.stride(1),
            stride_dn=weight.stride(2),
            stride_bias_b=bias.stride(0) if bias is not None else 0,
            stride_om=output.stride(0),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            HAS_BIAS=bias is not None,
        )

        ctx.save_for_backward(offsets, index, jagged, weight, bias)
        ctx.E = E
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.N = N

        return output

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[
        None,
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        offsets, index, jagged, weight, bias = ctx.saved_tensors
        E, K, N = ctx.E, ctx.K, ctx.N
        d_jagged_expanded = torch.zeros(
            (jagged.shape[0], index.shape[1], E, K),
            device=jagged.device,
            dtype=torch.float32,
        )
        d_weight = torch.empty_like(weight)
        d_bias = torch.empty_like(bias)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.max_seq_len, meta["BLOCK_M"]),
            E,
            triton.cdiv(K, meta["BLOCK_N"]),
        )

        _jagged_bmm_index_add_3D[grid](
            seq_offsets=offsets,
            Index=index,
            Jagged=d_out,
            Dense=weight,
            Out=d_jagged_expanded,
            # M is only used for triggering autotune
            M=triton.next_power_of_2(ctx.max_seq_len),
            N=K,
            K=N,
            stride_jm=d_out.stride(0),
            stride_jlk=d_jagged_expanded.stride(1),
            stride_je=d_jagged_expanded.stride(2),
            stride_db=weight.stride(0),
            stride_dk=weight.stride(2),
            stride_dn=weight.stride(1),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(K, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _indexed_jagged_jagged_bmm_reduce_sum_3D[grid](
            seq_offsets=offsets,
            Index=index.view(-1) // index.shape[1],
            JaggedA=jagged,
            JaggedB=d_out,
            Out=d_weight,
            ReduceOut=d_bias,
            M=K,
            N=N,
            # K is only used for triggering autotune
            K=triton.next_power_of_2(ctx.max_seq_len),
            stride_jl=jagged.stride(0),
            stride_jlk=jagged.stride(1),
            stride_bk=d_out.stride(0),
            stride_ob=d_weight.stride(0),
            stride_om=d_weight.stride(1),
            stride_on=d_weight.stride(2),
            stride_orb=d_bias.stride(0),
            stride_orn=d_bias.stride(1),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        return (
            None,
            None,
            None,
            d_jagged_expanded.sum(dim=1).to(jagged.dtype),
            d_weight,
            d_bias,
        )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _index_select_jagged_bmm_3D(
    seq_offsets,
    Index,
    Jagged,
    Dense,
    Bias,
    Out,
    M,
    N,
    K,
    A,
    stride_jl,
    stride_je,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_m = tl.program_id(1)
    off_n = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b, eviction_policy="evict_last")
    seq_end = tl.load(seq_offsets + off_b + 1, eviction_policy="evict_last")
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Index += seq_start
    Dense += off_b * stride_db
    Out += seq_start.to(tl.int64) * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    idx_ptrs = Index + offs_m
    idx = tl.load(idx_ptrs, mask=offs_m < seq_len, other=0)
    idx = idx // A

    jg_ptrs = Jagged + idx[:, None] * stride_jl + offs_k[None, :] + off_b * stride_je
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )

        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N, eviction_policy="evict_last")
        accumulator = (accumulator + bias[None, :].to(tl.float32)).to(
            Out.dtype.element_ty
        )

    out_ptrs = Out + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        accumulator,
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _jagged_bmm_reduce_sum(
    seq_offsets,
    JaggedA,
    JaggedB,
    Out,
    ReduceOut,
    M,
    N,
    AUTOTUNE_MAX_SEQ_LEN,
    stride_ak,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
    stride_orb,
    stride_orn,
    REDUCE_JAGGEDB: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computing bmm Out = Jagged x Jagged
    K is the jagged dimension
    JaggedA has shape (sum_B(K_i), M), JaggedB has shape (sum_B(K_i), N), and Out has shape (B, M, N)
    """

    off_b = tl.program_id(0)
    off_m = tl.program_id(1)
    off_n = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Out += off_b.to(tl.int64) * stride_ob
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if REDUCE_JAGGEDB:
        out_reduce_ptrs = ReduceOut + off_b * stride_orb + offs_n * stride_orn
        acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if seq_len == 0:
        out = accumulator.to(Out.dtype.element_ty)
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        if REDUCE_JAGGEDB:
            if off_m == 0:
                tl.store(
                    out_reduce_ptrs,  # pyre-ignore [61]
                    acc_reduce.to(ReduceOut.dtype.element_ty),
                    mask=(offs_n < N),
                )
        return

    JaggedA += seq_start * stride_ak
    JaggedB += seq_start * stride_bk
    offs_k = tl.arange(0, BLOCK_K)
    jg_a_ptrs = JaggedA + offs_k[None, :] * stride_ak + offs_m[:, None]
    jg_b_ptrs = JaggedB + offs_k[:, None] * stride_bk + offs_n[None, :]

    for k in range(0, seq_len, BLOCK_K):
        jg_a = tl.load(
            jg_a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < M) and ((k + offs_k)[None, :] < seq_len),
            other=0.0,
        )
        jg_b = tl.load(
            jg_b_ptrs,
            mask=(offs_n[None, :] < N) and ((k + offs_k)[:, None] < seq_len),
            other=0.0,
        )

        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if REDUCE_JAGGEDB:
            if off_m == 0:
                acc_reduce += tl.sum(jg_b.to(tl.float32), axis=0)

        jg_a_ptrs += BLOCK_K * stride_ak
        jg_b_ptrs += BLOCK_K * stride_bk

    out = accumulator.to(Out.dtype.element_ty)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    if REDUCE_JAGGEDB:
        if off_m == 0:
            tl.store(
                out_reduce_ptrs,  # pyre-ignore [61]
                acc_reduce.to(ReduceOut.dtype.element_ty),
                mask=(offs_n < N),
            )
