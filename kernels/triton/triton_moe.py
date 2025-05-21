#!/usr/bin/env python3

# pyre-unsafe

import dataclasses
from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from fast_moe.kernels.triton.grouped_gemm import grouped_gemm
from fast_moe.kernels.triton.triton_general_ops import (
    triton_index_select,
    triton_silu_backward,
    triton_sum_dim1,
)
from fast_moe.kernels.triton.utils import (
    fast_sigmoid,
    get_bmm_configs,
    get_bmm_split_k_configs,
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from torch.autograd.profiler import record_function


@dataclasses.dataclass
class IndexSelectJaggedBmmOption:
    d_jagged_use_grouped_gemm: bool = True
    d_jagged_use_wrap_specialization: bool = True
    d_jagged_gemm_out_type: torch.dtype = torch.float32
    d_weight_optimization: bool = True
    d_weight_split_k_kernel: bool = False
    d_weight_split_k_kernel_tma: bool = False


@torch.fx.wrap
def triton_index_select_jagged_bmm(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    option: Optional[IndexSelectJaggedBmmOption] = None,
) -> torch.Tensor:
    return IndexSelectJaggedBmm.apply(
        max_seq_len, offsets, index, jagged, weight, bias, option
    )


def triton_index_select_jagged_bmm_wrapper(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    option: Optional[IndexSelectJaggedBmmOption] = None,
) -> torch.Tensor:
    return triton_index_select_jagged_bmm(
        max_seq_len=max_seq_len,
        offsets=offsets,
        index=index,
        jagged=jagged,
        weight=weight,
        bias=bias,
        option=option,
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
        option: Optional[IndexSelectJaggedBmmOption],
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
        if option is None:
            option = IndexSelectJaggedBmmOption()
        if not torch.cuda.get_device_capability() >= (9, 0):
            # TMA and warp specialization are only supported on Hopper and above
            option.d_jagged_use_wrap_specialization = False
            option.d_weight_split_k_kernel_tma = False

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
        ctx.L = L
        ctx.E = E
        ctx.A = A
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.N = N
        ctx.option = option

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
        None,
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
                - None: No gradient is computed for `option`.
        """
        # offsets: [E + 1]
        # index: [L, A]
        # jagged: [L, K]
        # weight: [E, K, N]
        # bias: [E, N]
        offsets, index, jagged, weight, bias = ctx.saved_tensors
        L, E, A, K, N = ctx.L, ctx.E, ctx.A, ctx.K, ctx.N
        max_seq_len = ctx.max_seq_len
        option: IndexSelectJaggedBmmOption = ctx.option

        with record_function("#### d_jagged ####"):
            d_jagged = IndexSelectJaggedBmm._calc_d_jagged(
                weight, d_out, index, offsets, L, E, A, K, N, max_seq_len, option
            )

        with record_function("#### d_weight ####"):
            d_weight, d_bias = IndexSelectJaggedBmm._calc_d_weight_bias(
                weight,
                bias,
                offsets,
                index,
                d_out,
                jagged,
                E,
                K,
                N,
                max_seq_len,
                option,
            )

        return (
            None,
            None,
            None,
            d_jagged,
            d_weight,
            d_bias,
            None,
        )

    @staticmethod
    def _calc_d_jagged(
        weight: torch.Tensor,
        d_out: torch.Tensor,
        index: torch.Tensor,
        offsets: torch.Tensor,
        L: int,
        E: int,
        A: int,
        K: int,
        N: int,
        max_seq_len: int,
        option: IndexSelectJaggedBmmOption,
    ) -> torch.Tensor:
        if option.d_jagged_use_grouped_gemm:
            assert N % 64 == 0, "N must be a multiple of 64"
            m_sizes = (offsets[1:] - offsets[:-1]).to(torch.int32)  # [E]
            weight_grouped = weight.reshape(-1, N)  # [E, K, N] -> [E * K, N]
            d_jagged_expanded = grouped_gemm(
                x=d_out,  # [L * A, N]
                w=weight_grouped,  # [E * K, N]
                m_sizes=m_sizes,  # [E]
                use_fast_accum=False,
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                _use_warp_specialization=option.d_jagged_use_wrap_specialization,
                _out_type=option.d_jagged_gemm_out_type,
                _out_index=index.flatten(),
            )  # [L * A, K]
            d_jagged_expanded = d_jagged_expanded.view((L, A, K))  # [L, A, K]
            assert d_jagged_expanded.dtype == option.d_jagged_gemm_out_type
        else:
            d_jagged_expanded = torch.empty(
                (L, A, K),
                device=d_out.device,
                dtype=option.d_jagged_gemm_out_type,
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
        d_jagged = triton_sum_dim1(d_jagged_expanded)
        return d_jagged

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
        option: IndexSelectJaggedBmmOption,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if option.d_weight_optimization:
            jagged_a = triton_index_select(
                jagged, index.flatten() // index.shape[1]
            )  # [L * A, K]
            if option.d_weight_split_k_kernel or option.d_weight_split_k_kernel_tma:
                d_weight, d_bias = triton_jagged_bmm_reduce_sum_split_k(
                    JaggedA=jagged_a,  # [L * A, K]
                    JaggedB=d_out,  # [L * A, N]
                    offsets=offsets,  # [E + 1]
                    reduce_sum=bias is not None,
                    use_tma=option.d_weight_split_k_kernel_tma,
                )  # [E, K, N], [E, N]
            else:
                d_weight, d_bias = triton_jagged_bmm_reduce_sum(
                    JaggedA=jagged_a,  # [L * A, K]
                    JaggedB=d_out,  # [L * A, N]
                    offsets=offsets,  # [E + 1]
                    reduce_sum=bias is not None,
                )  # [E, K, N], [E, N]
        else:
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
    key=["M", "N", "AUTOTUNE_K"],
)
@triton.jit
def _jagged_jagged_bmm(
    seq_offsets,
    JaggedA,
    JaggedB,
    Out,
    M,
    N,
    AUTOTUNE_K,
    stride_ak,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
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

    off_m = tl.program_id(0)
    off_n = tl.program_id(1)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Out += off_b.to(tl.int64) * stride_ob
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[None, :] * stride_on + offs_n[:, None] * stride_om

    JaggedA += seq_start * stride_ak
    JaggedB += seq_start * stride_bk
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)
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

        jg_a_ptrs += BLOCK_K * stride_ak
        jg_b_ptrs += BLOCK_K * stride_bk

    out = accumulator.to(Out.dtype.element_ty)
    tl.store(out_ptrs, out.T, mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton_autotune(
    configs=get_bmm_split_k_configs(),
    key=["M", "N", "AUTOTUNE_K", "USE_TMA"],
    restore_value=["Out"],
)
@triton.jit
def _jagged_jagged_bmm_split_k(
    seq_offsets,
    JaggedA,
    JaggedB,
    Out,
    M,
    N,
    AUTOTUNE_K,
    stride_ak,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
    workspace_ptr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    """
    Computing bmm Out = Jagged x Jagged
    K is the jagged dimension
    JaggedA has shape (sum_B(K_i), M), JaggedB has shape (sum_B(K_i), N), and Out has shape (B, M, N)
    """
    off_k = tl.program_id(0)

    off_mn = tl.program_id(1)
    num_n_blocks = cdiv_fn(N, BLOCK_N)
    off_m = off_mn // num_n_blocks
    off_n = off_mn % num_n_blocks

    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    num_k_blocks = cdiv_fn(seq_len, BLOCK_K)
    num_k_blocks_per_split = cdiv_fn(num_k_blocks, SPLIT_K)
    k_start = off_k * num_k_blocks_per_split * BLOCK_K

    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    Out += off_b.to(tl.int64) * stride_ob
    JaggedA += seq_start * stride_ak
    JaggedB += seq_start * stride_bk

    actual_num_k_blocks_per_split = num_k_blocks_per_split
    if off_k == tl.num_programs(0) - 1:
        actual_num_k_blocks_per_split = min(
            num_k_blocks - off_k * num_k_blocks_per_split, num_k_blocks_per_split
        )

    desc_a, desc_b, desc_c = None, None, None
    offs_m, offs_n, offs_k, out_ptrs = None, None, None, None

    if USE_TMA:
        tile_idx = (
            off_k
            + off_mn * tl.num_programs(0)
            + off_b * tl.num_programs(0) * tl.num_programs(1)
        )

        TMA_SIZE: tl.constexpr = tl.constexpr(128)

        workspace_base = workspace_ptr + tile_idx * TMA_SIZE * 3
        desc_a = workspace_base
        desc_b = workspace_base + TMA_SIZE  # + TMA_SIZE
        desc_c = workspace_base + 2 * TMA_SIZE

        # pyre-ignore
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=desc_a,
            global_address=JaggedA,
            load_size=[BLOCK_K, BLOCK_M],
            global_size=[seq_len.to(tl.int32), M],
            element_ty=JaggedA.dtype.element_ty,
        )
        # pyre-ignore
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=desc_b,
            global_address=JaggedB,
            load_size=[BLOCK_K, BLOCK_N],
            global_size=[seq_len.to(tl.int32), N],
            element_ty=JaggedB.dtype.element_ty,
        )
        # pyre-ignore
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=desc_c,
            global_address=Out,
            load_size=[BLOCK_M, BLOCK_N],
            global_size=[M, N],
            element_ty=Out.dtype.element_ty,
        )
        # pyre-ignore
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_a)
        # pyre-ignore
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_b)
        # pyre-ignore
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(desc_c)

    else:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

        offs_k = tl.arange(0, BLOCK_K)
        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
        offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    for k in range(0, actual_num_k_blocks_per_split):
        if USE_TMA:
            jg_a = tl._experimental_descriptor_load(
                desc_a,
                [
                    (k_start + k * BLOCK_K).to(tl.int32),
                    (start_m).to(tl.int32),
                ],
                [BLOCK_K, BLOCK_M],
                JaggedA.dtype.element_ty,
            )
            jg_b = tl._experimental_descriptor_load(
                desc_b,
                [
                    (k_start + k * BLOCK_K).to(tl.int32),
                    (start_n).to(tl.int32),
                ],
                [BLOCK_K, BLOCK_N],
                JaggedB.dtype.element_ty,
            )
        else:
            offs_k_block = k_start + k * BLOCK_K + offs_k[:, None]
            jg_a = tl.load(
                JaggedA + offs_k_block * stride_ak + offs_m[None, :],
                mask=(offs_m[None, :] < M) and (offs_k_block < seq_len),
                other=0.0,
            )
            jg_b = tl.load(
                JaggedB + offs_k_block * stride_bk + offs_n[None, :],
                mask=(offs_n[None, :] < N) and (offs_k_block < seq_len),
                other=0.0,
            )

        accumulator += tl.dot(jg_a.T, jg_b, allow_tf32=ALLOW_TF32)

    out = accumulator.to(Out.dtype.element_ty)

    if USE_TMA:
        tl._experimental_descriptor_store(
            desc_c,
            out,
            [
                start_m.to(tl.int32),
                start_n.to(tl.int32),
            ],
            store_reduce="add",
        )
    else:
        tl.atomic_add(
            out_ptrs,
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            sem="relaxed",
        )


@triton_autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_M in [32, 64, 128]
        for BLOCK_N in [32, 64, 128]
        for num_stages in [2, 3]
        for num_warps in [4, 8]
    ],
    key=["M", "N"],
)
@triton.jit
def _jagged_reduce_sum(
    seq_offsets,
    Jagged,
    ReduceOut,
    M,
    N,
    stride_jm,
    stride_jn,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_m = tl.program_id(1)

    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M

    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = start_m + tl.arange(0, BLOCK_M)

    offs_n = tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    jg_ptrs = (
        Jagged
        + seq_start.to(tl.int64) * stride_jn
        + offs_m.to(tl.int64)[:, None] * stride_jm
        + offs_n[None, :].to(tl.int64) * stride_jn,
    )

    out_reduce_ptrs = ReduceOut + off_b * M + offs_m

    for k in range(0, seq_len, BLOCK_N):
        jg = tl.load(
            jg_ptrs,
            mask=(offs_m[:, None] < M) and ((offs_n[None, :] + k) < seq_len),
            other=0.0,
        )

        accum += jg
        jg_ptrs += BLOCK_N * stride_jn

    _accum = tl.sum(accum, axis=1)
    out = _accum.to(ReduceOut.dtype.element_ty)

    tl.store(out_reduce_ptrs, out, mask=(offs_m[:] < M))


@triton_autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "SPLIT_K": SPLIT_K,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_M in [32, 64, 128]
        for BLOCK_N in [32, 64, 128]
        for SPLIT_K in [1, 2, 4, 8, 16]
        for num_stages in [2, 3]
        for num_warps in [4, 8]
    ],
    key=["M", "N"],
    restore_value=["ReduceOut"],
)
@triton.jit
def _jagged_reduce_sum_split_k(
    seq_offsets,
    Jagged,
    ReduceOut,
    M,
    N,
    stride_jm,
    stride_jn,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    off_k = tl.program_id(0)
    off_m = tl.program_id(1)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M

    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_m = start_m + tl.arange(0, BLOCK_M)

    offs_n = tl.arange(0, BLOCK_N)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    jg_ptrs = (
        Jagged
        + seq_start.to(tl.int64) * stride_jn
        + offs_m.to(tl.int64)[:, None] * stride_jm
        + offs_n[None, :].to(tl.int64) * stride_jn,
    )

    out_reduce_ptrs = ReduceOut + off_b * M + offs_m

    num_k_blocks = cdiv_fn(seq_len, BLOCK_N)
    num_k_blocks_per_split = cdiv_fn(num_k_blocks, SPLIT_K)
    k_start = off_k * num_k_blocks_per_split * BLOCK_N

    actual_num_k_blocks_per_split = num_k_blocks_per_split
    if off_k == tl.num_programs(0) - 1:
        actual_num_k_blocks_per_split = min(
            num_k_blocks - off_k * num_k_blocks_per_split, num_k_blocks_per_split
        )

    jg_ptrs += k_start * stride_jn
    # for k in range(0, seq_len, BLOCK_N):
    for k in range(0, actual_num_k_blocks_per_split):
        jg = tl.load(
            jg_ptrs,
            mask=(offs_m[:, None] < M) and ((offs_n[None, :] + k * BLOCK_N) < seq_len),
            other=0.0,
        )

        accum += jg
        jg_ptrs += BLOCK_N * stride_jn

    _accum = tl.sum(accum, axis=1)
    out = _accum.to(ReduceOut.dtype.element_ty)

    tl.atomic_add(out_reduce_ptrs, out, mask=(offs_m[:] < M))


def triton_jagged_bmm_reduce_sum(
    JaggedA: torch.Tensor,  # (sum_B(K_i), M)
    JaggedB: torch.Tensor,  # (sum_B(K_i), N)
    offsets: torch.Tensor,  # (B+1)
    reduce_sum: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # d_weight: (B, M, N), d_bias: (B, N)
    dtype = JaggedA.dtype
    device = JaggedA.device
    B = offsets.shape[0] - 1
    M = JaggedA.shape[1]
    N = JaggedB.shape[1]
    d_weight = torch.empty((B, M, N), dtype=dtype, device=device)
    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        B,
    )
    _jagged_jagged_bmm[grid](
        seq_offsets=offsets,
        JaggedA=JaggedA,
        JaggedB=JaggedB,
        Out=d_weight,
        M=M,
        N=N,
        # To avoid use max_seq_len here, we use total seq length as the max_seq_len
        AUTOTUNE_K=triton.next_power_of_2(JaggedA.shape[0]),
        stride_ak=JaggedA.stride(0),
        stride_bk=JaggedB.stride(0),
        stride_ob=d_weight.stride(0),
        stride_om=d_weight.stride(2),
        stride_on=d_weight.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    d_bias = None
    if reduce_sum:
        d_bias = torch.empty((B, N), dtype=dtype, device=device)
        grid = lambda meta: (  # noqa E731
            B,
            triton.cdiv(N, meta["BLOCK_M"]),
        )
        _jagged_reduce_sum[grid](
            seq_offsets=offsets,
            Jagged=JaggedB,
            ReduceOut=d_bias,
            M=N,
            N=triton.next_power_of_2(JaggedB.shape[0]),
            stride_jm=JaggedB.stride(1),
            stride_jn=JaggedB.stride(0),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )
    return d_weight, d_bias


@torch.fx.wrap
def triton_index_select_jagged_bmm_swiglu(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_p: torch.Tensor,
    bias_p: Optional[torch.Tensor],
) -> torch.Tensor:
    return IndexSelectJaggedBmmSwiglu.apply(
        max_seq_len,
        offsets,
        index,
        jagged,
        weight,
        bias,
        weight_p,
        bias_p,
    )


def triton_index_select_jagged_bmm_swiglu_wrapper(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_p: torch.Tensor,
    bias_p: Optional[torch.Tensor],
) -> torch.Tensor:
    return triton_index_select_jagged_bmm_swiglu(
        max_seq_len=max_seq_len,
        offsets=offsets,
        index=index,
        jagged=jagged,
        weight=weight,
        bias=bias,
        weight_p=weight_p,
        bias_p=bias_p,
    )


class IndexSelectJaggedBmmSwiglu(torch.autograd.Function):
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
        weight_p: torch.Tensor,
        bias_p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            max_seq_len (int): Maximum number of input tokens for any expert.
            offsets (torch.Tensor): A tensor of shape [E + 1] representing the cumulative number of tokens dispatched to each expert.
            index (torch.Tensor): A tensor of shape [L, A] that is flattened and sorted by expert.
            jagged (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
            weight (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
            bias (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
            weight_p (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
            bias_p (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
        Returns:
            torch.Tensor: A tensor of shape [L * A, N] containing the output after applying the linear transformation.
        """
        index = switch_to_contiguous_if_needed(index)
        jagged = switch_to_contiguous_if_needed(jagged)
        if bias is not None:
            bias = switch_to_contiguous_if_needed(bias)
        if bias_p is not None:
            bias_p = switch_to_contiguous_if_needed(bias_p)

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
        alpha = torch.empty(
            (L * A, N), dtype=jagged.dtype, device=jagged.device
        )  # [L * A, N]
        alpha_silu = torch.empty(
            (L * A, N), dtype=jagged.dtype, device=jagged.device
        )  # [L * A, N]
        beta = torch.empty(
            (L * A, N), dtype=jagged.dtype, device=jagged.device
        )  # [L * A, N]

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        _index_select_jagged_bmm_swiglu[grid](
            seq_offsets=offsets,
            Index=index,
            Jagged=jagged,
            Dense=weight,
            Bias=bias,
            Dense_P=weight_p,
            Bias_P=bias_p,
            Out=output,
            ALPHA=alpha,
            ALPHA_SILU=alpha_silu,
            BETA=beta,
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

        ctx.save_for_backward(
            offsets,
            index,
            jagged,
            weight,
            bias,
            weight_p,
            bias_p,
            alpha,
            alpha_silu,
            beta,
        )
        ctx.E = E
        ctx.max_seq_len = max_seq_len
        ctx.L = L
        ctx.A = A
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
                - torch.Tensor: Gradient with respect to `weight_p`, of shape [E, K, N].
                - torch.Tensor: Gradient with respect to `bias_p`, of shape [E, N].
        """
        # offsets: [E + 1]
        # index: [L, A]
        # jagged: [L, K]
        # weight: [E, K, N]
        # bias: [E, N]
        # weight_p: [E, K, N]
        # bias_p: [E, N]
        # a: [L * A, N]
        # beta: [L * A, N]
        offsets, index, jagged, weight, bias, weight_p, bias_p, a, alpha, beta = (
            ctx.saved_tensors
        )
        _, K, L, A, N = ctx.E, ctx.K, ctx.L, ctx.A, ctx.N
        has_bias = bias is not None

        d_alpha = d_out * beta
        d_beta = d_out * alpha

        d_a = triton_silu_backward(a, d_alpha)

        d_jagged_expanded_1 = torch.empty(
            (jagged.shape[0], index.shape[1], jagged.shape[1]),
            device=jagged.device,
            dtype=torch.float32,
        )  # [L, A, K]
        d_jagged_expanded_2 = torch.empty(
            (jagged.shape[0], index.shape[1], jagged.shape[1]),
            device=jagged.device,
            dtype=torch.float32,
        )  # [L, A, K]
        # tensors below needs to be initialized with zeros as there could be unused
        # rows in the weight and bias
        d_weight = torch.zeros_like(weight)  # [E, K, N]
        d_weight_p = torch.zeros_like(weight_p)  # [E, K, N]
        if has_bias:
            d_bias = torch.zeros_like(bias)  # [E, N]
            d_bias_p = torch.zeros_like(bias_p)  # [E, N]
        else:
            d_bias = None
            d_bias_p = None

        jagged_a = triton_index_select(
            jagged, index.flatten() // index.shape[1]
        )  # [L * A, K]
        d_weight, d_bias = triton_jagged_bmm_reduce_sum(
            JaggedA=jagged_a,  # [L * A, K]
            JaggedB=d_a,  # [L * A, N]
            offsets=offsets,  # [E + 1]
            reduce_sum=bias is not None,
        )  # [E, K, N], [E, N]

        d_weight_p, d_bias_p = triton_jagged_bmm_reduce_sum(
            JaggedA=jagged_a,  # [L * A, K]
            JaggedB=d_beta,  # [L * A, N]
            offsets=offsets,  # [E + 1]
            reduce_sum=bias_p is not None,
        )  # [E, K, N], [E, N]

        assert N % 64 == 0, "N must be a multiple of 64"
        m_sizes = (offsets[1:] - offsets[:-1]).to(torch.int32)  # [E]
        weight_grouped = weight.reshape(-1, N)  # [E, K, N] -> [E * K, N]
        weight_p_grouped = weight_p.reshape(-1, N)  # [E, K, N] -> [E * K, N]
        d_jagged_expanded_1 = grouped_gemm(
            x=d_a,  # [L * A, N]
            w=weight_grouped,  # [E * K, N]
            m_sizes=m_sizes,  # [E]
            use_fast_accum=False,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            _use_warp_specialization=True,
            _out_type=jagged.dtype,
            _out_index=index.flatten(),
        )  # [L * A, K]
        d_jagged_expanded_1 = d_jagged_expanded_1.view((L, A, K))  # [L, A, K]

        d_jagged_expanded_2 = grouped_gemm(
            x=d_beta,  # [L * A, N]
            w=weight_p_grouped,  # [E * K, N]
            m_sizes=m_sizes,  # [E]
            use_fast_accum=False,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            _use_warp_specialization=True,
            _out_type=jagged.dtype,
            _out_index=index.flatten(),
        )  # [L * A, K]
        d_jagged_expanded_2 = d_jagged_expanded_2.view((L, A, K))  # [L, A, K]

        d_jagged = triton_sum_dim1(d_jagged_expanded_1) + triton_sum_dim1(
            d_jagged_expanded_2
        )

        return (
            None,
            None,
            None,
            d_jagged.to(jagged.dtype),
            d_weight,
            d_bias,
            d_weight_p,
            d_bias_p,
        )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _index_select_jagged_bmm_swiglu(
    seq_offsets,  # [B+1]
    Index,  # [Sum_B(M)], jagged indices in range [0, L * A)
    Jagged,  # [L, K]
    Dense,  # [B, K, N]
    Bias,  # [B, N]
    Dense_P,  # [B, K, N]
    Bias_P,  # [B, N]
    Out,  # [Sum_B(M), N]
    ALPHA,  # [Sum_B(M), N]
    ALPHA_SILU,  # [Sum_B(M), N]
    BETA,  # [Sum_B(M), N]
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
    SWiGlu(Swish-Gated Linear Unit) baseded on index_select_jagged_bmm

    Out = Silu(Jagged @ Dense + Bias) * (Jagged @ Dense_P + Bias_P)

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
    Dense_P += off_b * stride_db
    Out += seq_start.to(tl.int64) * stride_om
    ALPHA += seq_start.to(tl.int64) * stride_om
    ALPHA_SILU += seq_start.to(tl.int64) * stride_om
    BETA += seq_start.to(tl.int64) * stride_om

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
    # [BLOCK_K, BLOCK_N]
    dnp_ptrs = Dense_P + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
    accumulator2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
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
        dnp = tl.load(
            dnp_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )  # [BLOCK_K, BLOCK_N]

        acc1 = tl.dot(jg, dn, allow_tf32=ALLOW_TF32)  # [BLOCK_M, BLOCK_N]
        acc2 = tl.dot(jg, dnp, allow_tf32=ALLOW_TF32)  # [BLOCK_M, BLOCK_N]
        accumulator1 += acc1
        accumulator2 += acc2
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk
        dnp_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        # load bias
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n  # [BLOCK_N]
        biasp_ptrs = Bias_P + off_b * stride_bias_b + offs_n  # [BLOCK_N]

        bias = tl.load(
            bias_ptrs, mask=offs_n < N, eviction_policy="evict_last"
        )  # [BLOCK_N]
        biasp = tl.load(
            biasp_ptrs, mask=offs_n < N, eviction_policy="evict_last"
        )  # [BLOCK_N]

        # add bias to accumulator [BLOCK_M, BLOCK_N]
        A = accumulator1 + bias[None, :].to(tl.float32)
        B = accumulator2 + biasp[None, :].to(tl.float32)
    else:
        A = accumulator1
        B = accumulator2

    # Apply Silu to A
    a_sigmoid = fast_sigmoid(A)
    A_SILU = A * a_sigmoid

    out = (A_SILU * B).to(Out.dtype.element_ty)

    # write back [BLOCK_M, BLOCK_N]
    out_ptrs = Out + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    alpha_ptrs = ALPHA + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    alpha_silu_ptrs = (
        ALPHA_SILU + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    )
    beta_ptrs = BETA + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )
    tl.store(
        alpha_ptrs,
        A.to(ALPHA.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )
    tl.store(
        alpha_silu_ptrs,
        A_SILU.to(ALPHA_SILU.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )

    tl.store(
        beta_ptrs,
        B.to(BETA.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )


@torch.fx.wrap
def triton_index_select_jagged_gating_bmm(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged_a: torch.Tensor,
    jagged_b: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    bias_a: Optional[torch.Tensor],
    bias_b: Optional[torch.Tensor],
) -> torch.Tensor:
    return IndexSelectJaggedGatingBmm.apply(
        max_seq_len,
        offsets,
        index,
        jagged_a,
        jagged_b,
        weight_a,
        weight_b,
        bias_a,
        bias_b,
    )


def triton_index_select_jagged_gating_bmm_wrapper(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged_a: torch.Tensor,
    jagged_b: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    bias_a: Optional[torch.Tensor],
    bias_b: Optional[torch.Tensor],
) -> torch.Tensor:
    return triton_index_select_jagged_gating_bmm(
        max_seq_len=max_seq_len,
        offsets=offsets,
        index=index,
        jagged_a=jagged_a,
        jagged_b=jagged_b,
        weight_a=weight_a,
        weight_b=weight_b,
        bias_a=bias_a,
        bias_b=bias_b,
    )


class IndexSelectJaggedGatingBmm(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        offsets: torch.Tensor,
        index: torch.Tensor,
        jagged_a: torch.Tensor,
        jagged_b: torch.Tensor,
        weight_a: torch.Tensor,
        weight_b: torch.Tensor,
        bias_a: Optional[torch.Tensor],
        bias_b: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            max_seq_len (int): Maximum number of input tokens for any expert.
            offsets (torch.Tensor): A tensor of shape [E + 1] representing the cumulative number of tokens dispatched to each expert.
            index (torch.Tensor): A tensor of shape [L, A] that is flattened and sorted by expert.
            jagged_a (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
            jagged_b (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
            weight_a (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
            weight_b (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
            bias_a (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
            bias_b (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
        Returns:
            torch.Tensor: A tensor of shape [L * A, N] containing the output after applying the linear transformation.
        """
        index = switch_to_contiguous_if_needed(index)
        jagged_a = switch_to_contiguous_if_needed(jagged_a)
        jagged_b = switch_to_contiguous_if_needed(jagged_b)
        if bias_a is not None:
            bias_a = switch_to_contiguous_if_needed(bias_a)
        if bias_b is not None:
            bias_b = switch_to_contiguous_if_needed(bias_b)

        # L: number of input tokens
        # A: number of activated experts
        # E: number of total experts
        # K: input dimension
        # N: output dimension
        L, A = index.shape
        _, K = jagged_a.shape
        E, _, N = weight_a.shape
        output = torch.empty(
            (L * A, N), dtype=jagged_a.dtype, device=jagged_a.device
        )  # [L * A, N]
        alpha = torch.empty(
            (L * A, N), dtype=jagged_a.dtype, device=jagged_a.device
        )  # [L * A, N]
        alpha_silu = torch.empty(
            (L * A, N), dtype=jagged_a.dtype, device=jagged_a.device
        )  # [L * A, N]
        beta = torch.empty(
            (L * A, N), dtype=jagged_b.dtype, device=jagged_b.device
        )  # [L * A, N]

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        _index_select_jagged_gating_bmm[grid](
            seq_offsets=offsets,
            Index=index,
            Jagged_A=jagged_a,
            Jagged_B=jagged_b,
            Dense_A=weight_a,
            Dense_B=weight_b,
            Bias_A=bias_a,
            Bias_B=bias_b,
            Out=output,
            ALPHA=alpha,
            ALPHA_SILU=alpha_silu,
            BETA=beta,
            # M is only used for trigger autotune
            M=triton.next_power_of_2(max_seq_len),
            N=N,
            K=K,
            A=A,
            stride_jm=jagged_a.stride(0),
            stride_db=weight_a.stride(0),
            stride_dk=weight_a.stride(1),
            stride_dn=weight_a.stride(2),
            stride_bias_b=bias_a.stride(0) if bias_a is not None else 0,
            stride_om=output.stride(0),
            HAS_BIAS=bias_a is not None,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        ctx.save_for_backward(
            offsets,
            index,
            jagged_a,
            jagged_b,
            weight_a,
            weight_b,
            bias_a,
            bias_b,
            alpha,
            alpha_silu,
            beta,
        )
        ctx.E = E
        ctx.max_seq_len = max_seq_len
        ctx.L = L
        ctx.A = A
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
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
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
                - torch.Tensor: Gradient with respect to `jagged_a`, of shape [L, K].
                - torch.Tensor: Gradient with respect to `jagged_b`, of shape [L, K].
                - torch.Tensor: Gradient with respect to `weight_a`, of shape [E, K, N].
                - torch.Tensor: Gradient with respect to `weight_b`, of shape [E, K, N].
                - torch.Tensor: Gradient with respect to `bias_a`, of shape [E, N].
                - torch.Tensor: Gradient with respect to `bias_b`, of shape [E, N].
        """
        # offsets: [E + 1]
        # index: [L, A]
        # jagged_a: [L, K]
        # jagged_b: [L, K]
        # weight_a: [E, K, N]
        # bias_a: [E, N]
        # weight_b: [E, K, N]
        # bias_b: [E, N]
        # a: [L * A, N]
        # beta: [L * A, N]
        (
            offsets,
            index,
            jagged_a,
            jagged_b,
            weight_a,
            weight_b,
            bias_a,
            bias_b,
            a,
            alpha,
            beta,
        ) = ctx.saved_tensors
        _, K, L, A, N = ctx.E, ctx.K, ctx.L, ctx.A, ctx.N
        has_bias = bias_a is not None

        d_alpha = d_out * beta
        d_beta = d_out * alpha

        d_a = triton_silu_backward(a, d_alpha)

        d_jagged_expanded_1 = torch.empty(
            (jagged_a.shape[0], index.shape[1], jagged_a.shape[1]),
            device=jagged_a.device,
            dtype=torch.float32,
        )  # [L, A, K]
        d_jagged_expanded_2 = torch.empty(
            (jagged_b.shape[0], index.shape[1], jagged_b.shape[1]),
            device=jagged_b.device,
            dtype=torch.float32,
        )  # [L, A, K]
        # tensors below needs to be initialized with zeros as there could be unused
        # rows in the weight and bias
        d_weight_a = torch.zeros_like(weight_a)  # [E, K, N]
        d_weight_b = torch.zeros_like(weight_b)  # [E, K, N]
        if has_bias:
            d_bias_a = torch.zeros_like(bias_a)  # [E, N]
            d_bias_b = torch.zeros_like(bias_b)  # [E, N]
        else:
            d_bias_a = None
            d_bias_b = None

        jagged_a_selected = triton_index_select(
            jagged_a, index.flatten() // index.shape[1]
        )  # [L * A, K]
        jagged_b_selected = triton_index_select(
            jagged_b, index.flatten() // index.shape[1]
        )  # [L * A, K]
        d_weight_a, d_bias_a = triton_jagged_bmm_reduce_sum(
            JaggedA=jagged_a_selected,  # [L * A, K]
            JaggedB=d_a,  # [L * A, N]
            offsets=offsets,  # [E + 1]
            reduce_sum=bias_a is not None,
        )  # [E, K, N], [E, N]

        d_weight_b, d_bias_b = triton_jagged_bmm_reduce_sum(
            JaggedA=jagged_b_selected,  # [L * A, K]
            JaggedB=d_beta,  # [L * A, N]
            offsets=offsets,  # [E + 1]
            reduce_sum=bias_b is not None,
        )  # [E, K, N], [E, N]

        assert N % 64 == 0, "N must be a multiple of 64"
        m_sizes = (offsets[1:] - offsets[:-1]).to(torch.int32)  # [E]
        weight_a_grouped = weight_a.reshape(-1, N)  # [E, K, N] -> [E * K, N]
        weight_b_grouped = weight_b.reshape(-1, N)  # [E, K, N] -> [E * K, N]
        d_jagged_expanded_1 = grouped_gemm(
            x=d_a,  # [L * A, N]
            w=weight_a_grouped,  # [E * K, N]
            m_sizes=m_sizes,  # [E]
            use_fast_accum=False,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            _use_warp_specialization=True,
            _out_type=jagged_a.dtype,
            _out_index=index.flatten(),
        )  # [L * A, K]
        d_jagged_expanded_1 = d_jagged_expanded_1.view((L, A, K))  # [L, A, K]

        d_jagged_expanded_2 = grouped_gemm(
            x=d_beta,  # [L * A, N]
            w=weight_b_grouped,  # [E * K, N]
            m_sizes=m_sizes,  # [E]
            use_fast_accum=False,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            _use_warp_specialization=True,
            _out_type=jagged_b.dtype,
            _out_index=index.flatten(),
        )  # [L * A, K]
        d_jagged_expanded_2 = d_jagged_expanded_2.view((L, A, K))  # [L, A, K]

        d_jagged_a = triton_sum_dim1(d_jagged_expanded_1)
        d_jagged_b = triton_sum_dim1(d_jagged_expanded_2)

        return (
            None,
            None,
            None,
            d_jagged_a.to(jagged_a.dtype),
            d_jagged_b.to(jagged_b.dtype),
            d_weight_a,
            d_weight_b,
            d_bias_a,
            d_bias_b,
        )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _index_select_jagged_gating_bmm(
    seq_offsets,  # [B+1]
    Index,  # [Sum_B(M)], jagged indices in range [0, L * A)
    Jagged_A,  # [L, K]
    Jagged_B,  # [L, K]
    Dense_A,  # [B, K, N]
    Dense_B,  # [B, K, N]
    Bias_A,  # [B, N]
    Bias_B,  # [B, N]
    Out,  # [Sum_B(M), N]
    ALPHA,  # [Sum_B(M), N]
    ALPHA_SILU,  # [Sum_B(M), N]
    BETA,  # [Sum_B(M), N]
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
    index_select_jagged_bmm with gating

    Out = Silu(Jagged_A @ Dense_A + Bias_A) * (Jagged_B @ Dense_B + Bias_B)

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
    Dense_A += off_b * stride_db
    Dense_B += off_b * stride_db
    Out += seq_start.to(tl.int64) * stride_om
    ALPHA += seq_start.to(tl.int64) * stride_om
    ALPHA_SILU += seq_start.to(tl.int64) * stride_om
    BETA += seq_start.to(tl.int64) * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    # load index for all rows to be processed by this block
    idx_ptrs = Index + offs_m
    idx = tl.load(idx_ptrs, mask=offs_m < seq_len, other=0)  # [BLOCK_M]
    idx = idx // A

    # [BLOCK_M, BLOCK_K]
    jg_a_ptrs = Jagged_A + idx[:, None] * stride_jm + offs_k[None, :]
    # [BLOCK_M, BLOCK_K]
    jg_b_ptrs = Jagged_B + idx[:, None] * stride_jm + offs_k[None, :]
    # [BLOCK_K, BLOCK_N]
    dn_a_ptrs = Dense_A + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn
    # [BLOCK_K, BLOCK_N]
    dn_b_ptrs = Dense_B + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
    accumulator2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # [BLOCK_M, BLOCK_N]
    for k in range(0, K, BLOCK_K):
        jga = tl.load(
            jg_a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        jgb = tl.load(
            jg_b_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        dna = tl.load(
            dn_a_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )  # [BLOCK_K, BLOCK_N]
        dnb = tl.load(
            dn_b_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )  # [BLOCK_K, BLOCK_N]

        acc1 = tl.dot(jga, dna, allow_tf32=ALLOW_TF32)  # [BLOCK_M, BLOCK_N]
        acc2 = tl.dot(jgb, dnb, allow_tf32=ALLOW_TF32)  # [BLOCK_M, BLOCK_N]
        accumulator1 += acc1
        accumulator2 += acc2
        jg_a_ptrs += BLOCK_K
        jg_b_ptrs += BLOCK_K

        dn_a_ptrs += BLOCK_K * stride_dk
        dn_b_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        # load bias
        bias_a_ptrs = Bias_A + off_b * stride_bias_b + offs_n  # [BLOCK_N]
        bias_b_ptrs = Bias_B + off_b * stride_bias_b + offs_n  # [BLOCK_N]

        bias_a = tl.load(
            bias_a_ptrs, mask=offs_n < N, eviction_policy="evict_last"
        )  # [BLOCK_N]
        bias_b = tl.load(
            bias_b_ptrs, mask=offs_n < N, eviction_policy="evict_last"
        )  # [BLOCK_N]

        # add bias to accumulator [BLOCK_M, BLOCK_N]
        A = accumulator1 + bias_a[None, :].to(tl.float32)
        B = accumulator2 + bias_b[None, :].to(tl.float32)
    else:
        A = accumulator1
        B = accumulator2

    # Apply Silu to A
    a_sigmoid = fast_sigmoid(A)
    A_SILU = A * a_sigmoid

    out = (A_SILU * B).to(Out.dtype.element_ty)

    # write back [BLOCK_M, BLOCK_N]
    out_ptrs = Out + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    alpha_ptrs = ALPHA + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    alpha_silu_ptrs = (
        ALPHA_SILU + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    )
    beta_ptrs = BETA + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )
    tl.store(
        alpha_ptrs,
        A.to(ALPHA.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )
    tl.store(
        alpha_silu_ptrs,
        A_SILU.to(ALPHA_SILU.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )

    tl.store(
        beta_ptrs,
        B.to(BETA.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N),
        eviction_policy="evict_first",
    )


@triton_autotune(
    configs=get_bmm_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "N", "K"],
)
@triton.jit
def _silu_jagged_dense_bmm_broadcast_add_fwd_kernel(
    seq_offsets,
    Jagged,
    Silu_Jagged,
    Dense,
    Bias,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    N,
    K,
    stride_jm,
    stride_sjm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    STORE_SILU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computing bmm Out = Silu(Jagged) @ Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N), and Out has shape (sum_B(M_i), N)
    """

    off_n = tl.program_id(0)
    off_m = tl.program_id(1)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += seq_start * stride_jm
    if STORE_SILU:
        Silu_Jagged += seq_start * stride_sjm
    Dense += off_b.to(tl.int64) * stride_db
    Out += seq_start * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    jg_silu_ptrs = None
    if STORE_SILU:
        jg_silu_ptrs = Silu_Jagged + offs_m[:, None] * stride_sjm + offs_k[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        # Apply Silu
        jg_fp32 = jg.to(tl.float32)
        jg_sigmoid = fast_sigmoid(jg_fp32)
        jg = (jg * jg_sigmoid).to(jg.dtype)
        if STORE_SILU:
            tl.store(
                jg_silu_ptrs,
                jg,
                mask=((offs_m[:, None] < seq_len) & ((k + offs_k)[None, :] < K)),
            )

        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk
        if STORE_SILU:
            jg_silu_ptrs += BLOCK_K

    if HAS_BIAS:
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N)
        accumulator += bias[None, :].to(tl.float32)

    out = accumulator.to(Out.dtype.element_ty)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N))


@triton_autotune(
    configs=get_bmm_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "N", "K"],
)
@triton.jit
def jagged_dense_bmm_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Bias,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    N,
    K,
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
    Computing bmm Out = Jagged x Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N), and Out has shape (sum_B(M_i), N)
    """

    off_n = tl.program_id(0)
    off_m = tl.program_id(1)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += seq_start * stride_jm
    Dense += off_b.to(tl.int64) * stride_db
    Out += seq_start * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
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
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N)
        accumulator += bias[None, :].to(tl.float32)

    out = accumulator.to(Out.dtype.element_ty)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N))


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _jagged_jagged_bmm_reduce_sum(
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


@triton_autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_M in [64, 128]
        for BLOCK_N in [32, 64]
        for BLOCK_K in [64, 128]
        for num_stages in [2, 3]
        for num_warps in [4, 8]
    ],
    key=["AUTOTUNE_MAX_SEQ_LEN", "N", "K"],
)
@triton.jit
def _silu_jagged_dense_bmm_broadcast_add_bwd_kernel(
    seq_offsets,
    Jagged,
    Dense,
    D_Out,
    D_Jagged,
    AUTOTUNE_MAX_SEQ_LEN,
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
    Computing backward of bmm Out = Silu(Jagged) @ Dense + Bias
    d_silu = (sigmoid(jagged) * (jagged * (1 - sigmoid(jagged)) + 1))
    d_jagged = d_bmm_out * dense.T * d_silu
    (M_i, K)   (M_i, N)    (K, N)    (M_i, K)

    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), D_Out has shape (sum_B(M_i), N),
    and D_Jagged has shape (sum_B(M_i), K)
    """
    off_k = tl.program_id(0)
    off_m = tl.program_id(1)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_k = off_k * BLOCK_K
    if start_m >= seq_len:
        return

    Jagged += seq_start * stride_jm
    D_Jagged += seq_start * stride_jm
    Dense += off_b.to(tl.int64) * stride_db
    D_Out += seq_start * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_k = start_k + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    jg = tl.load(
        jg_ptrs,
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < K),
        other=0.0,
    )
    # Apply silu bwd
    jg_fp32 = jg.to(tl.float32)
    jg_sigmoid = fast_sigmoid(jg_fp32)
    d_silu = (jg_sigmoid * (jg_fp32 * (1 - jg_sigmoid) + 1)).to(jg.dtype)

    d_out_ptrs = D_Out + offs_m[:, None] * stride_om + offs_n[None, :]
    dn_ptrs = Dense + offs_n[:, None] * stride_dn + offs_k[None, :] * stride_dk
    accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        d_out = tl.load(
            d_out_ptrs,
            mask=(offs_m[:, None] < seq_len) & ((n + offs_n[None, :]) < N),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((n + offs_n[:, None]) < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        accumulator += tl.dot(d_out, dn, allow_tf32=ALLOW_TF32)
        d_out_ptrs += BLOCK_N
        dn_ptrs += BLOCK_N * stride_dn

    d_jagged = accumulator.to(D_Jagged.dtype.element_ty) * d_silu
    d_jagged_ptrs = D_Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    tl.store(
        d_jagged_ptrs,
        d_jagged,
        mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < K),
    )


@torch.fx.wrap
def triton_silu_jagged_bmm_combine(
    max_seq_len: int,
    offsets: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    index: torch.Tensor,
    reverse_index: torch.Tensor,
    k: int,
    gates: Optional[torch.Tensor] = None,
    gates_index: Optional[torch.Tensor] = None,
    activation_checkpointing: bool = False,
    has_silu: bool = True,
) -> torch.Tensor:
    return SiluJaggedBmmCombine.apply(
        max_seq_len,
        offsets,
        jagged,
        weight,
        bias,
        reverse_index,
        k,
        gates,
        gates_index,
        activation_checkpointing,
        has_silu,
    )


def triton_silu_jagged_bmm_combine_wrapper(
    max_seq_len: int,
    offsets: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    index: torch.Tensor,
    reverse_index: torch.Tensor,
    k: int,
    gates: Optional[torch.Tensor] = None,
    gates_index: Optional[torch.Tensor] = None,
    activation_checkpointing: bool = False,
    has_silu: bool = True,
) -> torch.Tensor:
    return triton_silu_jagged_bmm_combine(
        max_seq_len=max_seq_len,
        offsets=offsets,
        jagged=jagged,
        weight=weight,
        bias=bias,
        index=index,
        reverse_index=reverse_index,
        k=k,
        gates=gates,
        gates_index=gates_index,
        has_silu=has_silu,
        activation_checkpointing=activation_checkpointing,
    )


class SiluJaggedBmmCombine(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        offsets: torch.Tensor,
        jagged: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        index: torch.Tensor,
        k: int,
        gates: Optional[torch.Tensor] = None,
        gates_index: Optional[torch.Tensor] = None,
        activation_checkpointing: Optional[bool] = False,
        has_silu: bool = True,
    ) -> torch.Tensor:
        jagged = switch_to_contiguous_if_needed(jagged)
        has_bias = bias is not None
        if has_bias:
            bias = switch_to_contiguous_if_needed(bias)  # pyre-ignore
            stride_bias_b = bias.stride(0)
        else:
            stride_bias_b = 0

        index = switch_to_contiguous_if_needed(index)
        # L: number of tokens
        # k: number of activated experts
        # N: number of tokens after dispatch, i.e., N = L * k
        # E: number of experts
        # D_in: input dimension
        # D_out: output dimension
        N, D_in = jagged.shape
        E, _, D_out = weight.shape
        assert N % k == 0
        L = N // k

        bmm_out = torch.empty((N, D_out), dtype=jagged.dtype, device=jagged.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(D_out, meta["BLOCK_N"]),
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            E,
        )

        if has_silu:
            silu_jagged: torch.Tensor = torch.empty_like(jagged)
            _silu_jagged_dense_bmm_broadcast_add_fwd_kernel[grid](
                seq_offsets=offsets,
                Jagged=jagged,
                Silu_Jagged=silu_jagged,
                Dense=weight,
                Bias=bias,
                Out=bmm_out,
                AUTOTUNE_MAX_SEQ_LEN=triton.next_power_of_2(max_seq_len),
                N=D_out,
                K=D_in,
                stride_jm=jagged.stride(0),
                stride_sjm=silu_jagged.stride(0),
                stride_db=weight.stride(0),
                stride_dk=weight.stride(1),
                stride_dn=weight.stride(2),
                stride_bias_b=stride_bias_b,
                stride_om=bmm_out.stride(0),
                HAS_BIAS=has_bias,
                ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
                STORE_SILU=not activation_checkpointing,
            )
        else:
            silu_jagged: torch.Tensor = jagged
            jagged_dense_bmm_broadcast_add_kernel[grid](
                seq_offsets=offsets,
                Jagged=jagged,
                Dense=weight,
                Bias=bias,
                Out=bmm_out,
                AUTOTUNE_MAX_SEQ_LEN=triton.next_power_of_2(max_seq_len),
                N=D_out,
                K=D_in,
                stride_jm=jagged.stride(0),
                stride_db=weight.stride(0),
                stride_dk=weight.stride(1),
                stride_dn=weight.stride(2),
                stride_bias_b=stride_bias_b,
                stride_om=bmm_out.stride(0),
                HAS_BIAS=has_bias,
                ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            )

        if gates is not None:
            assert gates_index is not None
            assert N == gates.numel() == gates_index.numel()
            g = switch_to_contiguous_if_needed(gates)
            g_index = switch_to_contiguous_if_needed(gates_index)
            g_stride = g.stride(0)
        else:
            g = torch.empty((), dtype=jagged.dtype, device=jagged.device)
            g_index = torch.empty((), dtype=index.dtype, device=index.device)
            g_stride = 0

        has_weight = gates is not None

        output = torch.empty((L, D_out), dtype=jagged.dtype, device=jagged.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(D_out, meta["BLOCK_D"]),
            triton.cdiv(L, meta["BLOCK_N"]),
        )

        _mul_merge_k_add_fwd[grid](
            index,
            bmm_out,
            g,
            g_index,
            output,
            L,
            D_out,
            g_stride,
            k,
            has_weight,
        )

        if activation_checkpointing:
            ctx.save_for_backward(offsets, jagged, weight, bias, index, g, g_index)
        else:
            ctx.save_for_backward(
                offsets, jagged, weight, bias, index, g, g_index, silu_jagged, bmm_out
            )

        ctx.L = L
        ctx.N = N
        ctx.k = k
        ctx.E = E
        ctx.D_in = D_in
        ctx.D_out = D_out
        ctx.has_weight = has_weight
        ctx.max_seq_len = max_seq_len
        ctx.activation_checkpointing = activation_checkpointing
        ctx.has_silu = has_silu

        return output

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_output: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        None,
        None,
        Optional[torch.Tensor],
        None,
        None,
        None,
    ]:
        offsets, jagged, weight, bias, index, g, g_index = (
            ctx.saved_tensors
            if ctx.activation_checkpointing
            else ctx.saved_tensors[:-2]
        )
        has_bias = bias is not None
        if has_bias:
            stride_bias_b = bias.stride(0)
        else:
            stride_bias_b = 0

        if ctx.activation_checkpointing:
            # Recomputation
            bmm_out = torch.empty(
                (ctx.N, ctx.D_out), dtype=jagged.dtype, device=jagged.device
            )

            grid = lambda meta: (  # noqa E731
                triton.cdiv(ctx.D_out, meta["BLOCK_N"]),
                triton.cdiv(ctx.max_seq_len, meta["BLOCK_M"]),
                ctx.E,
            )

            if ctx.has_silu:
                silu_jagged: torch.Tensor = torch.empty_like(jagged)
                _silu_jagged_dense_bmm_broadcast_add_fwd_kernel[grid](
                    seq_offsets=offsets,
                    Jagged=jagged,
                    Silu_Jagged=silu_jagged,
                    Dense=weight,
                    Bias=bias,
                    Out=bmm_out,
                    AUTOTUNE_MAX_SEQ_LEN=triton.next_power_of_2(ctx.max_seq_len),
                    N=ctx.D_out,
                    K=ctx.D_in,
                    stride_jm=jagged.stride(0),
                    stride_sjm=silu_jagged.stride(0),
                    stride_db=weight.stride(0),
                    stride_dk=weight.stride(1),
                    stride_dn=weight.stride(2),
                    stride_bias_b=stride_bias_b,
                    stride_om=bmm_out.stride(0),
                    HAS_BIAS=has_bias,
                    ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
                    STORE_SILU=True,
                )
            else:
                silu_jagged: torch.Tensor = jagged
                jagged_dense_bmm_broadcast_add_kernel[grid](
                    seq_offsets=offsets,
                    Jagged=jagged,
                    Dense=weight,
                    Bias=bias,
                    Out=bmm_out,
                    AUTOTUNE_MAX_SEQ_LEN=triton.next_power_of_2(ctx.max_seq_len),
                    N=ctx.D_out,
                    K=ctx.D_in,
                    stride_jm=jagged.stride(0),
                    stride_db=weight.stride(0),
                    stride_dk=weight.stride(1),
                    stride_dn=weight.stride(2),
                    stride_bias_b=stride_bias_b,
                    stride_om=bmm_out.stride(0),
                    HAS_BIAS=has_bias,
                    ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
                )
        else:
            silu_jagged, bmm_out = ctx.saved_tensors[-2:]

        # Backward computations
        d_output = switch_to_contiguous_if_needed(d_output)

        # TODO: avoid harding-coding BLOCK_D size
        BLOCK_D = 128

        d_bmm_out: torch.Tensor = torch.empty_like(bmm_out)
        dw_expanded: torch.Tensor = torch.empty(
            (g.numel(), triton.cdiv(ctx.D_out, BLOCK_D)), dtype=g.dtype, device=g.device
        )
        if g is not None:
            g_stride = g.stride(0)
        else:
            g_stride = 0

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.D_out, BLOCK_D),
            triton.cdiv(ctx.L, meta["BLOCK_N"]),
        )
        _mul_merge_k_add_bwd[grid](
            index,
            bmm_out,
            g,
            g_index,
            d_output,
            d_bmm_out,
            dw_expanded,
            ctx.L,
            ctx.D_out,
            g_stride,
            ctx.k,
            ctx.has_weight,
            BLOCK_D=BLOCK_D,
        )

        d_jagged = torch.empty_like(jagged)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.D_in, meta["BLOCK_K"]),
            triton.cdiv(ctx.max_seq_len, meta["BLOCK_M"]),
            ctx.E,
        )
        if ctx.has_silu:
            _silu_jagged_dense_bmm_broadcast_add_bwd_kernel[grid](
                seq_offsets=offsets,
                Jagged=jagged,
                Dense=weight,
                D_Out=d_bmm_out,
                D_Jagged=d_jagged,
                AUTOTUNE_MAX_SEQ_LEN=triton.next_power_of_2(ctx.max_seq_len),
                N=ctx.D_out,
                K=ctx.D_in,
                stride_jm=jagged.stride(0),
                stride_db=weight.stride(0),
                stride_dk=weight.stride(1),
                stride_dn=weight.stride(2),
                stride_om=d_bmm_out.stride(0),
                ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            )
        else:
            jagged_dense_bmm_broadcast_add_kernel[grid](
                seq_offsets=offsets,
                Jagged=d_bmm_out,
                Dense=weight,
                Bias=None,
                Out=d_jagged,
                AUTOTUNE_MAX_SEQ_LEN=triton.next_power_of_2(ctx.max_seq_len),
                N=ctx.D_in,
                K=ctx.D_out,
                stride_jm=d_bmm_out.stride(0),
                stride_db=weight.stride(0),
                stride_dk=weight.stride(2),
                stride_dn=weight.stride(1),
                stride_bias_b=0,
                stride_om=d_jagged.stride(0),
                HAS_BIAS=False,
                ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            )

        optimize = True
        if optimize:
            d_weight, d_bias = triton_jagged_bmm_reduce_sum(
                JaggedA=silu_jagged,
                JaggedB=d_bmm_out,
                offsets=offsets,
                reduce_sum=has_bias,
            )
        else:
            # tensors below needs to be initialized with zeros as there could be unused
            # rows in the weight and bias
            d_weight = torch.zeros_like(weight)

            if has_bias:
                d_bias = torch.zeros(
                    (ctx.E, ctx.D_out), device=d_output.device, dtype=d_output.dtype
                )
                stride_orb, stride_orn = d_bias.stride(0), d_bias.stride(1)
            else:
                d_bias = None
                stride_orb, stride_orn = 0, 0
            grid = lambda meta: (  # noqa E731
                ctx.E,
                triton.cdiv(ctx.D_in, meta["BLOCK_M"]),
                triton.cdiv(ctx.D_out, meta["BLOCK_N"]),
            )
            _jagged_jagged_bmm_reduce_sum[grid](
                seq_offsets=offsets,
                JaggedA=silu_jagged,
                JaggedB=d_bmm_out,
                Out=d_weight,
                ReduceOut=d_bias,
                M=ctx.D_in,
                N=ctx.D_out,
                AUTOTUNE_MAX_SEQ_LEN=triton.next_power_of_2(ctx.max_seq_len),
                stride_ak=silu_jagged.stride(0),
                stride_bk=d_bmm_out.stride(0),
                stride_ob=d_weight.stride(0),
                stride_om=d_weight.stride(1),
                stride_on=d_weight.stride(2),
                stride_orb=stride_orb,
                stride_orn=stride_orn,
                REDUCE_JAGGEDB=has_bias,
                ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            )

        return (
            None,  # max_seq_len
            None,  # offsets
            d_jagged,
            d_weight,
            d_bias,
            None,  # index
            None,  # k
            # TODO: might need FP32 accumulation
            (
                dw_expanded.sum(dim=1, keepdim=True).view(g.shape)
                if ctx.has_weight
                else None
            ),  # gates
            None,  # gates_index
            None,  # activation_checkpointing
            None,  # has_silu
        )


def triton_jagged_bmm_reduce_sum_split_k(
    JaggedA: torch.Tensor,  # (sum_B(K_i), M)
    JaggedB: torch.Tensor,  # (sum_B(K_i), N)
    offsets: torch.Tensor,  # (B+1)
    reduce_sum: bool = True,
    use_tma: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # d_weight: (B, M, N), d_bias: (B, N)
    dtype = JaggedA.dtype
    device = JaggedA.device
    B = offsets.shape[0] - 1
    M = JaggedA.shape[1]
    N = JaggedB.shape[1]
    d_weight = torch.zeros((B, M, N), dtype=torch.float32, device=device)
    grid = lambda meta: (  # noqa E731
        meta["SPLIT_K"],
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        B,
    )

    if use_tma:
        max_num_tiles = 64 * triton.cdiv(M, 64) * triton.cdiv(N, 64) * B
        TMA_SIZE = 128
        workspace_ptr = torch.empty(
            max_num_tiles * 3 * TMA_SIZE,
            dtype=torch.uint8,
            device="cuda",
        )
    else:
        workspace_ptr = None

    assert JaggedA.stride(0) == JaggedA.shape[1]
    assert JaggedB.stride(0) == JaggedB.shape[1]

    _jagged_jagged_bmm_split_k[grid](
        seq_offsets=offsets,
        JaggedA=JaggedA,
        JaggedB=JaggedB,
        Out=d_weight,
        M=M,
        N=N,
        # To avoid use max_seq_len here, we use total seq length as the max_seq_len
        AUTOTUNE_K=triton.next_power_of_2(JaggedA.shape[0]),
        stride_ak=JaggedA.stride(0),
        stride_bk=JaggedB.stride(0),
        stride_ob=d_weight.stride(0),
        stride_om=d_weight.stride(1),
        stride_on=d_weight.stride(2),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        workspace_ptr=workspace_ptr,
        USE_TMA=use_tma,
    )
    d_weight = d_weight.to(dtype)

    d_bias = None
    if reduce_sum:
        d_bias = torch.zeros((B, N), dtype=torch.float32, device=device)
        grid = lambda meta: (  # noqa E731
            meta["SPLIT_K"],
            triton.cdiv(N, meta["BLOCK_M"]),
            B,
        )
        _jagged_reduce_sum_split_k[grid](
            seq_offsets=offsets,
            Jagged=JaggedB,
            ReduceOut=d_bias,
            M=N,
            N=triton.next_power_of_2(JaggedB.shape[0]),
            stride_jm=JaggedB.stride(1),
            stride_jn=JaggedB.stride(0),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )
        d_bias = d_bias.to(dtype)
    return d_weight, d_bias
