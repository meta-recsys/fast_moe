#!/usr/bin/env python3

# pyre-unsafe

from typing import List, Optional

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

from fast_moe.kernels.triton.utils import triton_autotune


def _get_transpose_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [64, 128, 256]:
        for BLOCK_M in [64, 128, 256]:
            for num_warps in [8, 16]:
                for num_stages in [2, 3]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton_autotune(
    configs=_get_transpose_configs(),
    key=["AUTOTUNE_M", "AUTOTUNE_N", "HAS_INDEX"],
)
@triton.jit
def _kernel_index_transpose(
    IN,
    OUT,
    M,
    N,
    AUTOTUNE_M,
    AUTOTUNE_N,
    INDEX,
    HAS_INDEX: tl.constexpr,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    if HAS_INDEX:
        m_offsets = tl.load(INDEX + rm, mask=rm < M, other=0)  # [BLOCK_M]
    else:
        m_offsets = rm  # [BLOCK_M]
    in_offset = m_offsets[:, None].to(tl.int64) * N + rn[None, :]  # [BLOCK_M, BLOCK_N]
    out_offset = (
        rm[:, None].to(tl.int64) * stride_om + rn[None, :].to(tl.int64) * stride_on
    )  # [BLOCK_M, BLOCK_N]
    mask = (rm < M)[:, None] & (rn < N)[None, :]  # [BLOCK_M, BLOCK_N]
    data = tl.load(IN + in_offset, mask=mask)
    tl.store(OUT + out_offset, data, mask=mask)


def triton_transpose(x):
    return triton_index_transpose(x, None)


def triton_index_select(x, index: torch.Tensor):
    return triton_index_transpose(x, index, transpose=False)


def triton_index_transpose(
    input, index: Optional[torch.Tensor] = None, transpose: bool = True
):
    """
    Transpose a jagged tensor with index. output = input[index].T.contiguous()
    Args:
        input: [?, N] input tensor
        index: [M] tensor
        transpose: whether to transpose the output tensor
    Returns:
        output: [N, M] tensor
    """
    assert input.is_contiguous()

    has_index = False
    if index is not None:
        has_index = True
        assert index.is_contiguous()
        M = index.shape[0]
        _, N = input.shape
    else:
        M, N = input.shape

    if transpose:
        output = torch.empty(N, M, dtype=input.dtype, device=input.device)
        stride_on, stride_om = output.stride()
    else:
        output = torch.empty(M, N, dtype=input.dtype, device=input.device)
        stride_om, stride_on = output.stride()

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _kernel_index_transpose[grid](
        input,
        output,
        M,
        N,
        triton.next_power_of_2(M),
        triton.next_power_of_2(N),
        INDEX=index,
        HAS_INDEX=has_index,
        stride_om=stride_om,
        stride_on=stride_on,
    )
    return output


def _get_sum_dim1_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_K in [32, 64, 128, 256]:
        for num_warps in [4, 8, 16]:
            for num_stages in [1, 2, 3]:
                configs.append(
                    triton.Config(
                        {"BLOCK_K": BLOCK_K},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton_autotune(
    configs=_get_sum_dim1_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _kernel_sum_dim1(
    a_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_m,
    stride_n,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    k_block = tl.program_id(1)

    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offs < K

    # use fp32 to compute the sum
    accumulator = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for a_idx in range(N):
        offs = m.to(tl.int64) * stride_m + a_idx * stride_n + k_offs
        data = tl.load(a_ptr + offs, mask=mask_k, other=0.0).to(tl.float32)
        accumulator += data

    out_offs = m.to(tl.int64) * K + k_offs
    tl.store(out_ptr + out_offs, accumulator.to(out_ptr.dtype.element_ty), mask=mask_k)


def triton_sum_dim1(x: torch.Tensor):
    """
    Sum along the second dimension of a 3D tensor. Accumulate in fp32.
    Summing along dim=1 was supposed to be faster than dim=0.
    Args:
        x: [M, N, K] input tensor
    Returns:
        output: [M, K] tensor, output = x.sum(dim=1)
    """
    assert x.is_contiguous()
    M, N, K = x.shape
    assert triton.next_power_of_2(K) == K
    output = torch.empty((M, K), dtype=x.dtype, device=x.device)

    def grid(META):
        return (M, triton.cdiv(K, META["BLOCK_K"]))

    _kernel_sum_dim1[grid](
        x, output, triton.next_power_of_2(M), N, K, x.stride(0), x.stride(1)
    )
    return output


@triton_autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_N in [32, 64]
        for BLOCK_K in [64, 128]
        for num_stages in [2, 3]
        for num_warps in [4, 8]
    ],
    key=["N", "K"],
)
@triton.jit
def _kernel_jagged_reduce_sum(
    seq_offsets,
    JaggedB,
    ReduceOut,
    N,
    K,
    stride_bk,
    stride_orb,
    stride_orn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)

    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_n = off_n * BLOCK_N

    offs_n = start_n + tl.arange(0, BLOCK_N)

    out_reduce_ptrs = ReduceOut + off_b * stride_orb + offs_n * stride_orn
    acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if seq_len == 0:
        tl.store(
            out_reduce_ptrs,
            acc_reduce.to(ReduceOut.dtype.element_ty),
            mask=(offs_n < N),
        )
        return

    offs_k = tl.arange(0, BLOCK_K)

    JaggedB += seq_start.to(tl.int64) * stride_bk

    jg_b_ptrs = JaggedB + offs_k[:, None].to(tl.int64) * stride_bk + offs_n[None, :]

    for k in range(0, seq_len, BLOCK_K):
        jg_b = tl.load(
            jg_b_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_n[None, :] < N) and ((k + offs_k)[:, None] < seq_len),
            other=0.0,
        )

        acc_reduce += tl.sum(jg_b, axis=0)

        jg_b_ptrs += BLOCK_K * stride_bk

    tl.store(
        out_reduce_ptrs,
        acc_reduce.to(ReduceOut.dtype.element_ty),
        mask=(offs_n < N),
    )


def triton_jagged_reduce_sum(
    input: torch.Tensor, offsets: torch.Tensor, max_seq_len: int
) -> torch.Tensor:
    E = offsets.shape[0] - 1
    N = input.shape[1]
    out = torch.empty((E, N), dtype=input.dtype, device=input.device)

    grid = lambda meta: (  # noqa E731
        E,
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _kernel_jagged_reduce_sum[grid](
        seq_offsets=offsets,
        JaggedB=input,
        ReduceOut=out,
        N=N,
        # K is only used for triggering autotune
        K=triton.next_power_of_2(max_seq_len),
        stride_bk=input.stride(0),
        stride_orb=out.stride(0),
        stride_orn=out.stride(1),
    )
    return out


@triton_autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_N": BLOCK_N,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_N in [512, 1024]
        for num_stages in [2, 3]
        for num_warps in [4, 8]
    ],
    key=["N"],
)
@triton.jit
def _kernel_silu_backward(
    x_ptr,  # float*  : forward‑input tensor
    dy_ptr,  # float*  : upstream grad dY
    dx_ptr,  # float*  : output grad dX
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    sig = tl.sigmoid(x_fp32)  # σ(x)
    dx = dy * sig * (1.0 + x * (1.0 - sig))  # dX formula

    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def triton_silu_backward(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """
    Compute dX for SiLU given input x and upstream gradient dy.
    Both tensors must have identical shape and live on the same CUDA device.
    """
    assert x.shape == dy.shape and x.device == dy.device
    n = x.numel()
    dx = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)  # noqa E731
    _kernel_silu_backward[grid](x, dy, dx, n)
    return dx
