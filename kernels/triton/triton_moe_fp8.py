# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-unsafe

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from fast_moe.kernels.triton.triton_gemm_fp8 import grouped_gemm_fp8_rowwise_bias
from fast_moe.kernels.triton.triton_moe import (
    _jagged_jagged_bmm,
    _jagged_jagged_bmm_reduce_sum,
    _jagged_reduce_sum,
    _mul_merge_k_add_bwd,
    _mul_merge_k_add_fwd,
    _silu_jagged_dense_bmm_broadcast_add_bwd_kernel,
    _silu_jagged_dense_bmm_broadcast_add_fwd_kernel,
    jagged_dense_bmm_broadcast_add_kernel,
    triton_jagged_bmm_reduce_sum,
)
from fast_moe.kernels.triton.triton_quant_fp8 import (
    _rowwise_quant_fp8_kernel,
    calculate_scale,
    triton_rowwise_quant_fp8,
    triton_transpose_rowwise_quant_fp8,
)
from fast_moe.kernels.triton.utils import (
    fast_sigmoid,
    get_bmm_configs,
    next_power_of_2,
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from mslk.quantize.triton.fp8_quantize import quantize_fp8_row


def triton_silu_jagged_fp8(
    seq_offsets: torch.Tensor,  # [B + 1], offsets on dim L
    Jagged: torch.Tensor,  # [L, K]
    max_seq_len: int,  # max sequence length
    K: int,  # embedding dimension
    Silu_Jagged: torch.Tensor,  # [L, K]
    Silu_Jagged_fp8: torch.Tensor,  # [L, K]
    Silu_Jagged_Scale: torch.Tensor,  # [L]
) -> torch.Tensor:
    return _SiluJaggedFp8.apply(
        max_seq_len,
        seq_offsets,
        Jagged,
        Silu_Jagged,
        Silu_Jagged_fp8,
        Silu_Jagged_Scale,
    )


class _SiluJaggedFp8(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,  # max sequence length
        seq_offsets: torch.Tensor,  # [B + 1], offsets on dim L
        Jagged: torch.Tensor,  # [L, K]
        Silu_Jagged: torch.Tensor,  # [L, K]
        Silu_Jagged_fp8: torch.Tensor,  # [L, K]
        Silu_Jagged_Scale: torch.Tensor,  # [L]
    ) -> torch.Tensor:
        L, D_in = Jagged.shape
        E = seq_offsets.shape[0] - 1

        silu_jagged: torch.Tensor = torch.empty_like(Jagged, dtype=torch.float32)

        silu_jagged_fp8: torch.Tensor = torch.empty_like(
            Jagged, dtype=torch.float8_e4m3fn, device=Jagged.device
        )

        silu_jagged_scale: torch.Tensor = torch.empty(
            L, dtype=torch.float32, device=Jagged.device
        )
        grid = lambda meta: (  # noqa E731
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            E,
        )

        # pyre-ignore[9]
        MAX_FP8: triton.language.constexpr = 448.0

        _silu_jagged_fp8_kernel[grid](
            seq_offsets,
            Jagged,
            Silu_Jagged=silu_jagged,
            Silu_Jagged_fp8=silu_jagged_fp8,
            Silu_Jagged_Scale=silu_jagged_scale,
            MAX_SEQ_LEN_KEY=next_power_of_2(max_seq_len),
            K=D_in,
            stride_jm=Jagged.stride(0),
            MAX_FP8=MAX_FP8,
        )

        ctx.save_for_backward(seq_offsets, Jagged, silu_jagged, silu_jagged_scale)
        ctx.E = E
        ctx.max_seq_len = max_seq_len
        ctx.D_in = D_in
        ctx.L = L
        return silu_jagged_fp8


def _get_jagged_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_K in [32, 64]:
            for num_stages in [2, 3]:
                for num_warps in [4, 8]:
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": BLOCK_M,
                                "BLOCK_K": BLOCK_K,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


@triton_autotune(
    configs=_get_jagged_configs(),
    key=["MAX_SEQ_LEN_KEY", "K"],
)
@triton.jit
def _silu_jagged_fp8_kernel(
    seq_offsets,
    Jagged,
    Silu_Jagged,
    Silu_Jagged_fp8,
    Silu_Jagged_Scale,
    MAX_SEQ_LEN_KEY,
    K,
    stride_jm,
    MAX_FP8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off_m = tl.program_id(0)
    off_b = tl.program_id(1)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    if start_m >= seq_len:
        return
    Jagged += seq_start * stride_jm
    Silu_Jagged += seq_start * stride_jm
    Silu_Jagged_fp8 += seq_start * stride_jm

    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]

    jg_silu_fp8_ptrs = Silu_Jagged_fp8 + offs_m[:, None] * stride_jm + offs_k[None, :]
    jg_silu_ptrs = Silu_Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]

    cur_row_max = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs + k,
            mask=(offs_m[:, None] < seq_len) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        # Apply Silu
        jg_fp32 = jg.to(tl.float32)
        jg_sigmoid = fast_sigmoid(jg_fp32)
        jg_mul = jg_fp32 * jg_sigmoid
        cur_row_max = tl.maximum(tl.max(tl.abs(jg_mul), axis=1), cur_row_max)

        tl.store(
            jg_silu_ptrs + k,
            jg_mul,
            mask=((offs_m[:, None] < seq_len) & ((k + offs_k)[None, :] < K)),
        )

    jg_silu_scale = calculate_scale(cur_row_max, MAX_FP8)
    scale_addr = Silu_Jagged_Scale + seq_start

    tl.store(
        scale_addr + offs_m,
        jg_silu_scale,
        mask=(offs_m < seq_len),
    )

    # quantize silu_jagged to fp8
    for k in range(0, K, BLOCK_K):
        jg_silu = tl.load(
            jg_silu_ptrs + k,
            mask=((offs_m[:, None] < seq_len) & ((k + offs_k)[None, :] < K)),
        )
        jg_silu_fp8_ = jg_silu * jg_silu_scale[:, None]
        jg_silu_fp8 = jg_silu_fp8_.to(tl.float8e4nv)

        tl.store(
            jg_silu_fp8_ptrs + k,
            jg_silu_fp8,
            mask=((offs_m[:, None] < seq_len) & ((k + offs_k)[None, :] < K)),
        )


def triton_bmm_weight_rowwise_quant_fp8(
    weight: torch.Tensor,  # [E, D_in, D_out]
) -> torch.Tensor:
    return BMMWeightRowwiseQuantFp8.apply(weight)


class BMMWeightRowwiseQuantFp8(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        weight: torch.Tensor,  # [E, K, D_out]
    ) -> torch.Tensor:
        assert weight.dim() == 3
        E, D_in, D_out = weight.shape

        weight_fp8: torch.Tensor = torch.empty_like(
            weight, dtype=torch.float8_e4m3fn, device=weight.device
        )

        weight_scale: torch.Tensor = torch.zeros(
            E * D_out, dtype=torch.float32, device=weight.device
        )

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(D_out, meta["BLOCK_K"]),
        )

        # pyre-ignore[9]
        MAX_FP8: triton.language.constexpr = 448.0

        _bmm_weight_rowwise_quant_fp8_fwd_kernel[grid](
            weight,
            weight_scale,
            weight_fp8,
            D_IN=D_in,
            K=D_out,
            stride_km=D_out,
            stride_mk=D_in,
            MAX_FP8=MAX_FP8,
        )

        return weight_fp8


def _get_bmm_weight_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_K in [32, 64]:
            for num_stages in [2, 3]:
                for num_warps in [4, 8]:
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": BLOCK_M,
                                "BLOCK_K": BLOCK_K,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


@triton_autotune(
    configs=_get_bmm_weight_configs(),
    key=["D_IN", "K"],
)
@triton.jit
def _bmm_weight_rowwise_quant_fp8_fwd_kernel(
    weight,
    weight_scale,
    weight_fp8,
    D_IN,
    K,
    stride_km,
    stride_mk,
    MAX_FP8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_k = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = off_k * BLOCK_K + tl.arange(0, BLOCK_K)

    stride_b = D_IN * K
    weight += off_b * stride_b
    weight_fp8 += off_b * stride_b
    scale_addr = weight_scale + off_b * K

    cur_col_max = tl.zeros([BLOCK_K], dtype=tl.float32)

    for m in range(0, D_IN, BLOCK_M):
        w = tl.load(
            weight + (m + offs_m[:, None]) * stride_km + offs_k[None, :],
            mask=((m + offs_m[:, None]) < D_IN) & (offs_k[None, :] < K),
            other=0.0,
        )

        cur_col_max = tl.maximum(tl.max(tl.abs(w.to(tl.float32)), axis=0), cur_col_max)

    w_scale = calculate_scale(cur_col_max, MAX_FP8)

    tl.store(
        scale_addr + offs_k,
        w_scale,
        mask=(offs_k < K),
    )

    # quantize weight to fp8
    for m in range(0, D_IN, BLOCK_M):
        w = tl.load(
            weight + (m + offs_m[:, None]) * stride_km + offs_k[None, :],
            mask=((m + offs_m[:, None]) < D_IN) & (offs_k[None, :] < K),
            other=0.0,
        )
        w_fp8_ = w * w_scale[None, :]
        w_fp8 = w_fp8_.to(tl.float8e4nv)

        tl.store(
            weight_fp8 + (m + offs_m[:, None]) * stride_km + offs_k[None, :],
            w_fp8,
            mask=((m + offs_m[:, None]) < D_IN) & (offs_k[None, :] < K),
        )


def triton_silu_jagged_bmm_fp8(
    seq_offsets: torch.Tensor,  # [B + 1], offsets on dim L
    max_seq_len: int,  # max sequence length
    jagged: torch.Tensor,  # [L, K]
    weight: torch.Tensor,  # [E, K, D_out]
    bias: torch.Tensor,  # [E, D_out]
    use_grouped_gemm: bool = True,
) -> torch.Tensor:
    if not use_grouped_gemm:
        return SiluJaggedBmmFp8.apply(seq_offsets, max_seq_len, jagged, weight, bias)
    else:
        return SiluJaggedBmmFp8MixedGemmCombine.apply(
            seq_offsets, max_seq_len, jagged, weight, bias
        )


class SiluJaggedBmmFp8(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        jagged: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        E, D_in, D_out = weight.shape
        L = jagged.shape[0]
        jagged_out: torch.Tensor = torch.empty(
            (L, D_out), dtype=jagged.dtype, device=jagged.device
        )

        weight_scale: torch.Tensor = torch.zeros(
            E * D_out, dtype=torch.float32, device=weight.device
        )

        weight_fp8: torch.Tensor = torch.empty_like(
            weight, dtype=torch.float8_e4m3fn, device=weight.device
        )

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(D_out, meta["BLOCK_K"]),
        )

        MAX_FP8 = 448.0

        _bmm_weight_rowwise_quant_fp8_fwd_kernel[grid](
            weight,
            weight_scale,
            weight_fp8,
            D_IN=D_in,
            K=D_out,
            stride_km=D_out,
            stride_mk=D_in,
            MAX_FP8=MAX_FP8,
        )

        silu_jagged: torch.Tensor = torch.empty_like(jagged, dtype=torch.float32)

        silu_jagged_fp8: torch.Tensor = torch.empty_like(
            jagged, dtype=torch.float8_e4m3fn, device=jagged.device
        )

        silu_jagged_scale: torch.Tensor = torch.empty(
            L, dtype=torch.float32, device=jagged.device
        )
        grid = lambda meta: (  # noqa E731
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            E,
        )

        _silu_jagged_fp8_kernel[grid](
            seq_offsets,
            jagged,
            Silu_Jagged=silu_jagged,
            Silu_Jagged_fp8=silu_jagged_fp8,
            Silu_Jagged_Scale=silu_jagged_scale,
            MAX_SEQ_LEN_KEY=next_power_of_2(max_seq_len),
            K=D_in,
            stride_jm=jagged.stride(0),
            MAX_FP8=MAX_FP8,
        )

        grid = lambda meta: (  # noqa E731
            triton.cdiv(D_out, meta["BLOCK_N"]),
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            E,
        )

        _jagged_bmm_fp8_kernel[grid](
            seq_offsets=seq_offsets,
            jagged_fp8=silu_jagged_fp8,
            jagged_scale=silu_jagged_scale,
            weight_fp8=weight_fp8,
            weight_scale=weight_scale,
            bias=bias,
            jagged_out=jagged_out,
            AUTOTUNE_MAX_SEQ_LEN=next_power_of_2(max_seq_len),
            N=D_out,
            K=D_in,
            stride_jm=jagged.stride(0),
            stride_db=weight.stride(0),
            stride_dk=weight.stride(1),
            stride_dn=weight.stride(2),
            stride_bias_b=bias.stride(0),
            stride_om=jagged_out.stride(0),
            HAS_BIAS=True,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        return jagged_out


class SiluJaggedBmmFp8GroupedGemm(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        jagged: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        E, D_out, D_in = weight.shape
        L = jagged.shape[0]

        xq = torch.empty_like(jagged, dtype=torch.float8_e4m3fn, device=jagged.device)
        x_scale = torch.empty(L, dtype=torch.float32, device=jagged.device)
        _jagged_silu = torch.empty_like(jagged, dtype=jagged.dtype)
        grid = lambda meta: (  # noqa E731
            triton.cdiv(L, meta["BLOCK_M"]),
        )

        _rowwise_quant_fp8_kernel[grid](
            jagged,
            x_scale,
            xq,
            D_IN=L,
            K=D_in,
            stride_km=D_in,
            silu_out=_jagged_silu,
            APPLY_SILU=True,
        )

        wq = torch.empty(
            (E * D_out, D_in), dtype=torch.float8_e4m3fn, device=weight.device
        )
        w_scale = torch.empty((E * D_out), dtype=torch.float32, device=weight.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(E * D_out, meta["BLOCK_M"]),
        )

        _rowwise_quant_fp8_kernel[grid](
            weight,
            w_scale,
            wq,
            D_IN=E * D_out,
            K=D_in,
            stride_km=D_in,
            silu_out=None,
            APPLY_SILU=False,
        )

        m_sizes = seq_offsets[1:] - seq_offsets[:-1]
        m_sizes = m_sizes.to(torch.int32)

        jagged_out = grouped_gemm_fp8_rowwise_bias(
            xq,
            wq,
            m_sizes,
            x_scale,
            w_scale,
            bias=bias,
            _use_warp_specialization=False,
        )

        ctx.save_for_backward(seq_offsets, _jagged_silu, m_sizes, jagged, weight, bias)

        ctx.L = L
        ctx.E = E
        ctx.D_in = D_in
        ctx.D_out = D_out
        ctx.max_seq_len = max_seq_len

        return jagged_out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_bmm_out: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            offsets,
            silu,
            m_sizes,
            jagged,
            weight,
            bias,
        ) = ctx.saved_tensors

        d_jagged = torch.empty_like(jagged)

        # weight has been flattened to 2D in forward
        weight = weight.view(ctx.E, ctx.D_out, ctx.D_in)
        # after permute, weight is [E, D_in, D_out]
        weight = weight.permute(0, 2, 1).contiguous()

        weight = weight.view(-1, weight.shape[-1])
        wq, w_scale = quantize_fp8_row(weight)

        q_d_bmm = torch.empty_like(
            d_bmm_out, dtype=torch.float8_e4m3fn, device=d_bmm_out.device
        )
        q_d_bmm_scale = torch.empty(
            (ctx.L,), dtype=torch.float32, device=d_bmm_out.device
        )
        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.L, meta["BLOCK_M"]),
        )

        _rowwise_quant_fp8_kernel[grid](
            d_bmm_out,
            q_d_bmm_scale,
            q_d_bmm,
            D_IN=ctx.L,
            K=ctx.D_out,
            stride_km=ctx.D_out,
            silu_out=None,
            APPLY_SILU=False,
        )

        d_silu = grouped_gemm_fp8_rowwise_bias(
            q_d_bmm,
            wq,
            m_sizes,
            q_d_bmm_scale,
            w_scale,
            bias=None,
            _use_warp_specialization=False,
        )

        # d_weight shape is [E, D_in, D_out]
        d_weight = torch.empty(
            (ctx.E, ctx.D_out, ctx.D_in),
            dtype=weight.dtype,
            device=weight.device,
        )
        d_bias = torch.empty_like(bias, dtype=bias.dtype, device=bias.device)
        grid = lambda meta: (  # noqa E731
            ctx.E,
            triton.cdiv(ctx.D_in, meta["BLOCK_M"]),
            triton.cdiv(ctx.D_out, meta["BLOCK_N"]),
        )

        _jagged_jagged_bmm[grid](
            seq_offsets=offsets,
            JaggedA=silu,
            JaggedB=d_bmm_out,
            Out=d_weight,
            M=ctx.D_in,
            N=ctx.D_out,
            AUTOTUNE_K=triton.next_power_of_2(ctx.max_seq_len),
            stride_ak=silu.stride(0),
            stride_bk=d_bmm_out.stride(0),
            stride_ob=d_weight.stride(0),
            stride_om=d_weight.stride(1),
            stride_on=d_weight.stride(2),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        grid = lambda meta: (  # noqa E731
            ctx.E,
            triton.cdiv(ctx.D_out, meta["BLOCK_M"]),
        )

        _jagged_reduce_sum[grid](
            seq_offsets=offsets,
            Jagged=d_bmm_out,
            ReduceOut=d_bias,
            M=ctx.D_out,
            N=ctx.L,
            stride_jm=d_bmm_out.stride(1),
            stride_jn=d_bmm_out.stride(0),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        torch.ops.aten.silu_backward(d_silu, silu, grad_input=d_jagged)

        return (
            None,
            None,
            d_jagged,
            d_weight,
            d_bias,
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
def _jagged_bmm_fp8_kernel(
    seq_offsets,
    jagged_fp8,
    jagged_scale,
    weight_fp8,
    weight_scale,
    bias,
    jagged_out,
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

    jagged_fp8 += seq_start * stride_jm
    weight_fp8 += off_b * stride_db

    jagged_out += seq_start * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = jagged_fp8 + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = weight_fp8 + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs + k,
            mask=(offs_m[:, None] < seq_len) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )

        dn = tl.load(
            dn_ptrs + k * stride_dk,
            mask=(offs_n[None, :] < N) & ((k + offs_k)[:, None] < K),
            other=0.0,
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)

    w_scale = tl.load(weight_scale + off_b * stride_dk + offs_n, mask=(offs_n < N))
    j_scale = tl.load(jagged_scale + seq_start + offs_m, mask=(offs_m < seq_len))

    inv_w_scale = 1.0 / w_scale
    inv_j_scale = 1.0 / j_scale
    scale = inv_j_scale[:, None] * inv_w_scale[None, :]
    accumulator = accumulator * scale

    if HAS_BIAS:
        bias_ptrs = bias + off_b * stride_bias_b + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N)
        accumulator += bias[None, :].to(tl.float32)

    out = accumulator.to(jagged_out.dtype.element_ty)

    out_ptrs = jagged_out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N))


def triton_index_select_jagged_bmm_fp8(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return IndexSelectJaggedBmm.apply(max_seq_len, offsets, index, jagged, weight, bias)


@triton_autotune(
    configs=get_bmm_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _index_select_jagged_bmm(
    seq_offsets,  # [B+1]
    Index,  # [Sum_B(M)], jagged indices in range [0, L * A)
    Jagged,  # [L, K]
    JaggedScale,  # [L]
    Dense,  # [B, K, N]
    DenseScale,  # [B, N]
    Bias,  # [B, N]
    Out,  # [Sum_B(M), N]
    M,
    N,
    K,
    A,
    stride_jm,
    stride_jsm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_dsb,
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
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )  # [BLOCK_K, BLOCK_N]

        # TODO: add switch for fp8 fast accumulation
        # accumulator = tl.dot(jg, dn, accumulator, out_dtype=tl.float32, allow_tf32=allow_tf32)
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    # Invert scaling.
    jg_scale_ptrs = JaggedScale + idx * stride_jsm  # [BLOCK_M]
    dn_scale_ptrs = DenseScale + off_b * stride_dsb + offs_n  # [BLOCK_N]
    jg_scale = tl.load(jg_scale_ptrs, mask=offs_m < seq_len, other=0.0)
    dn_scale = tl.load(dn_scale_ptrs, mask=offs_n < N, other=0.0)
    # Invert vector, then multiply on matrix for speed.
    inv_jg_scale = 1.0 / jg_scale
    inv_dn_scale = 1.0 / dn_scale
    accumulator = (accumulator * inv_jg_scale[:, None]) * inv_dn_scale[None, :]

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
    JaggedScale,  # [Sum_B(M)]
    Dense,  # [B, K, N]
    DenseScale,  # [B, N]
    Out,  # [Sum_B(M), N]
    M,
    N,
    K,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_dsb,
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

    # Invert scaling.
    jg_scale_ptrs = JaggedScale + offs_m.to(tl.int64)  # [BLOCK_M]
    dn_scale_ptrs = DenseScale + off_b * stride_dsb + offs_n  # [BLOCK_N]
    jg_scale = tl.load(jg_scale_ptrs, mask=offs_m < seq_len, other=0.0)
    dn_scale = tl.load(dn_scale_ptrs, mask=offs_n < N, other=0.0)
    # Invert vector, then multiply on matrix for speed.
    inv_jg_scale = 1.0 / jg_scale
    inv_dn_scale = 1.0 / dn_scale
    accumulator = (accumulator * inv_jg_scale[:, None]) * inv_dn_scale[None, :]

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
def _indexed_jagged_jagged_bmm(
    seq_offsets,  # [B+1]
    Index,  # [Sum_B(M)], jagged indices in range [0, L)
    JaggedA,  # [M, L]
    JaggedScaleA,  # [M]
    JaggedB,  # [N, Sum_B(M)]
    JaggedScaleB,  # [N]
    Out,  # [B, M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
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
    Compute Batched Matrix Multiplication (BMM) of the form:
    Out = JaggedA @ JaggedB.T  # [B, M, N]
    , specifically for b in range(B):
        JaggedA_b = JaggedA[:, Index[seq_offsets[b]:seq_offsets[b+1]]]  # [M, K]
        JaggedB_b = JaggedB[:, seq_offsets[b]:seq_offsets[b+1]]  # [N, K]
        Out_b = JaggedA_b @ JaggedB_b.T  # [M, N]
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

    if seq_len == 0:
        out = accumulator.to(Out.dtype.element_ty)
        tl.store(
            out_ptrs,
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )
        return

    Index += seq_start
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    idx_ptrs = Index + offs_k  # [BLOCK_K]

    JaggedB += seq_start.to(tl.int64) * stride_bk

    jg_b_ptrs = (
        JaggedB + offs_k[:, None].to(tl.int64) * stride_bk + offs_n[None, :] * stride_bn
    )  # [BLOCK_K, BLOCK_N]

    for k in range(0, seq_len, BLOCK_K):
        idx = tl.load(idx_ptrs, mask=(k + offs_k) < seq_len, other=0)  # [BLOCK_K]
        jg_a_ptrs = (
            JaggedA + idx[None, :] * stride_ak + offs_m[:, None] * stride_am
        )  # [BLOCK_M, BLOCK_K]
        jg_a = tl.load(
            jg_a_ptrs,
            mask=(offs_m[:, None] < M) and ((k + offs_k)[None, :] < seq_len),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        jg_b = tl.load(
            jg_b_ptrs,
            mask=(offs_n[None, :] < N) and ((k + offs_k)[:, None] < seq_len),
            other=0.0,
        )  # [BLOCK_K, BLOCK_N]

        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)

        idx_ptrs += BLOCK_K
        jg_b_ptrs += BLOCK_K * stride_bk

    # Invert scaling.
    ja_scale_ptrs = JaggedScaleA + offs_m  # [BLOCK_M]
    jb_scale_ptrs = JaggedScaleB + offs_n  # [BLOCK_N]
    ja_scale = tl.load(ja_scale_ptrs, mask=offs_m < M, other=0.0)
    jb_scale = tl.load(jb_scale_ptrs, mask=offs_n < N, other=0.0)
    # Invert vector, then multiply on matrix for speed.
    inv_ja_scale = 1.0 / ja_scale
    inv_jb_scale = 1.0 / jb_scale
    accumulator = (accumulator * inv_ja_scale[:, None]) * inv_jb_scale[None, :]

    # write back [BLOCK_M, BLOCK_N]
    out = accumulator.to(Out.dtype.element_ty)
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
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

        # Quantize weight and jagged tensors
        # TODO: use triton kernel instead of pytorch kernel
        # weight_t_fp8: [E, N, K], weight_t_fp8_scale: [E, N]
        weight_t_fp8, weight_t_fp8_scale = triton_transpose_rowwise_quant_fp8(weight)
        # jagged_fp8: [L, K], jagged_fp8_scale: [L]
        jagged_fp8, jagged_fp8_scale = triton_rowwise_quant_fp8(jagged)

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        _index_select_jagged_bmm[grid](
            seq_offsets=offsets,
            Index=index,
            Jagged=jagged_fp8,
            JaggedScale=jagged_fp8_scale,
            Dense=weight_t_fp8,
            DenseScale=weight_t_fp8_scale,
            Bias=bias,
            Out=output,
            # M is only used for trigger autotune
            M=triton.next_power_of_2(max_seq_len),
            N=N,
            K=K,
            A=A,
            stride_jm=jagged_fp8.stride(0),
            stride_jsm=jagged_fp8_scale.stride(0),
            stride_db=weight_t_fp8.stride(0),
            stride_dk=weight_t_fp8.stride(2),
            stride_dn=weight_t_fp8.stride(1),
            stride_dsb=weight_t_fp8_scale.stride(0),
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
        d_out = switch_to_contiguous_if_needed(d_out)

        offsets, index, jagged, weight, bias = ctx.saved_tensors
        E, K, N = ctx.E, ctx.K, ctx.N
        has_bias = bias is not None

        d_jagged_expanded = torch.empty(
            (jagged.shape[0], index.shape[1], jagged.shape[1]),
            device=jagged.device,
            dtype=torch.float32,
        )  # [L, A, K]
        d_weight = torch.empty_like(weight)  # [E, K, N]

        # Quantize d_out
        # d_out_fp8: [L * A, N], d_out_fp8_scale: [L * A]
        d_out_fp8, d_out_fp8_scale = triton_rowwise_quant_fp8(d_out)
        # Quantize weight
        # TODO: quantization can be done in forward pass
        # weight_fp8: [E, K, N], weight_fp8_scale: [E, K]
        weight_fp8, weight_fp8_scale = triton_rowwise_quant_fp8(weight)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.max_seq_len, meta["BLOCK_M"]),
            E,
            triton.cdiv(K, meta["BLOCK_N"]),
        )
        _jagged_bmm_index_add[grid](
            seq_offsets=offsets,
            Index=index,
            Jagged=d_out_fp8,
            JaggedScale=d_out_fp8_scale,
            Dense=weight_fp8,
            DenseScale=weight_fp8_scale,
            Out=d_jagged_expanded,
            # M is only used for triggering autotune
            M=triton.next_power_of_2(ctx.max_seq_len),
            N=K,
            K=N,
            stride_jm=d_out_fp8.stride(0),
            stride_db=weight_fp8.stride(0),
            stride_dk=weight_fp8.stride(2),
            stride_dn=weight_fp8.stride(1),
            stride_dsb=weight_fp8_scale.stride(0),
            stride_om=d_jagged_expanded.stride(1),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        # Quantize jagged
        # TODO: quantization can be done in forward pass
        # jagged_t_fp8: [K, L], weight_fp8_scale: [K]
        jagged_t_fp8, jagged_t_fp8_scale = triton_transpose_rowwise_quant_fp8(jagged)
        # Quantize d_out
        # d_out_fp8: [N, L * A], d_out_fp8_scale: [N]
        d_out_t_fp8, d_out_t_fp8_scale = triton_transpose_rowwise_quant_fp8(d_out)

        grid = lambda meta: (  # noqa E731
            E,
            triton.cdiv(K, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _indexed_jagged_jagged_bmm[grid](
            seq_offsets=offsets,
            Index=index.view(-1) // index.shape[1],
            JaggedA=jagged_t_fp8,  # [K, L]
            JaggedScaleA=jagged_t_fp8_scale,  # [K]
            JaggedB=d_out_t_fp8,  # [N, L * A]
            JaggedScaleB=d_out_t_fp8_scale,  # [N]
            Out=d_weight,  # [E, K, N]
            M=K,
            N=N,
            # K is only used for triggering autotune
            K=triton.next_power_of_2(ctx.max_seq_len),
            stride_am=jagged_t_fp8.stride(0),
            stride_ak=jagged_t_fp8.stride(1),
            stride_bn=d_out_t_fp8.stride(0),
            stride_bk=d_out_t_fp8.stride(1),
            stride_ob=d_weight.stride(0),
            stride_om=d_weight.stride(1),
            stride_on=d_weight.stride(2),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        if has_bias:
            d_bias = torch.empty_like(bias)  # [E, N]

            grid = lambda meta: (  # noqa E731
                E,
                triton.cdiv(N, meta["BLOCK_N"]),
            )
            _jagged_reduce_sum[grid](
                seq_offsets=offsets,
                Jagged=d_out,
                ReduceOut=d_bias,
                M=N,
                # N is only used for triggering autotune
                N=triton.next_power_of_2(ctx.max_seq_len),
                stride_jm=d_out.stride(1),
                stride_jn=d_out.stride(0),
                ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            )
        else:
            d_bias = None

        return (
            None,
            None,
            None,
            d_jagged_expanded.sum(dim=1).to(jagged.dtype),  # sum over A dimension
            d_weight,
            d_bias,
        )


def triton_silu_jagged_bmm_combine_fp8(
    max_seq_len: int,
    offsets: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    index: torch.Tensor,
    reverse_index: torch.Tensor,
    k: int,
    gating_scores: Optional[torch.Tensor] = None,
    gates_index: Optional[torch.Tensor] = None,
    activation_checkpointing: bool = False,
    has_silu: bool = True,
) -> torch.Tensor:
    return SiluJaggedBmmFp8MixedGemmCombine.apply(
        max_seq_len,
        offsets,
        jagged,
        weight,
        bias,
        reverse_index,
        k,
        gating_scores,
        gates_index,
        activation_checkpointing,
        has_silu,
    )


# SiluJaggedBmmCombine operator in hybrid precision:
# fwd: rowwise fp8
# bwd: bf16
class SiluJaggedBmmFp8MixedGemmCombine(torch.autograd.Function):
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
        E, D_out, D_in = weight.shape
        L = jagged.shape[0]

        xq = torch.empty_like(jagged, dtype=torch.float8_e4m3fn, device=jagged.device)
        x_scale = torch.empty(L, dtype=torch.float32, device=jagged.device)
        _jagged_silu = torch.empty_like(jagged, dtype=jagged.dtype)
        grid = lambda meta: (  # noqa E731
            triton.cdiv(L, meta["BLOCK_M"]),
        )

        _rowwise_quant_fp8_kernel[grid](
            jagged,
            x_scale,
            xq,
            D_IN=L,
            K=D_in,
            stride_km=D_in,
            silu_out=_jagged_silu,
            APPLY_SILU=True,
        )

        wq = torch.empty(
            (E * D_out, D_in), dtype=torch.float8_e4m3fn, device=weight.device
        )
        w_scale = torch.empty((E * D_out), dtype=torch.float32, device=weight.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(E * D_out, meta["BLOCK_M"]),
        )

        _rowwise_quant_fp8_kernel[grid](
            weight,
            w_scale,
            wq,
            D_IN=E * D_out,
            K=D_in,
            stride_km=D_in,
            silu_out=None,
            APPLY_SILU=False,
        )

        m_sizes = offsets[1:] - offsets[:-1]
        m_sizes = m_sizes.to(torch.int32)

        jagged_out = grouped_gemm_fp8_rowwise_bias(
            xq,
            wq,
            m_sizes,
            x_scale,
            w_scale,
            bias=bias,
            _use_warp_specialization=False,
        )

        if gates is not None:
            assert gates_index is not None
            assert L == gates.numel() == gates_index.numel()
            g = switch_to_contiguous_if_needed(gates)
            g_index = switch_to_contiguous_if_needed(gates_index)
            g_stride = g.stride(0)
        else:
            g = torch.empty((), dtype=jagged.dtype, device=jagged.device)
            g_index = torch.empty((), dtype=index.dtype, device=index.device)
            g_stride = 0

        has_weight = gates is not None

        N = L // k
        output = torch.empty((N, D_out), dtype=jagged.dtype, device=jagged.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(D_out, meta["BLOCK_D"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        _mul_merge_k_add_fwd[grid](
            index,
            jagged_out,
            g,
            g_index,
            output,
            N,
            D_out,
            g_stride,
            k,
            has_weight,
        )

        if activation_checkpointing:
            ctx.save_for_backward(offsets, jagged, weight, bias, index, g, g_index)
        else:
            ctx.save_for_backward(
                offsets,
                jagged,
                weight,
                bias,
                index,
                g,
                g_index,
                _jagged_silu,
                jagged_out,
            )

        ctx.L = N
        ctx.N = L
        ctx.k = k
        ctx.E = E
        ctx.D_in = D_in
        ctx.D_out = D_out
        ctx.has_weight = has_weight
        ctx.max_seq_len = max_seq_len
        ctx.activation_checkpointing = activation_checkpointing
        ctx.has_silu = has_silu

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

        weight = weight.permute(0, 2, 1)
        weight = switch_to_contiguous_if_needed(weight)

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
            d_weight.permute(0, 2, 1),
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
