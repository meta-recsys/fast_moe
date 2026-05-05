# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-strict

from __future__ import annotations

import torch

# @manual=//triton:triton
import triton.language as tl
from fast_moe.kernels.triton.utils import (
    _get_rowwise_quant_fp8_configs,
    fast_sigmoid,
    triton_autotune,
)

# @manual=//triton:triton
from triton import Config, jit  # @manual

MAX_FP8 = 448.0


def triton_rowwise_quant_fp8(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call the triton kernel to quantize to fp8 with row-wise scalings.

    Args:
        a (Tensor): higher precision input tensor, can be 2D or 3D.

    Returns:
        Tensor: fp8 scaled tensor.
        Tensor: scale tensor per row.
    """
    return _triton_transpose_rowwise_quant_fp8(a, transpose=False)


def triton_transpose_rowwise_quant_fp8(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call the triton kernel to transpose on -2 and -1 dimensions, and then quantize to fp8 with row-wise scalings.

    Args:
        a (Tensor): higher precision input tensor, can be 2D or 3D.

    Returns:
        Tensor: fp8 scaled tensor.
        Tensor: scale tensor per row.
    """
    return _triton_transpose_rowwise_quant_fp8(a, transpose=True)


def _triton_transpose_rowwise_quant_fp8(
    a: torch.Tensor, a_fp8: torch.Tensor | None = None, transpose: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    if transpose:
        if len(a.shape) == 2:
            m1 = 1
            n, m2 = a.shape
            stride_am1, stride_am2, stride_an = 0, a.stride(1), a.stride(0)
            fp8_shape = (m2, n)
            scale_shape = (m2,)
        elif len(a.shape) == 3:
            m1, n, m2 = a.shape
            stride_am1, stride_am2, stride_an = a.stride(0), a.stride(2), a.stride(1)
            fp8_shape = (m1, m2, n)
            scale_shape = (m1, m2)
            pass
        else:
            raise ValueError("Input tensor must be 2D or 3D")
        stride_fp8_am1, stride_fp8_am2, stride_fp8_an = m2 * n, n, 1
    else:
        if len(a.shape) == 2:
            m1 = 1
            m2, n = a.shape
            stride_am1, stride_am2, stride_an = 0, a.stride(0), a.stride(1)
            fp8_shape = (m2, n)
            scale_shape = (m2,)
        elif len(a.shape) == 3:
            m1, m2, n = a.shape
            stride_am1, stride_am2, stride_an = a.stride(0), a.stride(1), a.stride(2)
            fp8_shape = (m1, m2, n)
            scale_shape = (m1, m2)
        else:
            raise ValueError("Input tensor must be 2D or 3D")
        stride_fp8_am1, stride_fp8_am2, stride_fp8_an = (m2 * n, n, 1)
    num_rows = m1 * m2

    a_scale = torch.empty((num_rows), dtype=torch.float32, device=a.device)
    if a_fp8 is None:
        a_fp8 = torch.empty(fp8_shape, device=a.device, dtype=torch.float8_e4m3fn)

    grid = (num_rows,)

    _kernel_quantize_fp8_row[grid](
        A=a,
        A_scale=a_scale,
        A_fp8=a_fp8,
        M1=m1,
        M2=m2,
        N=n,
        stride_am1=stride_am1,
        stride_am2=stride_am2,
        stride_an=stride_an,
        stride_fp8_am1=stride_fp8_am1,
        stride_fp8_am2=stride_fp8_am2,
        stride_fp8_an=stride_fp8_an,
        MAX_FP8=MAX_FP8,
    )

    a_scale = a_scale.reshape(scale_shape)

    return a_fp8, a_scale


@triton_autotune(
    configs=[
        Config({"BLOCK_SIZE": 1024}),
    ],
    key=["N"],
)
@jit
def _kernel_quantize_fp8_row(
    A,
    A_scale,
    A_fp8,
    M1,
    M2,
    N,
    stride_am1,
    stride_am2,
    stride_an,
    stride_fp8_am1,
    stride_fp8_am2,
    stride_fp8_an,
    MAX_FP8: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): [m, n] higher precision input tensor.
        A_scale (Tensor): [m] scale tensor per row.
        A_fp8 (Tensor): [m, n] fp8 scaled tensor. A_fp8 = A * a_scale
        M (int): Number of rows.
        N (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        BLOCK_SIZE (int): Block size for reduction.
    """
    pid = tl.program_id(0)
    m1id = pid // M2
    m2id = pid % M2
    n_offset = tl.arange(0, BLOCK_SIZE)

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + m1id * stride_am1 + m2id * stride_am2 + n_offset * stride_an,
            mask=n_offset < N,
            other=0.0,
        )
        tile_max = tl.max(tl.abs(a))
        cur_max = tl.maximum(tile_max, cur_max)

        n_offset += BLOCK_SIZE

    # Scale and quantize.
    a_scale = calculate_scale(cur_max, MAX_FP8)
    tl.store(A_scale + pid, a_scale)
    n_offset = tl.arange(0, BLOCK_SIZE)
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + m1id * stride_am1 + m2id * stride_am2 + n_offset * stride_an,
            mask=n_offset < N,
            other=0.0,
        )
        a_fp8 = a * a_scale
        a_fp8 = a_fp8.to(tl.float8e4nv)
        tl.store(
            A_fp8
            + m1id * stride_fp8_am1
            + m2id * stride_fp8_am2
            + n_offset * stride_fp8_an,
            a_fp8,
            mask=n_offset < N,
        )
        n_offset += BLOCK_SIZE


@jit
def calculate_scale(
    cur_max,
    MAX_FP8: tl.constexpr,
) -> torch.Tensor:
    a_scale = MAX_FP8 / cur_max
    a_scale = tl.exp2(tl.floor(tl.log2(a_scale)))
    # Check for inf or nan
    is_nan = a_scale != a_scale
    is_inf = (a_scale == float("inf")) | (a_scale == float("-inf"))
    is_not_finite = is_nan | is_inf
    a_scale = tl.where(is_not_finite, 2.0**127, a_scale)
    return a_scale


@triton_autotune(
    configs=_get_rowwise_quant_fp8_configs(),
    key=["D_IN", "K"],
)
@jit
def _rowwise_quant_fp8_kernel(
    weight,
    weight_scale,
    weight_fp8,
    D_IN,
    K,
    stride_km,
    silu_out,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    APPLY_SILU: tl.constexpr,
) -> None:
    MAX_FP8 = 448.0

    off_m = tl.program_id(0) * BLOCK_M
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.multiple_of(offs_m, BLOCK_M)
    offs_k = tl.multiple_of(offs_k, BLOCK_K)

    cur_col_max = tl.zeros([BLOCK_M], dtype=tl.float32)

    weight_ptr = weight + off_m * stride_km + offs_m[:, None] * stride_km
    for k in range(0, K, BLOCK_K):
        w = tl.load(
            weight_ptr + offs_k[None, :] + k,
            mask=((off_m + offs_m[:, None]) < D_IN) & ((offs_k[None, :] + k) < K),
            other=0.0,
        )

        if APPLY_SILU:
            w_fp32 = w.to(tl.float32)
            w_sigmoid = fast_sigmoid(w_fp32)
            w = (w_fp32 * w_sigmoid).to(w.dtype)

            tl.store(
                silu_out + (offs_m[:, None] + off_m) * stride_km + offs_k[None, :] + k,
                w,
                mask=((off_m + offs_m[:, None]) < D_IN) & ((offs_k[None, :] + k) < K),
            )

        cur_col_max = tl.maximum(tl.max(tl.abs(w.to(tl.float32)), axis=1), cur_col_max)

    w_scale = calculate_scale(cur_col_max, MAX_FP8)

    tl.store(
        weight_scale + off_m + offs_m,
        1.0 / w_scale,
        mask=((off_m + offs_m) < D_IN),
    )

    # quantize weight to fp8
    weight_fp8_ptr = weight_fp8 + off_m * stride_km + offs_m[:, None] * stride_km
    for k in range(0, K, BLOCK_K):
        if not APPLY_SILU:
            w = tl.load(
                weight_ptr + offs_k[None, :] + k,
                mask=((off_m + offs_m[:, None]) < D_IN) & ((offs_k[None, :] + k) < K),
                other=0.0,
            )
        else:
            w = tl.load(
                silu_out + (offs_m[:, None] + off_m) * stride_km + offs_k[None, :] + k,
                mask=((off_m + offs_m[:, None]) < D_IN) & ((offs_k[None, :] + k) < K),
                other=0.0,
            )

        w_fp8_ = w * w_scale[:, None]
        w_fp8 = w_fp8_.to(tl.float8e4nv)

        tl.store(
            weight_fp8_ptr + offs_k[None, :] + k,
            w_fp8,
            mask=((off_m + offs_m[:, None]) < D_IN) & ((offs_k[None, :] + k) < K),
        )
