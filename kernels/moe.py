# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-strict

from typing import Optional

import torch
from fast_moe.kernels.pytorch.moe import (
    pytorch_index_select_jagged_bmm,
    pytorch_index_select_jagged_bmm_3D,
    pytorch_index_select_jagged_bmm_swiglu,
    pytorch_index_select_jagged_gating_bmm,
    pytorch_mul_merge_k_add,
    pytorch_silu_jagged_bmm_combine,
)
from fast_moe.kernels.triton.triton_moe import (
    IndexSelectJaggedBmmOption,
    SiluJaggedBmmCombineOption,
    triton_index_select_jagged_bmm_3D_wrapper,
    triton_index_select_jagged_bmm_swiglu_wrapper,
    triton_index_select_jagged_bmm_wrapper,
    triton_index_select_jagged_gating_bmm_wrapper,
    triton_mul_merge_k_add_wrapper,
    triton_silu_jagged_bmm_combine_wrapper,
)
from fast_moe.kernels.utils import KernelType
from torch.fx._symbolic_trace import is_fx_tracing


torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


def index_select_jagged_bmm(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    kernel: KernelType = KernelType.PYTORCH,
    triton_option: Optional[IndexSelectJaggedBmmOption] = None,
) -> torch.Tensor:
    if not is_fx_tracing():
        assert index.ndim == 2 and index.shape[0] == jagged.shape[0]
    if kernel == KernelType.TRITON:
        return triton_index_select_jagged_bmm_wrapper(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
            option=triton_option,
        )
    else:
        return pytorch_index_select_jagged_bmm(
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
        )


def mul_merge_k_add(
    index: torch.Tensor,
    reverse_index: torch.Tensor,
    value: torch.Tensor,
    stable_sorting: bool = False,
    weight: Optional[torch.Tensor] = None,
    weight_index: Optional[torch.Tensor] = None,
    kernel: KernelType = KernelType.PYTORCH,
    triton_cc_version: str = "",
) -> torch.Tensor:
    r"""
    Multiply ``value`` with ``weight`` and ``index_add`` the result to a zero tensor with ``index``.

    Args:
        index (torch.Tensor): a 1D tensor of indices in shape [N], indicating for each row in
            ``value``, which row in the output it should be accumulated to.
        reverse_index (torch.Tensor): a 2D tensor of indices in shape [L, K], where L * K == N,
            indicating for each row in the output, which rows in the input it should be accumulated from.
        value (torch.Tensor): a 2D tensor of values, with shape [N, D], where N == L * K.
        weight (Optional[torch.Tensor]): when present, multiply with ``value`` before accumulation.
        weight_index (Optional[torch.Tensor]): when present, use the indices in ``weight_index`` to reorder
            ``weight`` values before multiply.

    Return:
        A 2D tensor of shape [L, D]
    """
    N = index.shape[0]
    L, K = reverse_index.shape
    _, D = value.shape
    if not is_fx_tracing():
        assert index.ndim == 1 and reverse_index.ndim == 2
        assert N == L * K and N == value.shape[0]

    if kernel == KernelType.TRITON:
        return triton_mul_merge_k_add_wrapper(
            index, reverse_index, value, K, weight, weight_index
        )
    else:
        return pytorch_mul_merge_k_add(
            index=index,
            value=value,
            k=K,
            weight=weight,
            weight_index=weight_index,
        )


def index_select_jagged_bmm_3D(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    kernel: KernelType = KernelType.PYTORCH,
) -> torch.Tensor:
    """
    3D version of index_select_jagged_bmm
    weight: [E, D_in, D_out]
    jagged: [L, E, D_in]
    """
    if not is_fx_tracing():
        assert index.ndim == 2 and index.shape[0] == jagged.shape[0]
    if kernel == KernelType.TRITON:
        return triton_index_select_jagged_bmm_3D_wrapper(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
        )
    elif kernel == KernelType.PYTORCH:
        return pytorch_index_select_jagged_bmm_3D(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
        )
    else:
        raise NotImplementedError(f"Unsupported kernel {kernel}")


def index_select_jagged_bmm_swiglu(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_p: torch.Tensor,
    bias_p: Optional[torch.Tensor],
    kernel: KernelType = KernelType.PYTORCH,
) -> torch.Tensor:
    if not is_fx_tracing():
        assert index.ndim == 2 and index.shape[0] == jagged.shape[0]
    if kernel == KernelType.TRITON:
        return triton_index_select_jagged_bmm_swiglu_wrapper(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
            weight_p=weight_p,
            bias_p=bias_p,
        )
    else:
        return pytorch_index_select_jagged_bmm_swiglu(
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
            weight_p=weight_p,
            bias_p=bias_p,
        )


def index_select_jagged_gating_bmm(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged_a: torch.Tensor,
    jagged_b: torch.Tensor,
    weight_a: torch.Tensor,
    bias_a: Optional[torch.Tensor],
    weight_b: torch.Tensor,
    bias_b: Optional[torch.Tensor],
    kernel: KernelType = KernelType.PYTORCH,
) -> torch.Tensor:
    if not is_fx_tracing():
        assert (
            index.ndim == 2
            and index.shape[0] == jagged_a.shape[0] == jagged_b.shape[0]
            and jagged_a.shape == jagged_b.shape
            and weight_a.shape == weight_b.shape
        )
    if kernel == KernelType.TRITON:
        return triton_index_select_jagged_gating_bmm_wrapper(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged_a=jagged_a,
            jagged_b=jagged_b,
            weight_a=weight_a,
            bias_a=bias_a,
            weight_b=weight_b,
            bias_b=bias_b,
        )
    else:
        return pytorch_index_select_jagged_gating_bmm(
            offsets=offsets,
            index=index,
            jagged_a=jagged_a,
            jagged_b=jagged_b,
            weight_a=weight_a,
            bias_a=bias_a,
            weight_b=weight_b,
            bias_b=bias_b,
        )


def silu_jagged_bmm_combine(
    max_seq_len: int,
    offsets: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    index: torch.Tensor,
    reverse_index: torch.Tensor,
    gating_scores: Optional[torch.Tensor] = None,
    gates_index: Optional[torch.Tensor] = None,
    has_silu: bool = True,
    kernel: KernelType = KernelType.PYTORCH,
    triton_option: Optional[SiluJaggedBmmCombineOption] = None,
) -> torch.Tensor:
    N = index.shape[0]
    L, K = reverse_index.shape
    _, D_in = jagged.shape
    E, _, D_out = weight.shape
    if not is_fx_tracing():
        assert index.ndim == 1 and reverse_index.ndim == 2
        assert N == L * K and N == jagged.shape[0], f"{N=}, {L=}, {K=}, {jagged.shape=}"
    if kernel == KernelType.TRITON:
        if triton_option is None:
            triton_option = SiluJaggedBmmCombineOption()
        return triton_silu_jagged_bmm_combine_wrapper(
            max_seq_len=max_seq_len,
            offsets=offsets,
            jagged=jagged,
            weight=weight,
            bias=bias,
            index=index,
            reverse_index=reverse_index,
            k=K,
            gating_scores=gating_scores,
            gates_index=gates_index,
            has_silu=has_silu,
            activation_checkpointing=triton_option.activation_checkpointing,
            d_weight_optimization=triton_option.d_weight_optimization,
            d_weight_split_k_kernel=triton_option.d_weight_split_k_kernel,
            d_weight_split_k_kernel_tma=triton_option.d_weight_split_k_kernel_tma,
        )
    elif kernel == KernelType.TRITON_CC:
        raise NotImplementedError(
            "TRITON_CC is not supported for silu_jagged_bmm_combine"
        )
    else:
        return pytorch_silu_jagged_bmm_combine(
            offsets=offsets,
            jagged=jagged,
            weight=weight,
            bias=bias,
            index=index,
            k=K,
            gating_scores=gating_scores,
            gates_index=gates_index,
            has_silu=has_silu,
        )
