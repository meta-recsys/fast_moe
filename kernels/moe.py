#!/usr/bin/env python3

# pyre-strict

from typing import Optional

import torch
from fast_moe.kernels.pytorch.moe import (
    pytorch_index_select_jagged_bmm,
    pytorch_mul_merge_k_add,
)

from fast_moe.kernels.triton.triton_moe import (
    triton_index_select_jagged_bmm_wrapper,
    triton_mul_merge_k_add_wrapper,
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
