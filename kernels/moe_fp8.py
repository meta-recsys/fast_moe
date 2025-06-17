#!/usr/bin/env python3

# pyre-strict

from typing import Optional

import torch
from fast_moe.kernels.pytorch.moe import pytorch_silu_jagged_bmm_combine

from fast_moe.kernels.pytorch.moe_fp8 import (
    pytorch_bmm_weight_rowwise_quant_fp8,
    pytorch_index_select_jagged_bmm_fp8,
    pytorch_index_select_jagged_bmm_raw,
    pytorch_silu_jagged_bmm,
    pytorch_silu_jagged_fp8,
)

from fast_moe.kernels.triton.triton_moe_fp8 import (
    triton_bmm_weight_rowwise_quant_fp8,
    triton_index_select_jagged_bmm_fp8,
    triton_silu_jagged_bmm_combine_fp8,
    triton_silu_jagged_bmm_fp8,
    triton_silu_jagged_fp8,
)

from fast_moe.kernels.utils import KernelType


def silu_jagged_fp8(
    seq_offsets: torch.Tensor,  # [B + 1], offsets on dim L
    Jagged: torch.Tensor,  # [L, K]
    max_seq_len: int,  # max sequence length
    K: int,  # embedding dimension
    Silu_Jagged: torch.Tensor,  # [L, K]
    Silu_Jagged_fp8: torch.Tensor,  # [L, K]
    Silu_Jagged_Scale: torch.Tensor,  # [L]
    kernel: KernelType = KernelType.PYTORCH,
) -> torch.Tensor:
    r"""
    Apply silu to jagged tensor and convert to fp8.

    Args:
        seq_offsets (torch.Tensor): offsets of shape [B + 1], where B is the batch size.
        jagged (torch.Tensor): jagged tensor of shape [L, D], where L is the number of tokens, i.e., seq_offsets[-1].
        jagged_fp8 (JaggedTensorFP8): jagged tensor fp8 representation.

    Return:
        A jagged tensor of shape [L, D] in fp8 format.
    """

    if kernel == KernelType.TRITON:
        return triton_silu_jagged_fp8(
            seq_offsets,
            Jagged,
            max_seq_len,
            K,
            Silu_Jagged,
            Silu_Jagged_fp8,
            Silu_Jagged_Scale,
        )
    elif kernel == KernelType.PYTORCH:
        return pytorch_silu_jagged_fp8(
            seq_offsets,
            Jagged,
            max_seq_len,
            K,
            Silu_Jagged,
            Silu_Jagged_fp8,
            Silu_Jagged_Scale,
        )
    else:
        raise AssertionError("Unsupported kernel type")


def bmm_weight_rowwise_quant_fp8(
    weight: torch.Tensor,
    kernel: KernelType = KernelType.PYTORCH,
) -> torch.Tensor:
    r"""
    Apply rowwise quantization to weight tensor and convert to fp8.

    Args:
        weight (torch.Tensor): weight tensor of shape [E, D_IN, D_OUT].

    Return:
        A weight tensor of shape [E, D_IN, D_OUT] in fp8 format.
    """
    if kernel == KernelType.TRITON:
        return triton_bmm_weight_rowwise_quant_fp8(weight)
    elif kernel == KernelType.PYTORCH:
        return pytorch_bmm_weight_rowwise_quant_fp8(weight)
    else:
        raise AssertionError("pytorch_bmm_weight_rowwise_quant_fp8 not implemented yet")


def silu_jagged_bmm_fp8(
    seq_offsets: torch.Tensor,  # [B + 1], offsets on dim L
    max_seq_len: int,  # max sequence length
    jagged: torch.Tensor,  # [L, K]
    weight: torch.Tensor,  # [E, K, D_out]
    bias: torch.Tensor,  # [E, D_out]
    kernel: KernelType = KernelType.PYTORCH,
) -> torch.Tensor:
    r"""
    Apply bmm to jagged tensor and weight to fp8.

    Args:
        seq_offsets (torch.Tensor): offsets of shape [B + 1], where B is the batch size.
        jagged (torch.Tensor): jagged tensor of shape [L, D_IN].
        weight (torch.Tensor): weight tensor of shape [D_IN, D_OUT].

    Return:
        A jagged tensor of shape [L, D_OUT]
    """
    if kernel == KernelType.TRITON:
        return triton_silu_jagged_bmm_fp8(
            seq_offsets, max_seq_len, jagged, weight, bias
        )
    elif kernel == KernelType.PYTORCH:
        return pytorch_silu_jagged_bmm(seq_offsets, max_seq_len, jagged, weight, bias)
    else:
        raise AssertionError("triton_cc_jagged_bmm_fp8 not implemented yet")


def silu_jagged_bmm_combine_fp8(
    max_seq_len: int,
    offsets: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    index: torch.Tensor,
    reverse_index: torch.Tensor,
    gates: Optional[torch.Tensor] = None,
    gates_index: Optional[torch.Tensor] = None,
    activation_checkpointing: bool = False,
    has_silu: bool = True,
    kernel: KernelType = KernelType.PYTORCH,
) -> torch.Tensor:
    r"""
    Apply bmm to jagged tensor and weight to fp8.

    Args:
        seq_offsets (torch.Tensor): offsets of shape [B + 1], where B is the batch size.
        jagged (torch.Tensor): jagged tensor of shape [L, D_IN].
        weight (torch.Tensor): weight tensor of shape [D_IN, D_OUT].

    Return:
        A jagged tensor of shape [L, D_OUT]
    """

    L, K = reverse_index.shape
    if kernel == KernelType.TRITON:
        return triton_silu_jagged_bmm_combine_fp8(
            max_seq_len=max_seq_len,
            offsets=offsets,
            jagged=jagged,
            weight=weight,
            bias=bias,
            index=index,
            reverse_index=reverse_index,
            k=K,
            gates=gates,
            gates_index=gates_index,
            activation_checkpointing=activation_checkpointing,
            has_silu=has_silu,
        )
    elif kernel == KernelType.PYTORCH:
        return pytorch_silu_jagged_bmm_combine(
            offsets=offsets,
            jagged=jagged,
            weight=weight,
            bias=bias,
            index=index,
            k=K,
            gates=gates,
            gates_index=gates_index,
            has_silu=has_silu,
        )
    else:
        raise AssertionError("triton_cc_jagged_bmm_fp8 not implemented yet")


def index_select_jagged_bmm(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    kernel: KernelType = KernelType.PYTORCH,
    fp8: bool = False,
) -> torch.Tensor:
    """
    Performs a batched matrix multiplication with expert-specific weights and biases.
        output = torch.cat(
            [
                jagged[index.view(-1)[offsets[e] : offsets[e + 1]] // A] @ weight[e]
                + (bias[e] if bias is not None else 0)
                for e in range(E)
            ],
            dim=0,
        )
    Dimensions:
        L: number of input tokens
        E: number of total experts
        A: number of activated experts
        K: input dimension
        N: output dimension
    Args:
        max_seq_len (int): Maximum number of input tokens for any expert.
        offsets (torch.Tensor): A tensor of shape [E+1] representing the cumulative number of tokens dispatched to each expert.
        index (torch.Tensor): A tensor of shape [L, A] that is flattened and sorted by expert. Each entry is calculated as token_id * A + [0, A).
        jagged (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
        weight (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
        bias (Optional[torch.Tensor]): A tensor of shape [E, N] containing the biases for each expert.
    Returns:
        torch.Tensor: A tensor of shape [L*A, N] containing the output after applying the linear transformation for each expert's tokens.
    """
    if not fp8:
        assert kernel == KernelType.PYTORCH
        return pytorch_index_select_jagged_bmm_raw(
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
        )
    if kernel == KernelType.TRITON:
        return triton_index_select_jagged_bmm_fp8(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
        )
    elif kernel == KernelType.PYTORCH:
        return pytorch_index_select_jagged_bmm_fp8(
            offsets=offsets,
            index=index,
            jagged=jagged,
            weight=weight,
            bias=bias,
        )
    else:
        raise NotImplementedError(f"Unsupported kernel {kernel}")
