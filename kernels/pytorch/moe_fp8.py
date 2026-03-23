# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import List, Optional

import torch
from fast_moe.kernels.pytorch.quant_fp8 import (
    calculate_scale,
    pytorch_rowwise_quant_fp8,
)
from torch.nn import functional as F


def pytorch_silu_jagged_fp8(
    seq_offsets: torch.Tensor,  # [B + 1], offsets on dim L
    Jagged: torch.Tensor,  # [L, K]
    max_seq_len: int,  # max sequence length
    K: int,  # embedding dimension
    Silu_Jagged: torch.Tensor,  # [L, K]
    Silu_Jagged_fp8: torch.Tensor,  # [L, K]
    Silu_Jagged_Scale: torch.Tensor,  # [L]
) -> torch.Tensor:
    silu_jagged = F.silu(Jagged).to(torch.float32)

    silu_jagged_max, _ = torch.max(torch.abs(silu_jagged), dim=1)
    silu_jagged_scale = calculate_scale(silu_jagged_max)

    silu_jagged_fp8 = (silu_jagged * silu_jagged_scale[:, None]).to(torch.float8_e4m3fn)

    return silu_jagged_fp8


def pytorch_bmm_weight_rowwise_quant_fp8(
    weight: torch.Tensor,  # [B, K, D]
) -> torch.Tensor:
    weight_max_t, _ = torch.max(torch.abs(weight), dim=1)
    weight_scale_t = calculate_scale(weight_max_t)
    weight_fp8_t = (weight * weight_scale_t[:, None, :]).to(torch.float8_e4m3fn)

    return weight_fp8_t


def pytorch_silu_jagged_bmm(
    offsets: torch.Tensor,  # [B + 1], offsets on dim L
    max_seq_len: int,  # max sequence length
    jagged: torch.Tensor,  # [L, K]
    weight: torch.Tensor,  # [E, K, D_out]
    bias: torch.Tensor,  # [E, D_out]
) -> torch.Tensor:
    K = jagged.size(dim=1)
    L = jagged.size(dim=0)
    silu_jagged = torch.empty_like(jagged, dtype=torch.float32)
    silu_jagged_fp8 = torch.empty_like(jagged, dtype=torch.float8_e4m3fn)
    silu_jagged_scale = torch.empty(L, dtype=torch.float32)
    pytorch_silu_jagged_fp8(
        offsets, jagged, max_seq_len, K, silu_jagged, silu_jagged_fp8, silu_jagged_scale
    )

    pytorch_bmm_weight_rowwise_quant_fp8(weight)
    partition_sizes: List[int] = (offsets[1:] - offsets[:-1]).tolist()
    jagged_list: List[torch.Tensor] = list(F.silu(jagged).split(partition_sizes))

    hidden_list: List[torch.Tensor] = [
        F.linear(
            jagged_list[i],
            weight[i],
            bias[i],
        )
        for i in range(len(jagged_list))
    ]

    return torch.cat(hidden_list, dim=0)


def pytorch_index_select_jagged_bmm_raw(
    offsets: torch.Tensor,  # [E+1]
    index: torch.Tensor,  # [L, A]
    jagged: torch.Tensor,  # [L, K]
    weight: torch.Tensor,  # [E, K, N]
    bias: Optional[torch.Tensor],  # [E, N]
) -> torch.Tensor:  # [L*A, N]
    E = weight.shape[0]
    A = index.shape[-1]
    output = torch.cat(
        [
            jagged[index.view(-1)[offsets[e] : offsets[e + 1]] // A] @ weight[e]
            + (bias[e] if bias is not None else 0)
            for e in range(E)
        ],
        dim=0,
    )
    return output


def pytorch_index_select_jagged_bmm_fp8(
    offsets: torch.Tensor,  # [E+1]
    index: torch.Tensor,  # [L, A]
    jagged: torch.Tensor,  # [L, K]
    weight: torch.Tensor,  # [E, K, N]
    bias: Optional[torch.Tensor],  # [E, N]
) -> torch.Tensor:  # [L*A, N]
    assert jagged.dtype == weight.dtype and (bias is None or jagged.dtype == bias.dtype)
    output_type = jagged.dtype
    weight_t = weight.permute(0, 2, 1)  # [E, N, K]
    # Quantize weight and jagged tensors
    with torch.no_grad():
        # weight_t_fp8: [E, N, K], weight_t_fp8_scale: [E, N]
        weight_t_fp8, weight_t_fp8_scale = pytorch_rowwise_quant_fp8(weight_t)
        # jagged_fp8: [L, K], jagged_fp8_scale: [L]
        jagged_fp8, jagged_fp8_scale = pytorch_rowwise_quant_fp8(jagged)
    # Calculate the number of tokens per expert
    partition_sizes: List[int] = (offsets[1:] - offsets[:-1]).tolist()  # [E]
    # Flatten index and map each element to its corresponding token index
    index = index.view(-1) // index.shape[-1]  # [L*A]
    # Gather tokens for each expert based on the index
    # jagged_list: list of tensors, each of shape [num_tokens_per_expert[i], K]
    jagged_list: List[torch.Tensor] = list(
        jagged_fp8[index].contiguous().split(partition_sizes)
    )  # list of tensors, each of shape [num_tokens_per_expert[i], K]
    jagged_scale_list: List[torch.Tensor] = list(
        jagged_fp8_scale[index].contiguous().split(partition_sizes)
    )
    # Apply a linear transformation for each expert's tokens
    output_list: List[torch.Tensor] = [
        (
            torch.matmul(  # torch does not support fp8 matmul, cast to fp16 and back to fp8 instead
                jagged_list[i].to(torch.bfloat16), weight_t_fp8[i].T.to(torch.bfloat16)
            )
            # .to(torch.float8_e4m3fn)
            / (jagged_scale_list[i][:, None] * weight_t_fp8_scale[i][None, :])
        ).to(output_type)
        + (bias[i] if bias is not None else 0)
        for i in range(len(jagged_list))
    ]  # list of tensors, each of shape [num_tokens_per_expert[i], N]
    # Concatenate the outputs from all experts to form the final output tensor
    return torch.cat(output_list, dim=0)  # [L*A, N]
