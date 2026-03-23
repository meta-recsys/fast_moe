# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import List, Optional

import torch
from torch.fx._symbolic_trace import is_fx_tracing
from torch.nn import functional as F


def pytorch_mul_merge_k_add(
    index: torch.Tensor,
    value: torch.Tensor,
    k: int,
    weight: Optional[torch.Tensor] = None,
    weight_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weight is not None:
        if not is_fx_tracing():
            assert weight_index is not None
            assert weight.numel() == weight_index.numel() == value.shape[0], (
                f"{weight.shape=}, {weight_index.shape=}, {value.shape=}"
            )
        value = value.mul(weight.contiguous().view(-1)[weight_index].view(-1, 1))

    zeros = torch.zeros(
        index.shape[0] // k,
        value.shape[1],
        device=value.device,
        # use FP32 as accumulation type to avoid underflow
        dtype=torch.float32,
    )
    zeros.requires_grad_(True)
    return zeros.index_add(0, index, value.float()).to(value.dtype)


def pytorch_index_select_jagged_bmm(
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Performs a batched matrix multiplication with expert-specific weights and biases.
    This function processes input tokens dispatched to multiple experts, applying a linear transformation using expert-specific weights and biases. The input tokens are gathered based on the provided index, and the outputs are concatenated to form the final result.
    Dimensions:
        L: number of input tokens
        E: number of total experts
        A: number of activated experts
        K: input dimension
        N: output dimension
    Args:
        offsets (torch.Tensor): A tensor of shape [E+1] representing the cumulative number of tokens dispatched to each expert.
        index (torch.Tensor): A tensor of shape [L, A] that is flattened and sorted by expert. Each entry is calculated as token_id * A + [0, A).
        jagged (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
        weight (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
        bias (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
    Returns:
        torch.Tensor: A tensor of shape [L*A, N] containing the output after applying the linear transformation for each expert's tokens.
    """
    weight_t = weight.permute(0, 2, 1)  # [E, N, K]
    # Calculate the number of tokens per expert
    partition_sizes: List[int] = (offsets[1:] - offsets[:-1]).tolist()  # [E]
    # Flatten index and map each element to its corresponding token index
    index = index.view(-1) // index.shape[-1]  # [L*A]
    # Gather tokens for each expert based on the index
    jagged_list: List[torch.Tensor] = list(
        jagged[index].contiguous().split(partition_sizes)
    )  # list of tensors, each of shape [num_tokens_per_expert[i], K]
    # Apply a linear transformation for each expert's tokens
    output_list: List[torch.Tensor] = [
        F.linear(
            jagged_list[i],
            weight_t[i],
            bias[i] if bias is not None else None,
        )
        for i in range(len(jagged_list))
    ]  # list of tensors, each of shape [num_tokens_per_expert[i], N]
    # Concatenate the outputs from all experts to form the final output tensor
    return torch.cat(output_list, dim=0)  # [L*A, N]


def pytorch_index_select_jagged_bmm_swiglu(
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_p: torch.Tensor,
    bias_p: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    SWiGlu(Swish-Gated Linear Unit) based on index_select_jagged_bmm
    Dimensions:
        L: number of input tokens
        E: number of total experts
        A: number of activated experts
        K: input dimension
        N: output dimension
    Args:
        offsets (torch.Tensor): A tensor of shape [E+1] representing the cumulative number of tokens dispatched to each expert.
        index (torch.Tensor): A tensor of shape [L, A] that is flattened and sorted by expert. Each entry is calculated as token_id * A + [0, A).
        jagged (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
        weight (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
        bias (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
        weight_p (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
        bias_p (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
    Returns:
        torch.Tensor: A tensor of shape [L*A, N] containing the output after applying the linear transformation for each expert's tokens.
    """

    weight_t = weight.permute(0, 2, 1)  # [E, N, K]
    weight_p_t = weight_p.permute(0, 2, 1)  # [E, N, K]

    # Calculate the number of tokens per expert
    partition_sizes: List[int] = (offsets[1:] - offsets[:-1]).tolist()  # [E]
    # Flatten index and map each element to its corresponding token index
    index = index.view(-1) // index.shape[-1]  # [L*A]
    # Gather tokens for each expert based on the index
    jagged_list: List[torch.Tensor] = list(
        jagged[index].contiguous().split(partition_sizes)
    )  # list of tensors, each of shape [num_tokens_per_expert[i], K]

    # Apply linear transformations and SWiGLU activation directly in a single loop
    swiglu_output_list: List[torch.Tensor] = [
        F.silu(
            F.linear(
                jagged_list[i],
                weight_t[i],
                bias[i] if bias is not None else None,
            )
        )
        * F.linear(
            jagged_list[i],
            weight_p_t[i],
            bias_p[i] if bias_p is not None else None,
        )
        for i in range(len(jagged_list))
    ]  # list of tensors, each of shape [num_tokens_per_expert[i], N]

    # Concatenate the outputs from all experts to form the final output tensor
    return torch.cat(swiglu_output_list, dim=0)  # [L*A, N]


def pytorch_index_select_jagged_gating_bmm(
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
    Out = Silu(Jagged_A @ Dense_A + Bias_A) * (Jagged_B @ Dense_B + Bias_B)

    Dimensions:
        L: number of input tokens
        E: number of total experts
        A: number of activated experts
        K: input dimension
        N: output dimension
    Args:
        offsets (torch.Tensor): A tensor of shape [E+1] representing the cumulative number of tokens dispatched to each expert.
        index (torch.Tensor): A tensor of shape [L, A] that is flattened and sorted by expert. Each entry is calculated as token_id * A + [0, A).
        jagged_a (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
        jagged_b (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
        weight_a (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
        weight_b (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
        bias_a (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
        bias_b (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
    Returns:
        torch.Tensor: A tensor of shape [L*A, N] containing the output after applying the linear transformation for each expert's tokens.
    """

    weight_a_t = weight_a.permute(0, 2, 1)  # [E, N, K]
    weight_b_t = weight_b.permute(0, 2, 1)  # [E, N, K]

    # Calculate the number of tokens per expert
    partition_sizes: List[int] = (offsets[1:] - offsets[:-1]).tolist()  # [E]
    # Flatten index and map each element to its corresponding token index
    index = index.view(-1) // index.shape[-1]  # [L*A]
    # Gather tokens for each expert based on the index
    jagged_a_list: List[torch.Tensor] = list(
        jagged_a[index].contiguous().split(partition_sizes)
    )  # list of tensors, each of shape [num_tokens_per_expert[i], K]
    if jagged_a is jagged_b:
        jagged_b_list = jagged_a_list
    else:
        jagged_b_list: List[torch.Tensor] = list(
            jagged_b[index].contiguous().split(partition_sizes)
        )  # list of tensors, each of shape [num_tokens_per_expert[i], K]

    # Apply a linear transformation for each expert's tokens
    output_list: List[torch.Tensor] = [
        F.linear(
            jagged_a_list[i],
            weight_a_t[i],
            bias_a[i] if bias_a is not None else None,
        )
        for i in range(len(jagged_a_list))
    ]  # list of tensors, each of shape [num_tokens_per_expert[i], N]
    output_list_p: List[torch.Tensor] = [
        F.linear(
            jagged_b_list[i],
            weight_b_t[i],
            bias_b[i] if bias_b is not None else None,
        )
        for i in range(len(jagged_b_list))
    ]

    # Concatenate the outputs from all experts to form the final output tensor
    alpha = torch.cat(output_list, dim=0)  # [L*A, N]
    beta = torch.cat(output_list_p, dim=0)  # [L*A, N]

    return F.silu(alpha) * beta


def pytorch_index_select_jagged_bmm_3D(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    3D version of pytorch_index_select_jagged_bmm
    weight: [E, D_in, D_out]
    jagged: [L, E, D_in]
    """
    weight = weight.permute(0, 2, 1)
    partition_sizes: List[int] = (offsets[1:] - offsets[:-1]).tolist()
    index = index.view(-1) // index.shape[-1]
    jagged_list: List[torch.Tensor] = list(
        jagged[index].contiguous().split(partition_sizes)
    )
    output_list: List[torch.Tensor] = [
        F.linear(
            jagged_list[i][:, i, :],
            weight[i],
            bias[i] if bias is not None else None,
        )
        for i in range(len(jagged_list))
    ]

    return torch.cat(output_list, dim=0)


def pytorch_silu_jagged_bmm(
    offsets: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    has_silu: bool = True,
) -> torch.Tensor:
    partition_sizes: List[int] = (offsets[1:] - offsets[:-1]).tolist()
    if has_silu:
        jagged = F.silu(jagged)
    jagged_list: List[torch.Tensor] = list(jagged.split(partition_sizes))
    hidden_list: List[torch.Tensor] = [
        F.linear(
            jagged_list[i],
            weight[i],
            bias[i] if bias is not None else None,
        )
        for i in range(len(jagged_list))
    ]

    return torch.cat(hidden_list, dim=0)


def pytorch_silu_jagged_bmm_combine(
    offsets: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    index: torch.Tensor,
    k: int,
    gating_scores: Optional[torch.Tensor] = None,
    gates_index: Optional[torch.Tensor] = None,
    has_silu: bool = True,
) -> torch.Tensor:
    bmm_out = pytorch_silu_jagged_bmm(
        offsets=offsets,
        jagged=jagged,
        weight=weight.permute(0, 2, 1),
        bias=bias,
        has_silu=has_silu,
    )

    return pytorch_mul_merge_k_add(
        index=index,
        value=bmm_out,
        k=k,
        weight=gating_scores,
        weight_index=gates_index,
    )


def pytorch_fused_jagged_bmm_swiglu_combine(
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_p: torch.Tensor,
    bias_p: Optional[torch.Tensor],
    weight_out: torch.Tensor,
    bias_out: Optional[torch.Tensor],
    k: int,
    gates: Optional[torch.Tensor] = None,
    gates_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused kernel combining index_selected_jagged_bmm and additional activation layer.
    This combines the kernel from index_select_jagged_bmm and silu_jagged_bmm
    Dimensions:
            L: number of tokens
            E: number of experts
            K: activated number of experts
            M: input dimension into fused module
            N: intermediate dimension of what used to be index_select_jagged_bmm -> silu_jagged_bmm_combine
            D: output dimension of fused module
    Args:
        offsets (torch.Tensor): A tensor of shape [E+1] representing the cumulative number of tokens dispatched to each expert.
        index (torch.Tensor): A tensor of shape [L, K] that is flattened and sorted by expert. Each entry is calculated as token_id * K + [0, K).
        jagged (torch.Tensor): A tensor of shape [L, K] representing the input tokens.
        weight (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
        bias (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
        weight_p (torch.Tensor): A tensor of shape [E, K, N] containing the weights for each expert.
        bias_p (torch.Tensor): A tensor of shape [E, N] containing the biases for each expert.
        weight_out (torch.Tensor): A tensor of shape [E, N, D] containing the output weights for each expert.
        bias_out (torch.Tensor): A tensor of shape [E, D] containing the output biases for each expert.
        k (int): The factor for merging.
        gates (torch.Tensor): A tensor of shape [L*K 1] containing the gates for each token.
        gates_index (torch.Tensor): A tensor of shape [L*K] containing the indices for the gates.
    Returns:
        torch.Tensor: A tensor of shape [L*K, D] containing the output after applying the linear transformations and activation.
    """
    weight_t = weight.permute(0, 2, 1)  # [E, N, K]
    weight_p_t = weight_p.permute(0, 2, 1)  # [E, N, K]
    weight_out_t = weight_out.permute(0, 2, 1)  # [E, D, N]

    # Calculate the number of tokens per expert
    partition_sizes: List[int] = (offsets[1:] - offsets[:-1]).tolist()  # [E]
    # Flatten index and map each element to its corresponding token index
    index = index.view(-1) // index.shape[-1]

    # Gather tokens for each expert based on the index
    jagged_list: List[torch.Tensor] = list(
        jagged[index].contiguous().split(partition_sizes)
    )  # list of tensors, each of shape [num_tokens_per_expert[i], K]

    # Apply linear transformations, SWiGLU activation, and the second linear transformation directly in a single loop
    final_output_list: List[torch.Tensor] = [
        F.linear(
            F.silu(
                F.linear(
                    jagged_list[i],
                    weight_t[i],
                    bias[i] if bias is not None else None,
                )
            )
            * F.linear(
                jagged_list[i],
                weight_p_t[i],
                bias_p[i] if bias_p is not None else None,
            ),
            weight_out_t[i],
            bias_out[i] if bias_out is not None else None,
        )
        for i in range(len(jagged_list))
    ]  # list of tensors, each of shape [num_tokens_per_expert[i], D]

    # Concatenate the outputs from all experts to form the final output tensor
    final_output = torch.cat(final_output_list, dim=0)  # [L*K, D]

    # Combine with gates if provided
    return pytorch_mul_merge_k_add(
        index=index,
        value=final_output,
        k=k,
        weight=gates,
        weight_index=gates_index,
    )
