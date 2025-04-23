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
            assert (
                weight.numel() == weight_index.numel() == value.shape[0]
            ), f"{weight.shape=}, {weight_index.shape=}, {value.shape=}"
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
    SWiGlu(Swish-Gated Linear Unit) baseded on index_select_jagged_bmm
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

    # Apply a linear transformation for each expert's tokens
    output_list: List[torch.Tensor] = [
        F.linear(
            jagged_list[i],
            weight_t[i],
            bias[i] if bias is not None else None,
        )
        for i in range(len(jagged_list))
    ]  # list of tensors, each of shape [num_tokens_per_expert[i], N]
    output_list_p: List[torch.Tensor] = [
        F.linear(
            jagged_list[i],
            weight_p_t[i],
            bias_p[i] if bias_p is not None else None,
        )
        for i in range(len(jagged_list))
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
