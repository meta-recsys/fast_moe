# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-strict


import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_moe.kernels.moe import index_select_jagged_bmm, silu_jagged_bmm_combine
from fast_moe.kernels.utils import KernelType
from fast_moe.utils.utils import _create_fused_mlp_weights
from torch.autograd.profiler import record_function


# Define the Gating Network class
class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int) -> None:
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x), dim=2)


@torch.fx.wrap
def fx_infer_max_len(
    lengths: torch.Tensor,
) -> int:
    # Do not call ".item()" to avoid problems for lowering
    max_len = int(lengths.max())
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range [0, 10**9)
        torch._check_is_size(max_len)
        torch._check(max_len < 10**9)
        torch._check(max_len > 0)
    return max_len


# Define the Mixture of Experts Layer class
class FastMoELayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        kernel: KernelType = KernelType.PYTORCH,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super(FastMoELayer, self).__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.num_experts: int = num_experts
        self.kernel: KernelType = kernel
        self.dtype: torch.dtype = dtype

        self.gate = GatingNetwork(input_dim, num_experts)
        self._experts_hidden_w = torch.nn.Parameter(
            data=_create_fused_mlp_weights(
                self.num_experts,
                self.hidden_dim,
                input_dim,
            ).to(dtype=self.dtype)
        )
        self._experts_hidden_bias = torch.nn.Parameter(
            data=torch.empty(
                [self.num_experts, self.hidden_dim],
                dtype=self.dtype,
            ).fill_(0.0)
        )

        self._experts_w = torch.nn.Parameter(
            data=_create_fused_mlp_weights(
                self.num_experts,
                output_dim,
                self.hidden_dim,
            ).to(dtype=self.dtype)
        )
        self._experts_bias = torch.nn.Parameter(
            data=torch.empty(
                [self.num_experts, output_dim],
                dtype=self.dtype,
            ).fill_(0.0)
        )

    def _select_top_k_experts(
        self, x: torch.Tensor, num_experts_per_tok: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-k experts for each token based on gating scores."""
        gating_scores = self.gate(x)
        _, topk_indices = gating_scores.topk(num_experts_per_tok, dim=2, sorted=False)

        # Create mask for selected experts and normalize scores
        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        gating_scores = gating_scores * mask
        gating_scores = F.normalize(gating_scores, p=1, dim=2)

        return gating_scores, topk_indices

    def _compute_routing_metadata(
        self,
        x: torch.Tensor,
        gating_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        num_experts_per_tok: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int
    ]:
        """Compute routing indices and metadata for expert dispatch."""
        # Reshape tensors for processing
        gating_scores = gating_scores.view(-1, self.num_experts)
        topk_indices = topk_indices.view(-1, num_experts_per_tok)
        load = gating_scores > 0

        _, gate_index = topk_indices.view(-1).sort(stable=True)
        # shape [L*K]: for each row in Y, which token it is from
        token_index: torch.Tensor = gate_index // topk_indices.size(1)
        # shape [E]
        lengths: torch.Tensor = load.sum(0)

        # We only care about the topk gating scores for our computation
        gating_scores = gating_scores.gather(1, topk_indices)

        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        reverse_token_index: torch.Tensor = token_index.sort(
            stable=True,
        )[1].view_as(topk_indices)
        max_seq_len = fx_infer_max_len(lengths)

        return (
            gate_index,
            token_index,
            gating_scores,
            offsets,
            reverse_token_index,
            max_seq_len,
        )

    def _apply_expert_layers(
        self,
        x: torch.Tensor,
        gate_index: torch.Tensor,
        token_index: torch.Tensor,
        gating_scores: torch.Tensor,
        offsets: torch.Tensor,
        reverse_token_index: torch.Tensor,
        max_seq_len: int,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Apply expert neural network layers using optimized kernels."""
        with record_function("## expert_forward##"):
            x_flat = x.view(-1, self.input_dim)

            # First layer: input -> hidden with SiLU activation
            expert_hidden = index_select_jagged_bmm(
                max_seq_len=max_seq_len,
                offsets=offsets,
                index=gate_index.view(-1, num_experts_per_tok),
                jagged=x_flat,
                weight=self._experts_hidden_w.permute(0, 2, 1).to(x.dtype),
                bias=self._experts_hidden_bias.to(x.dtype),
                kernel=self.kernel,
            )

            # Second layer: hidden -> output with gating combination
            y = silu_jagged_bmm_combine(
                max_seq_len=max_seq_len,
                offsets=offsets,
                jagged=expert_hidden,
                weight=self._experts_w.permute(0, 2, 1).to(x.dtype),
                bias=self._experts_bias.to(x.dtype),
                index=token_index,
                reverse_index=reverse_token_index,
                gating_scores=gating_scores,
                gates_index=gate_index,
                kernel=self.kernel,
            )

        return y

    def forward(
        self,
        x: torch.Tensor,
        num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Mixture of Experts forward pass with optimized kernels."""
        original_shape = x.shape

        # Step 1: Select top-k experts per token
        gating_scores, topk_indices = self._select_top_k_experts(x, num_experts_per_tok)

        # Step 2: Compute routing indices and metadata
        (
            gate_index,
            token_index,
            gating_scores,
            offsets,
            reverse_token_index,
            max_seq_len,
        ) = self._compute_routing_metadata(
            x, gating_scores, topk_indices, num_experts_per_tok
        )

        # Step 3: Apply expert layers
        y = self._apply_expert_layers(
            x,
            gate_index,
            token_index,
            gating_scores,
            offsets,
            reverse_token_index,
            max_seq_len,
            num_experts_per_tok,
        )

        # Restore original shape
        y = y.view(original_shape[0], original_shape[1], self.output_dim)
        return y
