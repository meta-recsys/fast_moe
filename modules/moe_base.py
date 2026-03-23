# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-strict

from enum import Enum
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Activations(Enum):
    RELU = "relu"
    SILU = "silu"
    GELU = "gelu"


ACTIVATIONS_MAPPING: Dict[Activations, Callable[[torch.Tensor], torch.Tensor]] = {
    Activations.RELU: F.relu,
    Activations.SILU: F.silu,
    Activations.GELU: F.gelu,
}


# Define the Expert class
class Expert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: Activations = Activations.RELU,
    ) -> None:
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ACTIVATIONS_MAPPING[self.activation]((self.fc1(x)))
        return self.fc2(x)


# Define the Gating Network class
class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int) -> None:
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x), dim=2)


# Define the Mixture of Experts Layer class
class MoeBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        expert_activation: Activations = Activations.RELU,
    ) -> None:
        super(MoeBase, self).__init__()
        self.experts = nn.ModuleList(
            [
                Expert(input_dim, hidden_dim, output_dim, expert_activation)
                for _ in range(num_experts)
            ]
        )
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x: torch.Tensor, num_experts_per_token: int) -> torch.Tensor:
        gating_scores = self.gate(x)
        topk_gating_scores, topk_indices = gating_scores.topk(
            num_experts_per_token, dim=2, sorted=False
        )
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=2)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum("bte,bteo->bto", gating_scores, expert_outputs)
        return output
