# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-unsafe

"""
Numerics test to verify parity between MoeBase and FastMoELayer modules.

This test ensures that both MoE implementations produce identical outputs
when given the same input and identical frozen weights.
"""

import unittest

import hypothesis.strategies as st
import torch
from fast_moe.modules.fast_moe_module import FastMoELayer
from fast_moe.modules.moe_base import Activations, MoeBase
from hypothesis import given, settings, Verbosity


class TestMoEParity(unittest.TestCase):
    """Test class for verifying parity between MoE implementations."""

    @given(
        batch_size=st.sampled_from([4, 8]),
        num_tokens=st.sampled_from([4, 8, 16]),
        input_dim=st.sampled_from([10, 16]),
        hidden_dim=st.sampled_from([10, 16]),
        output_dim=st.sampled_from([10, 16]),
        num_experts=st.sampled_from([4, 8]),
        num_experts_per_token=st.sampled_from([1, 2]),
        weight_value=st.just(0.5),
        bias_value=st.sampled_from([0.0, 0.5]),
        tolerance=st.just(1e-5),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=5, deadline=None)
    def test_numerics_parity_with_silu_activation(self, *args, **kwargs) -> None:
        """Test parity between modules using GeLU activation with parameterized setup."""
        self._test_numerics_parity_with_silu_activation(*args, **kwargs)

    def _test_numerics_parity_with_silu_activation(
        self,
        batch_size: int,
        num_tokens: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        weight_value: float,
        bias_value: float,
        tolerance: float,
    ) -> None:
        """Test parity between modules using GeLU activation."""
        torch.manual_seed(42)
        x = torch.randn(batch_size, num_tokens, input_dim)

        moe_base = MoeBase(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            expert_activation=Activations.SILU,
        )

        # Freeze weights in moe_base to be the same across all experts
        for expert in moe_base.experts:
            layers = [expert.fc1, expert.fc2]

            for layer in layers:
                # pyre-ignore[16]
                torch.nn.init.constant_(layer.weight, weight_value)
                # pyre-ignore[16]
                torch.nn.init.constant_(layer.bias, bias_value)

        # Freeze gate weights
        torch.nn.init.constant_(moe_base.gate.gate.weight, weight_value)
        torch.nn.init.constant_(moe_base.gate.gate.bias, bias_value)

        fast_moe = FastMoELayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
        )

        # Freeze weights in fast_moe to be the same across all experts
        torch.nn.init.constant_(fast_moe._experts_hidden_w, weight_value)
        torch.nn.init.constant_(fast_moe._experts_hidden_bias, bias_value)
        torch.nn.init.constant_(fast_moe._experts_w, weight_value)
        torch.nn.init.constant_(fast_moe._experts_bias, bias_value)

        # Freeze gate weights
        torch.nn.init.constant_(fast_moe.gate.gate.weight, weight_value)
        torch.nn.init.constant_(fast_moe.gate.gate.bias, bias_value)

        with torch.no_grad():
            output_base = moe_base(x, num_experts_per_token)
            output_fast = fast_moe(x, num_experts_per_token)

        self.assertTrue(torch.allclose(output_base, output_fast, atol=tolerance))
