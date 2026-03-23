# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-unsafe

"""
Numerics test to verify parity between PyTorch and Triton kernel implementations.

This test ensures that PyTorch and Triton kernels produce identical outputs
when given the same inputs, verifying correctness of optimized Triton kernels.
"""

import unittest

import hypothesis.strategies as st
import torch
from fast_moe.kernels.utils import gpu_unavailable, KernelType
from fast_moe.modules.fast_moe_module import FastMoELayer
from hypothesis import given, settings, Verbosity


class TestKernelParity(unittest.TestCase):
    """Test class for verifying parity between PyTorch and Triton kernels."""

    @given(
        batch_size=st.sampled_from([4, 8]),
        seq_len=st.sampled_from([4, 8, 16]),
        input_dim=st.sampled_from([16, 32]),
        hidden_dim=st.sampled_from([16, 32]),
        output_dim=st.sampled_from([16, 32]),
        num_experts=st.sampled_from([4, 8]),
        num_experts_per_tok=st.sampled_from([1, 2]),
        tolerance=st.just(1e-4),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_fastmoe_end_to_end_kernel_parity(self, *args, **kwargs) -> None:
        """Test end-to-end parity using FastMoELayer with different kernels."""
        self._test_fastmoe_end_to_end_kernel_parity(*args, **kwargs)

    def _test_fastmoe_end_to_end_kernel_parity(
        self,
        batch_size: int,
        seq_len: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        tolerance: float,
    ) -> None:
        """Test end-to-end kernel parity using FastMoELayer."""
        device = torch.device("cuda")
        torch.manual_seed(42)

        # Create input tensor and move to GPU
        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        # Create FastMoELayer instance with PyTorch kernel
        fast_moe = FastMoELayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            kernel=KernelType.PYTORCH,
        ).to(device)

        torch.manual_seed(42)

        # Create Triton version with same weights
        fast_moe_triton = FastMoELayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            kernel=KernelType.TRITON,
        ).to(device)

        with torch.no_grad():
            # Test PyTorch version (default kernel)
            output_pytorch = fast_moe(x, num_experts_per_tok)

            # Test Triton version
            output_triton = fast_moe_triton(x, num_experts_per_tok)

        self.assertTrue(torch.allclose(output_pytorch, output_triton, atol=tolerance))


if __name__ == "__main__":
    unittest.main()
