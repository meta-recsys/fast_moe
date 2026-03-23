# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# pyre-strict

import unittest

import torch
from fast_moe.dev_settings import set_dev_mode
from fast_moe.kernels.utils import gpu_unavailable
from hypothesis import given, settings, strategies as st, Verbosity
from torch.functional import F


# buck2 test -c fbcode.disable_re_tests=True @//mode/opt fast_moe/kernels/tests:general_ops_test -- --print-passing-details


class GeneralOpsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        N=st.sampled_from([391336 * 2 * 1024, 32768 * 32 * 128]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_triton_silu_backward(
        self,
        N: int,
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        from fast_moe.kernels.triton.triton_general_ops import triton_silu_backward

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(0)

        x = (
            torch.empty(N, device=device, dtype=dtype)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        y = F.silu(x)
        dy = torch.randn_like(y) * 0.01
        y.backward(dy)

        torch.cuda.manual_seed(0)
        dx_triton = triton_silu_backward(x, dy)

        torch.testing.assert_close(
            x.grad,
            dx_triton,
            atol=None,
            rtol=None,
        )
