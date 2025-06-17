#!/usr/bin/env python3

# pyre-strict

# import random
import unittest
from typing import Tuple

import torch
from fast_moe.dev_settings import set_dev_mode

from fast_moe.kernels.utils import gpu_unavailable
from hypothesis import given, settings, strategies as st, Verbosity

# buck2 test -c fbcode.disable_re_tests=True @//mode/opt fast_moe/kernels/tests:quant_fp8_test -- --print-passing-details


class Fp8RowwiseTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        shape=st.sampled_from([(32, 16), (32, 8, 16), (7, 9, 13), (51, 5)]),
        transpose=st.booleans(),
        dtype=st.just(
            torch.bfloat16
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else torch.float32
        ),
        contiguous=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=500,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_transpose_rowwise_quant_fp8(self, *args, **kwargs) -> None:
        self._test_transpose_rowwise_quant_fp8(
            *args,
            **kwargs,
        )

    def _test_transpose_rowwise_quant_fp8(
        self,
        shape: Tuple[int, ...],
        transpose: bool,
        dtype: torch.dtype,
        contiguous: bool,
    ) -> None:
        set_dev_mode(True)
        a = torch.randn(shape, device="cuda", dtype=dtype)
        if not contiguous:
            a = (
                a.transpose(-2, -1)
                .contiguous()
                .transpose(-2, -1)
                .detach()
                .clone()
                .requires_grad_()
            )

        from fast_moe.kernels.pytorch.quant_fp8 import (
            pytorch_rowwise_quant_fp8,
            pytorch_transpose_rowwise_quant_fp8,
        )
        from fast_moe.kernels.triton.triton_quant_fp8 import (
            triton_rowwise_quant_fp8,
            triton_transpose_rowwise_quant_fp8,
        )

        if transpose:
            a_fp8_0, a_scale_0 = pytorch_transpose_rowwise_quant_fp8(a)
            a_fp8_1, a_scale_1 = triton_transpose_rowwise_quant_fp8(a)
        else:
            a_fp8_0, a_scale_0 = pytorch_rowwise_quant_fp8(a)
            a_fp8_1, a_scale_1 = triton_rowwise_quant_fp8(a)
        torch.testing.assert_close(
            a_fp8_0.to(torch.float32), a_fp8_1.to(torch.float32), check_stride=True
        )
        torch.testing.assert_close(a_scale_0, a_scale_1, check_stride=True)
