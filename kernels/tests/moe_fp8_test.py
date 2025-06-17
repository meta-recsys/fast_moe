#!/usr/bin/env python3

# pyre-strict

# import random
import unittest
from typing import Optional

import torch
from fast_moe.dev_settings import set_dev_mode

from fast_moe.kernels.utils import (
    gpu_unavailable,
    KernelType,
    to_fp32_if_pytorch_kernel,
)
from hypothesis import given, seed, settings, strategies as st, Verbosity

# buck2 test -c fbcode.disable_re_tests=True @//mode/opt fast_moe/kernels/tests:moe_fp8_test -- --print-passing-details


class MOEFp8Test(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @seed(0)
    # pyre-ignore
    @given(
        max_seq_len=st.sampled_from([32, 64, 200]),
        min_seq_len=st.just(10),
        E=st.sampled_from([4, 8]),
        D_in=st.sampled_from([16, 32, 64]),
        D_out=st.sampled_from([16, 32, 64]),
        dtype=st.sampled_from([torch.bfloat16, torch.float32]),
        allow_tf32=st.just(False),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=30,
        deadline=None,
    )
    def test_silu_jagged_fp8(
        self,
        *args,  # pyre-ignore[2]
        **kwargs,  # pyre-ignore[2]
    ) -> None:
        self._test_silu_jagged_fp8(
            *args,
            **kwargs,
            atol=1e-1,
            rtol=1.5e-1,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
        )

    def _test_silu_jagged_fp8(
        self,
        min_seq_len: int,
        max_seq_len: int,
        E: int,
        D_in: int,
        D_out: int,
        dtype: torch.dtype,
        allow_tf32: bool,
        ref_kernel: KernelType,
        real_kernel: KernelType,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        from fast_moe.kernels.moe_fp8 import silu_jagged_fp8

        torch.cuda.manual_seed(0)

        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        lengths = torch.randint(
            low=min_seq_len,
            high=max_seq_len + 1,
            size=(E,),
        )
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        lengths = lengths.to(device)
        offsets = offsets.to(device)
        max_seq_len = int(lengths.max().item())

        jagged_size = int(lengths.sum().item())
        jagged = (
            torch.empty((jagged_size, D_in), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        base_silu_jagged = torch.empty_like(jagged)
        base_silu_jagged_fp8 = torch.empty_like(jagged, dtype=torch.float8_e4m3fn)
        base_silu_jagged_scale = torch.empty(
            (jagged_size,), dtype=torch.float32, device=device
        )

        out_base = silu_jagged_fp8(
            seq_offsets=offsets,
            Jagged=jagged,
            max_seq_len=max_seq_len,
            K=D_in,
            Silu_Jagged=base_silu_jagged,
            Silu_Jagged_fp8=base_silu_jagged_fp8,
            Silu_Jagged_Scale=base_silu_jagged_scale,
            kernel=KernelType.PYTORCH,
        )

        out_test = silu_jagged_fp8(
            seq_offsets=offsets,
            Jagged=jagged,
            max_seq_len=max_seq_len,
            K=D_in,
            Silu_Jagged=base_silu_jagged,
            Silu_Jagged_fp8=base_silu_jagged_fp8,
            Silu_Jagged_Scale=base_silu_jagged_scale,
            kernel=KernelType.TRITON,
        )

        torch.testing.assert_close(
            out_test.to(torch.float32), out_base.to(torch.float32), atol=atol, rtol=rtol
        )

    @unittest.skipIf(*gpu_unavailable)
    @seed(0)
    # pyre-ignore
    @given(
        E=st.sampled_from([4, 8]),
        D_in=st.sampled_from([16, 32, 64]),
        D_out=st.sampled_from([16, 32, 64]),
        dtype=st.sampled_from([torch.bfloat16, torch.float32]),
        allow_tf32=st.just(False),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=30,
        deadline=None,
    )
    def test_bmm_weight_rowwise_quant_fp8(
        self,
        *args,  # pyre-ignore[2]
        **kwargs,  # pyre-ignore[2]
    ) -> None:
        self._test_bmm_weight_rowwise_quant_fp8(
            *args,
            **kwargs,
            atol=1e-1,
            rtol=1.5e-1,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
        )

    def _test_bmm_weight_rowwise_quant_fp8(
        self,
        E: int,
        D_in: int,
        D_out: int,
        dtype: torch.dtype,
        allow_tf32: bool,
        ref_kernel: KernelType,
        real_kernel: KernelType,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        from fast_moe.kernels.moe_fp8 import bmm_weight_rowwise_quant_fp8

        torch.cuda.manual_seed(0)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        weight = (
            torch.empty((E, D_in, D_out), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        out_base = bmm_weight_rowwise_quant_fp8(
            weight=weight,
            kernel=KernelType.PYTORCH,
        )

        out_test = bmm_weight_rowwise_quant_fp8(
            weight=weight,
            kernel=KernelType.TRITON,
        )

        torch.testing.assert_close(
            out_test.to(torch.float32), out_base.to(torch.float32), atol=atol, rtol=rtol
        )

    @unittest.skipIf(*gpu_unavailable)
    @seed(0)
    # pyre-ignore
    @given(
        max_seq_len=st.sampled_from([32, 64, 200]),
        min_seq_len=st.just(32),
        E=st.sampled_from([4, 8]),
        D_in=st.sampled_from([128, 256]),
        D_out=st.sampled_from([128, 256]),
        dtype=st.sampled_from([torch.bfloat16]),
        contiguous=st.just(True),
        allow_tf32=st.just(False),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=30,
        deadline=None,
    )
    def test_jagged_bmm_fp8(
        self,
        *args,  # pyre-ignore[2]
        **kwargs,  # pyre-ignore[2]
    ) -> None:
        self._test_jagged_bmm_fp8(
            *args,
            **kwargs,
            atol=5e-1,
            rtol=1.5e-1,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
            test_backward=True,
        )

    def _test_jagged_bmm_fp8(
        self,
        min_seq_len: int,
        max_seq_len: int,
        E: int,
        D_in: int,
        D_out: int,
        dtype: torch.dtype,
        contiguous: bool,
        allow_tf32: bool,
        ref_kernel: KernelType,
        real_kernel: KernelType,
        test_backward: bool,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        from fast_moe.kernels.moe_fp8 import silu_jagged_bmm_fp8

        torch.cuda.manual_seed(0)

        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        lengths = torch.randint(
            low=min_seq_len,
            high=max_seq_len + 1,
            size=(E,),
        )
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        lengths = lengths.to(device)
        offsets = offsets.to(device)
        max_seq_len = int(lengths.max().item())

        jagged_size = int(lengths.sum().item())
        jagged = (
            torch.empty((jagged_size, D_in), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight = (
            torch.empty((E, D_out, D_in), dtype=dtype, device=device).uniform_(
                -1.0, 1.0
            )
        ).requires_grad_()

        bias = (
            torch.empty((E, D_out), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if not contiguous:
            weight = (
                weight.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )

        jagged_test = jagged.detach().clone().requires_grad_()
        weight_test = weight.detach().clone().requires_grad_()
        bias_test = bias.detach().clone().requires_grad_()

        out_test = silu_jagged_bmm_fp8(
            seq_offsets=offsets,
            max_seq_len=max_seq_len,
            jagged=jagged_test,
            weight=weight_test,
            bias=bias_test,
            kernel=KernelType.TRITON,
        )

        out_base = silu_jagged_bmm_fp8(
            seq_offsets=offsets,
            max_seq_len=max_seq_len,
            jagged=jagged,
            weight=weight,
            bias=bias,
            kernel=KernelType.PYTORCH,
        )

        torch.testing.assert_close(
            out_test.to(torch.float32), out_base.to(torch.float32), atol=atol, rtol=rtol
        )

        if test_backward:
            dout = torch.randn_like(out_base) * 0.01
            out_base.backward(dout)
            out_test.backward(dout)

            for p_base, p_test in zip(
                [jagged, weight, bias],
                [jagged_test, weight_test, bias_test],
            ):
                torch.testing.assert_close(
                    p_base.grad,
                    p_test.grad,
                    atol=atol,
                    rtol=rtol,
                )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        L=st.sampled_from([16, 32, 100, 500]),
        E=st.sampled_from([4, 8, 16]),
        K=st.sampled_from([2, 4]),
        D_in=st.sampled_from([32, 64]),
        D_out=st.sampled_from([16, 32]),
        dtype=st.just(
            torch.bfloat16
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else torch.float32
        ),
        contiguous=st.booleans(),
        has_bias=st.booleans(),
        allow_tf32=st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=5000,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_index_select_jagged_bmm_fp8_with_raw(self, *args, **kwargs) -> None:
        self._test_index_select_jagged_bmm(
            *args,
            **kwargs,
            test_backward=True,
            atol=4e-1,
            rtol=1e-1,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
            ref_fp8=False,
            real_fp8=True,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        L=st.sampled_from([16, 32, 100, 500]),
        E=st.sampled_from([4, 8, 16]),
        K=st.sampled_from([2, 4]),
        D_in=st.sampled_from([32, 64]),
        D_out=st.sampled_from([16, 32]),
        dtype=st.just(
            torch.bfloat16
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else torch.float32
        ),
        contiguous=st.booleans(),
        has_bias=st.booleans(),
        allow_tf32=st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=5000,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_index_select_jagged_bmm_fp8_with_torch(self, *args, **kwargs) -> None:
        self._test_index_select_jagged_bmm(
            *args,
            **kwargs,
            test_backward=False,
            atol=1e-2,
            rtol=1e-2,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
            ref_fp8=True,
            real_fp8=True,
        )

    def _test_index_select_jagged_bmm(
        self,
        L: int,
        E: int,
        K: int,
        D_in: int,
        D_out: int,
        dtype: torch.dtype,
        ref_kernel: KernelType,
        real_kernel: KernelType,
        test_backward: bool,
        contiguous: bool,
        has_bias: bool,
        allow_tf32: bool,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        ref_fp8: bool = False,
        real_fp8: bool = False,
    ) -> None:
        set_dev_mode(True)
        from fast_moe.kernels.moe_fp8 import index_select_jagged_bmm

        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(0)

        gate = torch.randn(L, E, device=device).topk(K, dim=1).indices

        expert, index = gate.contiguous().view(-1).sort(stable=True)
        index = index.view(-1, K)

        zeros = torch.zeros(E, dtype=expert.dtype, device=device)
        lengths = zeros.scatter_add(0, expert, torch.ones_like(expert))

        jagged_base = (
            torch.empty((L, D_in), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight_base = (
            torch.empty((E, D_in, D_out), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        if has_bias:
            bias_base = (
                torch.empty((E, D_out), dtype=dtype, device=device)
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
            bias_test = bias_base.detach().clone().requires_grad_()
        else:
            bias_base, bias_test = None, None

        jagged_test = jagged_base.detach().clone().requires_grad_()
        weight_test = weight_base.detach().clone().requires_grad_()

        if not contiguous:
            weight_base = (
                weight_base.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )
            weight_test = (
                weight_test.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )

        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        max_seq_len = int(lengths.max().item())

        out_base = index_select_jagged_bmm(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=to_fp32_if_pytorch_kernel(jagged_base, ref_kernel),
            weight=to_fp32_if_pytorch_kernel(weight_base, ref_kernel),
            bias=to_fp32_if_pytorch_kernel(bias_base, ref_kernel)
            if bias_base is not None
            else None,
            kernel=ref_kernel,
            fp8=ref_fp8,
        ).to(jagged_base.dtype)

        torch.cuda.manual_seed(0)

        out_test = index_select_jagged_bmm(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged_test,
            weight=weight_test,
            bias=bias_test,
            kernel=real_kernel,
            fp8=real_fp8,
        )

        torch.testing.assert_close(
            out_base,
            out_test,
            atol=atol,
            rtol=rtol,
        )

        if test_backward:
            dout = torch.randn_like(out_test) * 0.01
            out_test.backward(dout)
            out_base.backward(dout)

            for p_base, p_test in zip(
                [jagged_base, weight_base],
                [jagged_test, weight_test],
            ):
                torch.testing.assert_close(
                    p_base.grad,
                    p_test.grad,
                    atol=atol,
                    rtol=rtol,
                )
            if bias_base is not None and bias_test is not None:
                torch.testing.assert_close(
                    bias_base.grad,
                    bias_test.grad,
                    atol=atol,
                    rtol=rtol,
                )
