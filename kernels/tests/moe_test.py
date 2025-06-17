#!/usr/bin/env python3

# pyre-strict

import unittest
from typing import Optional

import torch
from fast_moe.dev_settings import set_dev_mode
from fast_moe.kernels.triton.triton_moe import (
    IndexSelectJaggedBmmOption,
    SiluJaggedBmmCombineOption,
)
from fast_moe.kernels.utils import (
    gpu_unavailable,
    KernelType,
    to_fp32_if_pytorch_kernel,
)
from hypothesis import given, settings, strategies as st, Verbosity


# buck2 test -c fbcode.disable_re_tests=True @//mode/opt fast_moe/kernels/tests:moe_test -- --print-passing-details


class MOETest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        L=st.sampled_from([16, 32, 100, 500]),
        E=st.sampled_from([4, 8, 16]),
        K=st.sampled_from([2, 4]),
        D_in=st.sampled_from([12, 32, 64]),
        D_out=st.sampled_from([32, 64, 128]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        contiguous=st.booleans(),
        has_bias=st.booleans(),
        allow_tf32=st.sampled_from([False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_index_select_jagged_bmm_triton_d_jagged(self, *args, **kwargs) -> None:
        for gemm_out_type in [torch.float32, torch.bfloat16]:
            if (
                torch.cuda.get_device_capability(torch.device("cuda"))[0] < 8
                and gemm_out_type == torch.bfloat16
            ):
                continue
            # when input type is fp32, it's unreasonable to set gemm_out_type to bf16
            if kwargs["dtype"] == torch.float32 and gemm_out_type == torch.bfloat16:
                continue
            # set gemm_out_type to bf16 will decrease precision, so we set atol and rtol to be larger
            atol = 2e-3 if gemm_out_type == torch.bfloat16 else None
            rtol = 1e-2 if gemm_out_type == torch.bfloat16 else None
            for grouped_gemm in [True, False]:
                triton_option = IndexSelectJaggedBmmOption(
                    d_jagged_use_grouped_gemm=grouped_gemm,
                    d_jagged_gemm_out_type=gemm_out_type,
                )
                self._test_index_select_jagged_bmm(
                    *args,
                    **kwargs,
                    test_backward=True,
                    atol=atol,
                    rtol=rtol,
                    ref_kernel=KernelType.PYTORCH,
                    real_kernel=KernelType.TRITON,
                    test_triton_option=triton_option,
                )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        L=st.sampled_from([16, 32, 100, 500]),
        E=st.sampled_from([4, 8, 16]),
        K=st.sampled_from([2, 4]),
        D_in=st.sampled_from([12, 32, 64]),
        D_out=st.sampled_from([32, 64, 128]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        d_weight_optimize=st.booleans(),
        use_split_k=st.booleans(),
        use_split_k_tma=st.booleans(),
        contiguous=st.booleans(),
        has_bias=st.booleans(),
        allow_tf32=st.sampled_from([False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_index_select_jagged_bmm_triton_d_weight(self, *args, **kwargs) -> None:
        d_weight_optimize = kwargs.pop("d_weight_optimize")
        use_split_k = kwargs.pop("use_split_k")
        use_split_k_tma = kwargs.pop("use_split_k_tma")
        triton_option = IndexSelectJaggedBmmOption(
            d_weight_optimization=d_weight_optimize,
            d_weight_split_k_kernel=use_split_k,
            d_weight_split_k_kernel_tma=use_split_k_tma,
        )
        self._test_index_select_jagged_bmm(
            *args,
            **kwargs,
            test_backward=True,
            atol=None,
            rtol=None,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
            test_triton_option=triton_option,
        )

    # pyre-ignore[2]
    def test_index_select_jagged_bmm_big_shape_triton(self, *args, **kwargs) -> None:
        for gemm_out_type in [torch.float32, torch.bfloat16]:
            if (
                torch.cuda.get_device_capability(torch.device("cuda"))[0] < 8
                and gemm_out_type == torch.bfloat16
            ):
                continue
            for grouped_gemm in [True, False]:
                triton_option = IndexSelectJaggedBmmOption(
                    d_jagged_use_grouped_gemm=grouped_gemm,
                    d_jagged_gemm_out_type=gemm_out_type,
                )
                self._test_index_select_jagged_bmm(
                    # output: 4550700 * 256 * 2 = 2,329,958,400
                    L=4550700,
                    E=4,
                    K=2,
                    D_in=256,
                    D_out=256,
                    dtype=torch.bfloat16,
                    test_backward=True,
                    contiguous=False,
                    has_bias=True,
                    allow_tf32=False,
                    atol=1e-1,
                    rtol=1e-2,
                    ref_kernel=KernelType.PYTORCH,
                    real_kernel=KernelType.TRITON,
                    test_triton_option=triton_option,
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
        test_triton_option: Optional[IndexSelectJaggedBmmOption] = None,
    ) -> None:
        set_dev_mode(True)
        from fast_moe.kernels.moe import index_select_jagged_bmm

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
            triton_option=test_triton_option,
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

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        N=st.sampled_from([2, 32, 33, 100, 512, 1000]),
        # K must be power of 2 for now
        K=st.sampled_from([4, 16, 32]),
        D=st.sampled_from([128, 256]),
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
    def test_mul_merge_k_add_triton(self, *args, **kwargs) -> None:
        self._test_mul_merge_k_add(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
        )

    def _test_mul_merge_k_add(
        self,
        N: int,
        K: int,
        D: int,
        dtype: torch.dtype,
        ref_kernel: KernelType,
        real_kernel: KernelType,
        test_backward: bool,
        contiguous_weight: bool = False,
    ) -> None:
        set_dev_mode(True)
        from fast_moe.kernels.moe import mul_merge_k_add

        torch.manual_seed(0)

        x_base = (
            torch.empty((N * K, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        if contiguous_weight:
            w_st = 1
        else:
            w_st = 2
        w_base = (
            torch.empty((N, K * w_st), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        x_test = x_base.detach().clone().requires_grad_()
        w_test = w_base.detach().clone().requires_grad_()

        w_base = w_base[:, :K]
        w_test = w_test[:, :K]
        w_index = torch.randperm(w_base.numel(), device=torch.device("cuda"))

        perm = torch.randperm(N * K)
        o2i_index = torch.arange(N).repeat(K).view(-1)[perm].to("cuda")
        i2o_index = o2i_index.sort(stable=True)[1].view(-1, K)

        out_base: torch.Tensor = mul_merge_k_add(
            index=o2i_index,
            reverse_index=i2o_index,
            value=x_base,
            weight=w_base,
            weight_index=w_index,
            kernel=ref_kernel,
        )

        out_test: torch.Tensor = mul_merge_k_add(
            index=o2i_index,
            reverse_index=i2o_index,
            value=x_test,
            weight=w_test,
            weight_index=w_index,
            kernel=real_kernel,
        )

        torch.testing.assert_close(out_base, out_test)

        if test_backward:
            dout = torch.randn_like(out_base) * 0.01

            out_base.backward(dout)
            out_test.backward(dout)

            torch.testing.assert_close(x_base.grad, x_test.grad)
            torch.testing.assert_close(
                w_base.grad,
                w_test.grad,
                atol=1e-3 if dtype == torch.bfloat16 else None,
                rtol=1e-2 if dtype == torch.bfloat16 else None,
            )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        L=st.sampled_from([16, 32, 100, 500]),
        E=st.sampled_from([4, 8, 16]),
        K=st.sampled_from([2, 4]),
        D_in=st.just(8),
        D_out=st.just(4),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        contiguous=st.booleans(),
        allow_tf32=st.sampled_from([False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_index_select_jagged_bmm_3D_triton(self, *args, **kwargs) -> None:
        self._test_index_select_jagged_bmm_3D(
            *args,
            **kwargs,
            test_backward=True,
            atol=None,
            rtol=None,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
        )

    def _test_index_select_jagged_bmm_3D(
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
        allow_tf32: bool,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        triton_cc_version: str = "",
    ) -> None:
        set_dev_mode(True)

        from fast_moe.kernels.moe import index_select_jagged_bmm_3D

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
            torch.empty((E, L, D_in), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight_base = (
            torch.empty((E, D_in, D_out), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        bias_base = (
            torch.empty((E, D_out), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        jagged_test = jagged_base.detach().clone().requires_grad_()
        weight_test = weight_base.detach().clone().requires_grad_()
        bias_test = bias_base.detach().clone().requires_grad_()

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

        jagged_base = to_fp32_if_pytorch_kernel(jagged_base, ref_kernel).permute(
            1, 0, 2
        )
        out_base = index_select_jagged_bmm_3D(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged_base,
            weight=to_fp32_if_pytorch_kernel(weight_base, ref_kernel),
            bias=to_fp32_if_pytorch_kernel(bias_base, ref_kernel),
            kernel=ref_kernel,
        ).to(jagged_test.dtype)

        torch.cuda.manual_seed(0)

        jagged_test = jagged_test.permute(1, 0, 2)
        out_test = index_select_jagged_bmm_3D(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged_test,
            weight=weight_test,
            bias=bias_test,
            kernel=real_kernel,
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
                [jagged_base, weight_base, bias_base],
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
        D_out=st.sampled_from([64]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        contiguous=st.booleans(),
        has_bias=st.booleans(),
        allow_tf32=st.sampled_from([False]),
    )
    # pyre-ignore[2]
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_index_select_jagged_bmm_swiglu_triton(self, *args, **kwargs) -> None:
        self._test_index_select_jagged_bmm_swiglu(
            *args,
            **kwargs,
            test_backward=True,
            atol=None,
            rtol=None,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
        )

    def _test_index_select_jagged_bmm_swiglu(
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
        allow_tf32: bool,
        has_bias: bool,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        triton_cc_version: str = "",
    ) -> None:
        set_dev_mode(True)

        from fast_moe.kernels.moe import index_select_jagged_bmm_swiglu

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
        weight_p_base = (
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
            bias_p_base = (
                torch.empty((E, D_out), dtype=dtype, device=device)
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
            bias_test = bias_base.detach().clone().requires_grad_()
            bias_p_test = bias_p_base.detach().clone().requires_grad_()
        else:
            bias_base, bias_p_base, bias_test, bias_p_test = None, None, None, None

        jagged_test = jagged_base.detach().clone().requires_grad_()
        weight_test = weight_base.detach().clone().requires_grad_()
        weight_p_test = weight_p_base.detach().clone().requires_grad_()

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
            weight_p_base = (
                weight_p_base.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )
            weight_p_test = (
                weight_p_test.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )

        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        max_seq_len = int(lengths.max().item())

        out_base = index_select_jagged_bmm_swiglu(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=to_fp32_if_pytorch_kernel(jagged_base, ref_kernel),
            weight=to_fp32_if_pytorch_kernel(weight_base, ref_kernel),
            bias=to_fp32_if_pytorch_kernel(bias_base, ref_kernel)
            if bias_base is not None
            else None,
            weight_p=to_fp32_if_pytorch_kernel(weight_p_base, ref_kernel),
            bias_p=to_fp32_if_pytorch_kernel(bias_p_base, ref_kernel)
            if bias_p_base is not None
            else None,
            kernel=ref_kernel,
        ).to(jagged_test.dtype)

        torch.cuda.manual_seed(0)

        out_test = index_select_jagged_bmm_swiglu(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged_test,
            weight=weight_test,
            bias=bias_test,
            weight_p=weight_p_test,
            bias_p=bias_p_test,
            kernel=real_kernel,
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

            # Loosen numerical bar as we're comparing PT FP32 vs Triton BF16
            atol = 4e-3 if dtype == torch.bfloat16 else None
            rtol = 1e-2 if dtype == torch.bfloat16 else None
            for p_base, p_test in zip(
                [jagged_base, weight_base, weight_p_base, bias_base, bias_p_base],
                [jagged_test, weight_test, weight_p_test, bias_test, bias_p_test],
            ):
                if p_base is not None and p_test is not None:
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
        D_out=st.sampled_from([64]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        contiguous=st.booleans(),
        has_bias=st.booleans(),
        allow_tf32=st.sampled_from([False]),
    )
    # pyre-ignore[2]
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_index_select_jagged_gating_bmm_triton(self, *args, **kwargs) -> None:
        self._test_index_select_jagged_gating_bmm(
            *args,
            **kwargs,
            test_backward=True,
            atol=None,
            rtol=None,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
        )

    def _test_index_select_jagged_gating_bmm(
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
        allow_tf32: bool,
        has_bias: bool,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        triton_cc_version: str = "",
    ) -> None:
        set_dev_mode(True)

        from fast_moe.kernels.moe import index_select_jagged_gating_bmm

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

        jagged_a_base = (
            torch.empty((L, D_in), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        jagged_b_base = (
            torch.empty((L, D_in), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight_a_base = (
            torch.empty((E, D_in, D_out), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        weight_b_base = (
            torch.empty((E, D_in, D_out), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if has_bias:
            bias_a_base = (
                torch.empty((E, D_out), dtype=dtype, device=device)
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
            bias_b_base = (
                torch.empty((E, D_out), dtype=dtype, device=device)
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
            bias_a_test = bias_a_base.detach().clone().requires_grad_()
            bias_b_test = bias_b_base.detach().clone().requires_grad_()
        else:
            bias_a_base, bias_b_base, bias_a_test, bias_b_test = None, None, None, None

        jagged_a_test = jagged_a_base.detach().clone().requires_grad_()
        jagged_b_test = jagged_b_base.detach().clone().requires_grad_()
        weight_a_test = weight_a_base.detach().clone().requires_grad_()
        weight_b_test = weight_b_base.detach().clone().requires_grad_()

        if not contiguous:
            weight_a_base = (
                weight_a_base.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )
            weight_a_test = (
                weight_a_test.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )
            weight_b_base = (
                weight_b_base.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )
            weight_b_test = (
                weight_b_test.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )

        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        max_seq_len = int(lengths.max().item())

        out_base = index_select_jagged_gating_bmm(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged_a=to_fp32_if_pytorch_kernel(jagged_a_base, ref_kernel),
            jagged_b=to_fp32_if_pytorch_kernel(jagged_b_base, ref_kernel),
            weight_a=to_fp32_if_pytorch_kernel(weight_a_base, ref_kernel),
            weight_b=to_fp32_if_pytorch_kernel(weight_b_base, ref_kernel),
            bias_a=to_fp32_if_pytorch_kernel(bias_a_base, ref_kernel)
            if bias_a_base is not None
            else None,
            bias_b=to_fp32_if_pytorch_kernel(bias_b_base, ref_kernel)
            if bias_b_base is not None
            else None,
            kernel=ref_kernel,
        ).to(jagged_a_test.dtype)

        torch.cuda.manual_seed(0)

        out_test = index_select_jagged_gating_bmm(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged_a=jagged_a_test,
            jagged_b=jagged_b_test,
            weight_a=weight_a_test,
            weight_b=weight_b_test,
            bias_a=bias_a_test,
            bias_b=bias_b_test,
            kernel=real_kernel,
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

            # Loosen numerical bar as we're comparing PT FP32 vs Triton BF16
            atol = 4e-3 if dtype == torch.bfloat16 else None
            rtol = 1e-2 if dtype == torch.bfloat16 else None
            for p_base, p_test in zip(
                [
                    jagged_a_base,
                    jagged_b_base,
                    weight_a_base,
                    weight_b_base,
                    bias_a_base,
                    bias_b_base,
                ],
                [
                    jagged_a_test,
                    jagged_b_test,
                    weight_a_test,
                    weight_b_test,
                    bias_a_test,
                    bias_b_test,
                ],
            ):
                if p_base is not None and p_test is not None:
                    torch.testing.assert_close(
                        p_base.grad,
                        p_test.grad,
                        atol=atol,
                        rtol=rtol,
                    )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        max_seq_len=st.sampled_from([32, 64, 200]),
        min_seq_len=st.just(32),
        E=st.sampled_from([4, 8]),
        K=st.sampled_from([1, 2]),
        D_in=st.sampled_from([128, 256]),
        D_out=st.sampled_from([128, 256]),
        dtype=st.sampled_from([torch.bfloat16]),
        contiguous=st.just(True),
        allow_tf32=st.just(False),
        has_silu=st.booleans(),
        has_bias=st.booleans(),
        activation_checkpointing=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=30,
        deadline=None,
    )
    def test_jagged_bmm_combine(
        self,
        *args,  # pyre-ignore[2]
        **kwargs,  # pyre-ignore[2]
    ) -> None:
        self._test_jagged_bmm_combine(
            *args,
            **kwargs,
            atol=5e-1,
            rtol=1.5e-1,
            ref_kernel=KernelType.PYTORCH,
            real_kernel=KernelType.TRITON,
            test_backward=True,
        )

    def _test_jagged_bmm_combine(
        self,
        min_seq_len: int,
        max_seq_len: int,
        E: int,
        K: int,
        D_in: int,
        D_out: int,
        dtype: torch.dtype,
        ref_kernel: KernelType,
        real_kernel: KernelType,
        test_backward: bool,
        contiguous: bool,
        allow_tf32: bool,
        has_silu: bool,
        has_bias: bool,
        activation_checkpointing: bool = False,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        set_dev_mode(True)
        from fast_moe.kernels.moe import silu_jagged_bmm_combine

        torch.cuda.manual_seed(0)

        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        seq_len = torch.randint(
            low=min_seq_len, high=max_seq_len + 1, size=(E,), device=device
        )
        N = int(seq_len.sum().item())

        logits = torch.rand((N, E), device=device, dtype=dtype) + 1e-2
        topk_gates, topk_indices = logits.topk(k=K, dim=1)

        _, gates_index = topk_indices.view(-1).sort(stable=True)
        zeros = torch.zeros_like(logits, dtype=dtype)
        gates = zeros.scatter(1, topk_indices, topk_gates)
        index: torch.Tensor = gates_index // topk_indices.size(1)
        reverse_index: torch.Tensor = index.sort(stable=True)[1].view_as(topk_indices)
        lengths = (gates > 0).sum(0)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        max_seq_len = int(lengths.max().item())

        jagged_base = (
            torch.empty((N * K, D_in), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        weight_base = (
            torch.empty((E, D_in, D_out), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        bias_base = (
            (
                torch.empty((E, D_out), dtype=dtype, device=device)
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
            if has_bias
            else None
        )

        gates_base = topk_gates.clone().requires_grad_()

        if not contiguous:
            weight_base = (
                weight_base.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )

        jagged_test = jagged_base.detach().clone().requires_grad_()
        weight_test = weight_base.detach().clone().requires_grad_()
        # pyre-ignore
        bias_test = bias_base.detach().clone().requires_grad_() if has_bias else None
        gates_test = gates_base.detach().clone().requires_grad_()

        triton_option = SiluJaggedBmmCombineOption(
            activation_checkpointing=activation_checkpointing,
        )
        out_base = silu_jagged_bmm_combine(
            max_seq_len=max_seq_len,
            offsets=offsets,
            jagged=jagged_base,
            weight=weight_base,
            bias=bias_base,
            index=index,
            reverse_index=reverse_index,
            gates=gates_base,
            gates_index=gates_index,
            has_silu=has_silu,
            kernel=ref_kernel,
            triton_option=triton_option,
        )

        out_test = silu_jagged_bmm_combine(
            max_seq_len=max_seq_len,
            offsets=offsets.to(device),
            jagged=jagged_test.to(device),
            weight=weight_test.to(device),
            bias=None if bias_test is None else bias_test.to(device),
            index=index.to(device),
            reverse_index=reverse_index.to(device),
            gates=gates_test.to(device),
            gates_index=gates_index.to(device),
            has_silu=has_silu,
            kernel=real_kernel,
            triton_option=triton_option,
        )

        torch.testing.assert_close(
            out_base,
            out_test,
            atol=atol,
            rtol=rtol,
        )

        if test_backward:
            dout = torch.randn_like(out_base) * 0.01
            out_base.backward(dout)
            out_test.backward(dout.to(device))

            for p_base, p_test in zip(
                [jagged_base, weight_base, gates_base],
                [jagged_test, weight_test, gates_test],
            ):
                torch.testing.assert_close(
                    p_base.grad,
                    p_test.grad,
                    atol=atol,
                    rtol=rtol,
                )
            if has_bias and bias_base is not None and bias_test is not None:
                torch.testing.assert_close(
                    bias_base.grad,
                    bias_test.grad,
                    atol=atol,
                    rtol=rtol,
                )
