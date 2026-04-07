# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import pickle
from typing import List, Optional

import click
import torch

# @manual=//triton:triton
import triton
from fast_moe.kernels.moe import index_select_jagged_bmm, index_select_jagged_gating_bmm
from fast_moe.kernels.triton.triton_moe import _index_select_jagged_gating_bmm
from fast_moe.kernels.utils import KernelType
from torch.nn import functional as F
from torch.profiler import profile

# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:index_select_jagged_gating_bmm_bench -- --fwd-only
# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:index_select_jagged_gating_bmm_bench -- --dump-cache-dir=/home/${USER}/fbsource/fbcode/hammer/ops/triton/cc/index_select_jagged_bmm/autotune_cache.pkl


def get_kernel(provider: str) -> KernelType:
    if provider == "triton":
        return KernelType.TRITON
    elif provider == "pytorch":
        return KernelType.PYTORCH
    else:
        raise ValueError(f"Unknown provider {provider}")


def triton_unfused_impl(
    max_seq_len: int,
    offsets: torch.Tensor,
    index: torch.Tensor,
    jagged_a: torch.Tensor,
    jagged_b: torch.Tensor,
    weight_a: torch.Tensor,
    bias_a: Optional[torch.Tensor],
    weight_b: torch.Tensor,
    bias_b: Optional[torch.Tensor],
) -> torch.Tensor:
    return F.silu(
        index_select_jagged_bmm(
            max_seq_len=max_seq_len,
            offsets=offsets,
            index=index,
            jagged=jagged_a,
            weight=weight_a,
            bias=bias_a,
            kernel=KernelType.TRITON,
        )
    ) * index_select_jagged_bmm(
        max_seq_len=max_seq_len,
        offsets=offsets,
        index=index,
        jagged=jagged_b,
        weight=weight_b,
        bias=bias_b,
        kernel=KernelType.TRITON,
    )


@click.command()
@click.option(
    "--num-tokens",
    type=int,
    default=8192,
    show_default=True,
)
@click.option(
    "-e",
    type=int,
    default=128,
    show_default=True,
)
@click.option(
    "-k",
    type=int,
    default=32,
    show_default=True,
)
@click.option(
    "-m",
    type=int,
    default=512,
    show_default=True,
)
@click.option(
    "-n",
    type=int,
    default=128,
    show_default=True,
)
@click.option("--dtype", type=str, default="bf16")
@click.option("--fwd-only", is_flag=True)
@click.option(
    "--enable-profile",
    is_flag=True,
    default=False,
)
@click.option("--dump-cache-dir", type=str, default="")
def main(
    num_tokens: int,
    e: int,
    k: int,
    m: int,
    n: int,
    dtype: str,
    fwd_only: bool,
    enable_profile: bool,
    dump_cache_dir: str,
) -> None:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if dtype == "fp32":
        pt_dtype = torch.float32
    elif dtype == "fp16":
        pt_dtype = torch.float16
    elif dtype == "bf16":
        pt_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported data type: {dtype}.")

    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=[1024, 2048, 4096, 8192, 16384, 32768],
            line_arg="provider",
            line_vals=["triton", "triton_unfused", "pytorch"],
            line_names=["Triton", "Triton_Unfused", "Pytorch"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"index_select_jagged_bmm_swiglu-{mode}-L{num_tokens}-E{e}-K{k}-M{m}-N{n}-{dtype}",
            args={
                "e": e,
                "k": k,
                "m": m,
                "n": n,
                "dtype": pt_dtype,
                "mode": mode,
            },
        )
        for mode in (["fwd"] if fwd_only else ["fwd", "bwd"])
    ]

    @triton.testing.perf_report(configs)
    def bench_index_select_jagged_gating_bmm(
        num_tokens: int,
        e: int,
        k: int,
        m: int,
        n: int,
        mode: str,
        provider: str,
        dtype: torch.dtype,
    ) -> float:
        assert mode in ["fwd", "bwd"]
        warmup = 25
        rep = 100

        torch.manual_seed(0)

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        gate = torch.randn(num_tokens, e, device=device).topk(k, dim=1).indices

        expert, index = gate.contiguous().view(-1).sort(stable=True)
        index = index.view(-1, k)

        zeros = torch.zeros(e, dtype=expert.dtype, device=device)
        lengths = zeros.scatter_add(0, expert, torch.ones_like(expert))
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        max_seq_len = int(lengths.max().item())

        jagged_a = (
            torch.empty((num_tokens, m), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        jagged_b = (
            torch.empty((num_tokens, m), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight_a = (
            torch.empty((e, m, n), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        bias_a = (
            torch.empty((e, n), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight_b = (
            torch.empty((e, m, n), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        bias_b = (
            torch.empty((e, n), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if provider in ["triton", "pytorch"]:
            fn = lambda: index_select_jagged_gating_bmm(  # noqa E731
                max_seq_len=max_seq_len,
                offsets=offsets,
                index=index,
                jagged_a=jagged_a,
                jagged_b=jagged_b,
                weight_a=weight_a,
                bias_a=bias_a,
                weight_b=weight_b,
                bias_b=bias_b,
                kernel=get_kernel(provider),
            )
        elif provider == "triton_unfused":
            fn = lambda: triton_unfused_impl(  # noqa E731
                max_seq_len=max_seq_len,
                offsets=offsets,
                index=index,
                jagged_a=jagged_a,
                jagged_b=jagged_b,
                weight_a=weight_a,
                bias_a=bias_a,
                weight_b=weight_b,
                bias_b=bias_b,
            )
        else:
            raise ValueError(f"unsupported provider: {provider}")

        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        if enable_profile:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,  # track tensor memory allocations
                with_stack=True,  # gather stack traces
            ) as prof:
                fn()
            # Print a textual summary
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            # Export to a Chrome Trace file for visual inspection:
            prof.export_chrome_trace(f"trace_{provider}_{num_tokens}.json")

        return ms

    bench_index_select_jagged_gating_bmm.run(print_data=True)

    if dump_cache_dir:
        with open(dump_cache_dir, "wb") as data:
            # @lint-ignore PYTHONPICKLEISBAD
            pickle.dump(_index_select_jagged_gating_bmm.cache, data)


if __name__ == "__main__":
    main()
