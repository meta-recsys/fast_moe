# pyre-strict

from typing import List

import click

import torch

# @manual=//triton:triton
import triton
from fast_moe.kernels.moe_fp8 import silu_jagged_bmm_fp8
from fast_moe.kernels.utils import KernelType

# pyre-fixme[21]: Could not find name `ProfilerActivity` in `torch.profiler`.
from torch.profiler import profile, ProfilerActivity


# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:silu_jagged_bmm_fp8_bench -- --fwd-only


def get_kernel(provider: str) -> KernelType:
    if provider == "triton":
        return KernelType.TRITON
    elif provider == "pytorch":
        return KernelType.PYTORCH
    else:
        raise ValueError(f"Unknown provider {provider}")


@click.command()
@click.option(
    "--max-seq-len",
    type=int,
    default=2048,
    show_default=True,
)
@click.option(
    "--e",
    type=int,
    default=32,
    show_default=True,
)
@click.option(
    "--m",
    type=int,
    default=768,
    show_default=True,
)
@click.option(
    "--n",
    type=int,
    default=256,
    show_default=True,
)
@click.option("--dtype", type=str, default="bf16")
@click.option("--fwd-only", is_flag=True)
@click.option("--enable-profile", is_flag=True, default=False)
def main(
    max_seq_len: int,
    e: int,
    m: int,
    n: int,
    dtype: str,
    fwd_only: bool,
    enable_profile: bool,
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
            x_names=["E"],
            x_vals=[1],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton_Unfused", "Pytorch"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"silu_jagged_bmm-{mode}-seq_len{max_seq_len}-M{m}-N{n}-{dtype}",
            args={
                "seq_len": max_seq_len,
                "M": m,
                "N": n,
                "dtype": pt_dtype,
                "mode": mode,
            },
        )
        for mode in (["fwd"] if fwd_only else ["fwd", "bwd"])
    ]

    @triton.testing.perf_report(configs)
    def bench_silu_jagged_bmm_fp8(
        seq_len: int,
        E: int,
        M: int,
        N: int,
        mode: str,
        provider: str,
        dtype: torch.dtype,
    ) -> float:
        assert mode in ["fwd", "bwd"]
        warmup = 1
        rep = 1

        max_seq_len = seq_len
        lengths = torch.randint(max_seq_len + 1, size=(E,))
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        lengths = lengths.to(torch.device("cuda"))
        offsets = offsets.to(torch.device("cuda"))
        max_seq_len = int(lengths.max().item())
        jagged_size = int(lengths.sum().item())
        jagged = (
            torch.empty((jagged_size, M), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        weight = (
            torch.empty((E, M, N), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        bias = (
            torch.empty((E, N), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if enable_profile:
            with profile(
                # pyre-fixme[16]: Module `torch.profiler` has no attribute `ProfilerActivity`.
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                silu_jagged_bmm_fp8(  # noqa E731
                    max_seq_len=max_seq_len,
                    seq_offsets=offsets,
                    jagged=jagged,
                    weight=weight,
                    bias=bias,
                    kernel=get_kernel(provider),
                )
                print(
                    prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
                )

        if provider in ["triton", "pytorch"]:
            fn = lambda: silu_jagged_bmm_fp8(  # noqa E731
                max_seq_len=max_seq_len,
                seq_offsets=offsets,
                jagged=jagged,
                weight=weight,
                bias=bias,
                kernel=get_kernel(provider),
            )
            if mode == "bwd":
                if provider == "triton_cc":
                    return -1
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

            return ms
        else:
            raise ValueError(f"unsupported provider: {provider}")

    bench_silu_jagged_bmm_fp8.run(print_data=True)


if __name__ == "__main__":
    main()
