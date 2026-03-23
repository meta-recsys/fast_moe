# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import pickle
from typing import List

import click
import torch

# @manual=//triton:triton
import triton
from fast_moe.kernels.moe import index_select_jagged_bmm
from fast_moe.kernels.triton.triton_moe import _index_select_jagged_bmm
from fast_moe.kernels.utils import KernelType


# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:index_select_jagged_bmm_bench -- --fwd-only
# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:index_select_jagged_bmm_bench -- --dump-cache-dir=/home/${USER}/fbsource/fbcode/hammer/ops/triton/cc/index_select_jagged_bmm/autotune_cache.pkl


def get_kernel(provider: str) -> KernelType:
    if provider == "triton":
        return KernelType.TRITON
    elif provider == "pytorch":
        return KernelType.PYTORCH
    else:
        raise ValueError(f"Unknown provider {provider}")


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
@click.option("--dump-cache-dir", type=str, default="")
def main(
    num_tokens: int,
    e: int,
    k: int,
    m: int,
    n: int,
    dtype: str,
    fwd_only: bool,
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
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "Pytorch"],
            styles=[("red", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"index_select_jagged_bmm-{mode}-L{num_tokens}-E{e}-K{k}-M{m}-N{n}-{dtype}",
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
    def bench_index_select_jagged_bmm(
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

        jagged = (
            torch.empty((num_tokens, m), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight = (
            torch.empty((e, m, n), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        bias = (
            torch.empty((e, n), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if provider in ["triton", "pytorch"]:
            fn = lambda: index_select_jagged_bmm(  # noqa E731
                max_seq_len=max_seq_len,
                offsets=offsets,
                index=index,
                jagged=jagged,
                weight=weight,
                bias=bias,
                kernel=get_kernel(provider),
            )
        else:
            raise ValueError(f"unsupported provider: {provider}")

        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_index_select_jagged_bmm.run(print_data=True)

    if dump_cache_dir:
        with open(dump_cache_dir, "wb") as data:
            # @lint-ignore PYTHONPICKLEISBAD
            pickle.dump(_index_select_jagged_bmm.cache, data)


if __name__ == "__main__":
    main()
