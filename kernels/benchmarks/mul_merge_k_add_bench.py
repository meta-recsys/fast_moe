# pyre-strict
import pickle
from typing import List, Optional, Tuple

import click
import pandas as pd
import torch

# @manual=//triton:triton
import triton
from fast_moe.kernels.moe import mul_merge_k_add
from fast_moe.kernels.triton.triton_moe import _mul_merge_k_add_fwd

from fast_moe.kernels.utils import KernelType


def get_kernel(provider: str) -> KernelType:
    if provider == "triton":
        return KernelType.TRITON
    elif provider == "pytorch":
        return KernelType.PYTORCH
    else:
        raise ValueError(f"Unknown provider {provider}")


# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:mul_merge_k_add_bench
# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:mul_merge_k_add_bench -- --dump-cache-dir=/home/${USER}/fbsource/fbcode/hammer/ops/triton/cc/mul_merge_k_add/autotune_cache.pkl


@click.command()
@click.option("--activated", type=int, default=32)
@click.option("--embedding-dim", type=int, default=256)
@click.option("--mode", type=str, default="train")
@click.option("--return-result", type=bool, default=False)
@click.option("--dump-cache-dir", type=str, default="")
def main(
    activated: int,
    embedding_dim: int,
    mode: str,
    return_result: bool,
    dump_cache_dir: str,
) -> Optional[Tuple[List[triton.testing.Benchmark], List[pd.DataFrame]]]:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if mode == "train":
        Ns = [1024, 2048, 4096, 8192, 16384]
        line_vals = ["triton", "pytorch"]
        line_names = ["Triton", "PyTorch"]
        styles = [("red", "-"), ("blue", "-")]
        test_pass = ["fwd", "bwd"]
    else:
        Ns = [3000, 5000, 8192, 16384, 30000]
        line_vals = ["triton", "pytorch"]
        line_names = ["Triton", "PyTorch"]
        styles = [("red", "-"), ("blue", "-")]
        test_pass = ["fwd"]

    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=Ns,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            styles=styles,
            ylabel="ms",
            plot_name=f"mul_merge_k_add-K{activated}-D{embedding_dim}-{run_pass}",
            args={
                "K": activated,
                "D": embedding_dim,
                "run_pass": run_pass,
            },
        )
        for run_pass in test_pass
    ]

    @triton.testing.perf_report(configs)
    def bench_mul_merge_k_add(
        N: int,
        K: int,
        D: int,
        provider: str,
        run_pass: str,
    ) -> float:
        assert run_pass in ["fwd", "bwd"]
        warmup = 25
        rep = 200
        torch.manual_seed(1001)  # for reproducibility

        dtype = torch.bfloat16
        x = (
            torch.empty((N * K, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        weight = (
            torch.empty((N * K, 1), dtype=dtype, device=torch.device("cuda"))
            .uniform_(0.1, 1.0)
            .requires_grad_()
        )
        weight_index = torch.randperm(weight.numel(), device=torch.device("cuda"))

        perm = torch.randperm(N * K)
        index = torch.arange(N).repeat(K).view(-1)[perm].to("cuda")
        reverse_index = index.sort(stable=True)[1].view(-1, K)

        fn = lambda: mul_merge_k_add(  # noqa E731
            index=index,
            reverse_index=reverse_index,
            value=x,
            weight=weight,
            weight_index=weight_index,
            kernel=get_kernel(provider),
        )
        if run_pass == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)  # noqa E731

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    df = bench_mul_merge_k_add.run(print_data=True, return_df=return_result)

    if dump_cache_dir:
        with open(dump_cache_dir, "wb") as data:
            # @lint-ignore PYTHONPICKLEISBAD
            pickle.dump(_mul_merge_k_add_fwd.cache, data)

    if return_result:
        return configs, df


if __name__ == "__main__":
    main()
