# pyre-strict
import csv
import os
from typing import Any, Dict, Optional

import click
import torch
from fast_moe.kernels.benchmarks.configs import ProfilerParams
from fast_moe.kernels.benchmarks.utils import (
    BenchmarkResult,
    ModuleFactory,
    TrainModuleBench,
)
from fast_moe.kernels.pytorch.moe import pytorch_fused_jagged_bmm_swiglu_combine

# buck2 run @mode/{opt,inplace} fbcode//fast_moe/kernels/benchmarks:fused_index_select_swiglu_jagged_bmm_bench


class PytorchFusedJaggedBmmSwigluCombineModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        provider: str,
        offsets: torch.Tensor,
        jagged: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        index: torch.Tensor,
        weight_p: torch.Tensor,
        weight_out: torch.Tensor,
        reverse_index: torch.Tensor,
        gates: Optional[torch.Tensor] = None,
        gates_index: Optional[torch.Tensor] = None,
        bias_p: Optional[torch.Tensor] = None,
        bias_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, K = reverse_index.shape
        if provider in ["pytorch"]:
            # TODO: Currently only supporting pytorch
            return pytorch_fused_jagged_bmm_swiglu_combine(
                offsets=offsets,
                index=index,
                jagged=jagged,
                weight=weight,
                bias=bias,
                weight_p=weight_p,
                bias_p=bias_p,
                weight_out=weight_out,
                bias_out=bias_out,
                k=K,
                gates=gates,
                gates_index=gates_index,
            )
        else:
            raise ValueError(f"unsupported provider: {provider}")


class PytorchFusedJaggedBmmSwigluCombineModuleFactory(ModuleFactory):
    def __init__(
        self,
        provider: str,
        L: int = 8192,
        E: int = 128,
        K: int = 32,
        M: int = 512,
        N: int = 256,
        D: int = 128,
    ) -> None:
        """
        Args:
            provider: provider to use for the fused kernel
            L: number of tokens
            E: number of experts
            K: activated number of experts
            M: input dimension into fused module
            N: intermediate dimension of what used to be index_select_jagged_bmm -> silu_jagged_bmm_combine
            D: output dimension of fused module
        """

        self.provider = provider
        self.L = L
        self.E = E
        self.K = K
        self.M = M
        self.N = N
        self.D = D

    def module_name(self) -> str:
        return "pytorch_fused_jagged_bmm_swiglu_combine"

    def create_module(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        module = PytorchFusedJaggedBmmSwigluCombineModule().to(
            device=device, dtype=dtype
        )

        return module

    def create_inputs(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        module_inputs: Dict[str, Any] = {}
        L, E, K, M, N, D = self.L, self.E, self.K, self.M, self.N, self.D

        logits = torch.rand((L, E), device=device, dtype=dtype) + 1e-2
        topk_gates, topk_indices = logits.topk(k=K, dim=1)

        _, gates_index = topk_indices.view(-1).sort(stable=True)
        zeros = torch.zeros_like(logits, dtype=dtype)
        gates = zeros.scatter(1, topk_indices, topk_gates).to(device)
        index: torch.Tensor = gates_index // topk_indices.size(1)

        reverse_index: torch.Tensor = index.sort(stable=True)[1].view_as(topk_indices)
        lengths = (gates > 0).sum(0)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        jagged = (
            torch.empty((L * K, M), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        weight = (
            torch.empty((E, M, N), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        bias = (
            torch.empty((E, N), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight_p = (
            torch.empty((E, M, N), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        bias_p = (
            torch.empty((E, N), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight_out = (
            torch.empty((E, N, D), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        bias_out = (
            torch.empty((E, D), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        gates = topk_gates.clone().requires_grad_()

        module_inputs["provider"] = self.provider
        module_inputs["offsets"] = offsets
        module_inputs["jagged"] = jagged
        module_inputs["weight"] = weight
        module_inputs["bias"] = bias
        module_inputs["index"] = index
        module_inputs["gates"] = gates
        module_inputs["gates_index"] = gates_index
        module_inputs["weight_p"] = weight_p
        module_inputs["bias_p"] = bias_p
        module_inputs["weight_out"] = weight_out
        module_inputs["bias_out"] = bias_out
        module_inputs["reverse_index"] = reverse_index

        return module_inputs


@click.command()
@click.option(
    "--provider",
    type=str,
    default="pytorch",
    show_default=True,
)
@click.option(
    "--num-tokens",
    type=int,
    default=2048,
    show_default=True,
)
@click.option(
    "-e",
    type=int,
    default=64,
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
    default=512,
    show_default=True,
)
@click.option(
    "-d",
    type=int,
    default=256,
    show_default=True,
)
def main(
    provider: str, num_tokens: int, e: int, k: int, m: int, n: int, d: int
) -> None:
    factory = PytorchFusedJaggedBmmSwigluCombineModuleFactory(
        provider=provider, L=num_tokens, E=e, K=k, M=m, N=n, D=d
    )
    bench = TrainModuleBench(
        factory,
        precision="bf16",
        run_backward=True,
        profiler_params=ProfilerParams(
            wait_cycles=5,
            warmup_cycles=10,
            active_cycles=10,
        ),
    )

    params = {
        "provider": provider,
        "num_tokens": num_tokens,
        "e": e,
        "k": k,
        "m": m,
        "n": n,
        "d": d,
    }

    try:
        benchmark = bench.run_benchmark(return_result=True)
        if benchmark is not None:
            save_results_to_csv(
                params,
                benchmark,
            )
    except Exception as exception:
        print(f"Exception is {exception}")
        save_results_to_csv(
            params,
            None,
        )


def save_results_to_csv(
    params: Dict[str, int | str],
    benchmark: BenchmarkResult | None,
) -> None:
    file_path_name = "fused_index_select_swiglu_jagged_bmm.csv"
    file_exists = os.path.isfile(file_path_name)
    with open(file_path_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                list(params.keys())
                + [
                    "peak_memory_MB",
                    "cuda_time_average_sec",
                ]
            )
        writer.writerow(
            [params[key] for key in params.keys()]
            + [
                f"{benchmark.memory_result['allocated_bytes.all.peak'] / (1024**2):.2f}"
                if benchmark
                else "OOM",
                f"{benchmark.timer_result.mean_sec:.2f}" if benchmark else "OOM",
            ]
        )

    print(f"Results saved to {file_path_name}")


if __name__ == "__main__":
    main()
