# pyre-strict
from typing import Any, Dict, Optional

import click
import torch
from fast_moe.kernels.benchmarks.utils import (
    get_kernel,
    ModuleFactory,
    TrainModuleBench,
)
from fast_moe.kernels.moe import index_select_jagged_bmm
from fast_moe.kernels.moe_fp8 import (
    index_select_jagged_bmm as index_select_jagged_bmm_fp8,
)
from fast_moe.kernels.triton.triton_moe import IndexSelectJaggedBmmOption
from fast_moe.kernels.utils import KernelType

# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:index_select_jagged_bmm_bench1 -- --provider triton_fp8 --num-tokens 822171 -e 32 -k 4 -m 512 -n 256


class IndexSelectJaggedBMMModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        provider: str,
        max_seq_len: int,
        offsets: torch.Tensor,
        index: torch.Tensor,
        jagged: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        if provider in ["triton", "triton_split_k", "pytorch"]:
            if provider == "triton_split_k":
                option = IndexSelectJaggedBmmOption(d_weight_split_k_kernel=True)
            else:
                option = IndexSelectJaggedBmmOption()
            return index_select_jagged_bmm(  # noqa E731
                max_seq_len=max_seq_len,
                offsets=offsets,
                index=index,
                jagged=jagged,
                weight=weight,
                bias=bias,
                kernel=get_kernel(provider),
                triton_option=option,
            )
        elif provider == "triton_fp8":
            return index_select_jagged_bmm_fp8(  # noqa E731
                max_seq_len=max_seq_len,
                offsets=offsets,
                index=index,
                jagged=jagged,
                weight=weight,
                bias=bias,
                kernel=KernelType.TRITON,
                fp8=True,
            )
        else:
            raise ValueError(f"unsupported provider: {provider}")


class IndexSelectJaggedBMMModuleFactory(ModuleFactory):
    def __init__(
        self,
        provider: str,
        L: int = 8192,
        E: int = 128,
        K: int = 32,
        M: int = 512,
        N: int = 128,
        add_randomness: bool = False,
    ) -> None:
        self.provider = provider
        self.L = L
        self.E = E
        self.K = K
        self.M = M
        self.N = N
        self.add_randomness = add_randomness

    def module_name(self) -> str:
        return "index_select_jagged_bmm"

    def create_module(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        module = IndexSelectJaggedBMMModule().to(device=device, dtype=dtype)

        return module

    def create_inputs(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        module_inputs: Dict[str, Any] = {}

        torch.manual_seed(42)
        if self.add_randomness:
            gate_random_rate = torch.randn(self.E, device=device).view(1, -1)
        else:
            gate_random_rate = torch.zeros(self.E, device=device).view(1, -1)
        gate = (
            (torch.randn(self.L, self.E, device=device) + gate_random_rate)
            .topk(self.K, dim=1)
            .indices
        )

        _, counts = torch.unique(gate, return_counts=True)
        print("Tokens Per Expert:", counts)

        expert, index = gate.contiguous().view(-1).sort(stable=True)
        index = index.view(-1, self.K)

        zeros = torch.zeros(self.E, dtype=expert.dtype, device=device)
        lengths = zeros.scatter_add(0, expert, torch.ones_like(expert))
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        max_seq_len = int(lengths.max().item())

        jagged = (
            torch.empty((self.L, self.M), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        weight = (
            torch.empty((self.E, self.M, self.N), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        bias = (
            torch.empty((self.E, self.N), dtype=dtype, device=device)
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        module_inputs["provider"] = self.provider
        module_inputs["max_seq_len"] = max_seq_len
        module_inputs["offsets"] = offsets
        module_inputs["index"] = index
        module_inputs["jagged"] = jagged
        module_inputs["weight"] = weight
        module_inputs["bias"] = bias

        return module_inputs


@click.command()
@click.option(
    "--provider",
    type=str,
    default="triton",
    show_default=True,
)
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
@click.option(
    "--add-randomness",
    is_flag=True,
)
def main(
    provider: str, num_tokens: int, e: int, k: int, m: int, n: int, add_randomness: bool
) -> None:
    factory = IndexSelectJaggedBMMModuleFactory(
        provider=provider,
        L=num_tokens,
        E=e,
        K=k,
        M=m,
        N=n,
        add_randomness=add_randomness,
    )
    bench = TrainModuleBench(factory, precision="bf16", run_backward=True)
    bench.run_benchmark()


if __name__ == "__main__":
    main()
