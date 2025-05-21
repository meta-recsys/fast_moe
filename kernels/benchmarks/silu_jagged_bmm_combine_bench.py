# pyre-strict
from typing import Any, Dict, Optional

import click
import torch
from fast_moe.kernels.benchmarks.utils import (
    get_kernel,
    ModuleFactory,
    TrainModuleBench,
)
from fast_moe.kernels.moe import silu_jagged_bmm_combine
from fast_moe.kernels.triton.triton_moe import SiluJaggedBmmCombineOption

# buck2 run @mode/{opt,inplace} //fast_moe/kernels/benchmarks:silu_jagged_bmm_combine_bench -- --provider triton --num-tokens 822171 -e 32 -k 4 -m 512 -n 256


class SiluJaggedBmmCombineModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        provider: str,
        max_seq_len: int,
        offsets: torch.Tensor,
        jagged: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        index: torch.Tensor,
        reverse_index: torch.Tensor,
        gates: Optional[torch.Tensor] = None,
        gates_index: Optional[torch.Tensor] = None,
        activation_checkpointing: bool = False,
        has_silu: bool = True,
    ) -> torch.Tensor:
        if provider in ["triton", "pytorch"]:
            return silu_jagged_bmm_combine(  # noqa E731
                max_seq_len=max_seq_len,
                offsets=offsets,
                jagged=jagged,
                weight=weight,
                bias=bias,
                index=index,
                reverse_index=reverse_index,
                gates=gates,
                gates_index=gates_index,
                has_silu=has_silu,
                kernel=get_kernel(provider),
                triton_option=SiluJaggedBmmCombineOption(
                    activation_checkpointing=activation_checkpointing
                ),
            )
        else:
            raise ValueError(f"unsupported provider: {provider}")


class SiluJaggedBmmCombineModuleFactory(ModuleFactory):
    def __init__(
        self,
        provider: str,
        L: int = 8192,
        E: int = 128,
        K: int = 32,
        M: int = 512,
        N: int = 128,
        activation_checkpointing: bool = False,
        has_silu: bool = True,
    ) -> None:
        self.provider = provider
        self.L = L
        self.E = E
        self.K = K
        self.M = M
        self.N = N
        self.activation_checkpointing = activation_checkpointing
        self.has_silu = has_silu

    def module_name(self) -> str:
        return "silu_jagged_bmm_combine"

    def create_module(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        module = SiluJaggedBmmCombineModule().to(device=device, dtype=dtype)

        return module

    def create_inputs(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        module_inputs: Dict[str, Any] = {}
        L, E, K, M, N = self.L, self.E, self.K, self.M, self.N

        logits = torch.rand((L, E), device=device, dtype=dtype) + 1e-2
        topk_gates, topk_indices = logits.topk(k=K, dim=1)

        _, gates_index = topk_indices.view(-1).sort(stable=True)
        zeros = torch.zeros_like(logits, dtype=dtype)
        gates = zeros.scatter(1, topk_indices, topk_gates).to(device)
        index: torch.Tensor = gates_index // topk_indices.size(1)
        reverse_index: torch.Tensor = index.sort(stable=True)[1].view_as(topk_indices)
        lengths = (gates > 0).sum(0)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        max_seq_len = int(lengths.max().item())

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
        gates = topk_gates.clone().requires_grad_()

        module_inputs["provider"] = self.provider
        module_inputs["max_seq_len"] = max_seq_len
        module_inputs["offsets"] = offsets
        module_inputs["jagged"] = jagged
        module_inputs["weight"] = weight
        module_inputs["bias"] = bias
        module_inputs["index"] = index
        module_inputs["reverse_index"] = reverse_index
        module_inputs["gates"] = gates
        module_inputs["gates_index"] = gates_index
        module_inputs["activation_checkpointing"] = self.activation_checkpointing
        module_inputs["has_silu"] = self.has_silu

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
    default=2048,
    show_default=True,
)
@click.option(
    "-e",
    type=int,
    default=32,
    show_default=True,
)
@click.option(
    "-k",
    type=int,
    default=2,
    show_default=True,
)
@click.option(
    "-m",
    type=int,
    default=768,
    show_default=True,
)
@click.option(
    "-n",
    type=int,
    default=256,
    show_default=True,
)
def main(provider: str, num_tokens: int, e: int, k: int, m: int, n: int) -> None:
    factory = SiluJaggedBmmCombineModuleFactory(
        provider=provider,
        L=num_tokens,
        E=e,
        K=k,
        M=m,
        N=n,
        activation_checkpointing=True,
        has_silu=True,
    )
    bench = TrainModuleBench(factory, precision="bf16", run_backward=True)
    bench.run_benchmark()


if __name__ == "__main__":
    main()
