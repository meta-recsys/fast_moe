# pyre-strict
import argparse
import os
import random
import string

# import pickle
from datetime import datetime

# from typing import List

# import click

import torch

from fast_moe.kernels.moe_fp8 import silu_jagged_bmm_fp8
from fast_moe.kernels.utils import KernelType

# pyre-fixme[21]
from torch.profiler import profile, ProfilerActivity


def get_kernel(provider: str) -> KernelType:
    if provider == "triton":
        return KernelType.TRITON
    elif provider == "pytorch":
        return KernelType.PYTORCH
    else:
        raise ValueError(f"Unknown provider {provider}")


# pyre-ignore[3]
def manifold_trace_handler():
    """
    Outputs tracing files to directory of ``manifold://perfdoctor_gpu_traces_test/tree/traces/test/``.
    """

    # pyre-ignore[2]
    def handler_fn(prof) -> None:
        trace_filename = (
            "silu_jagged_bmm_fp8_pid"
            + str(os.getpid())
            + str(int(datetime.now().timestamp()))
            + "_random"
            + "".join(random.choices(string.digits, k=10))
            + ".json"
        )

        trace_filepath = f"{trace_filename}"
        prof.export_chrome_trace(
            "manifold://perfdoctor_gpu_traces_test/tree/traces/test/" + trace_filepath
        )
        trace_url = f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/test/{trace_filepath}.gz&bucket=perfdoctor_gpu_traces_test"
        print(trace_url)

    return handler_fn


def create_profiler() -> torch.profiler.profile:
    profiler = profile(
        schedule=torch.profiler.schedule(
            skip_first=50, wait=1, warmup=2, active=5, repeat=1
        ),
        # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=manifold_trace_handler(),
        record_shapes=True,
        profile_memory=False,  # profile_memory=True
        with_stack=False,  # with_stack=True
    )
    return profiler


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="silu_jagged_bmm_fp8 benchmark")
    parser.add_argument(
        "--provider",
        type=str,
        default="triton",
        help="triton, triton_cc, or pytorch",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fwd",
        help="fwd or bwd",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="max sequence length",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=256,
        help="min sequence length",
    )
    parser.add_argument(
        "--e",
        type=int,
        default=1,
        help="number of experts",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=256,
        help="hidden dimension",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=256,
        help="output dimension",
    )
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    # perf benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    max_seq_len = args.max_seq_len
    min_seq_len = args.min_seq_len
    E = args.e
    M = args.m
    N = args.n
    dtype = torch.bfloat16
    provider = args.provider

    lengths = torch.randint(min_seq_len, max_seq_len + 1, size=(E,))
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

    profiler = create_profiler()
    with profiler:
        for _ in range(100):
            dout = silu_jagged_bmm_fp8(  # noqa E731
                max_seq_len=max_seq_len,
                seq_offsets=offsets,
                jagged=jagged,
                weight=weight,
                bias=bias,
                kernel=get_kernel(provider),
            )

            if args.mode == "bwd":
                _d_out = torch.rand_like(dout) * 0.01
                dout.backward(_d_out, retain_graph=True)

            profiler.step()


if __name__ == "__main__":
    main()  # pragma: no cover
