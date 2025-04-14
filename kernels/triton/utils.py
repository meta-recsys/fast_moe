# pyre-strict
#!/usr/bin/env python3

import logging
from typing import List

import torch

# @manual=//triton:triton
import triton

from fast_moe.utils import get_verbose_level, is_dev_mode

# @manual=//triton:triton
from triton.runtime.autotuner import Autotuner, Heuristics

logger: logging.Logger = logging.getLogger(__name__)

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import fast_dividef, fast_expf
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import fast_dividef, fast_expf
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import fast_dividef, fast_expf

if torch.version.hip:
    # @manual=//triton:triton
    from triton.language import exp
else:
    # pyre-ignore[16]: Undefined attribute
    exp = fast_expf


@triton.jit
def fast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    return fast_dividef(1.0, 1 + exp(-x))


def next_power_of_2(x: int) -> int:
    out = triton.next_power_of_2(x)
    return out


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def get_bmm_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_N in [64, 128]:
            for BLOCK_K in [32, 64]:
                for num_stages in [2, 3]:
                    for num_warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": BLOCK_M,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_K": BLOCK_K,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


def _get_rowwise_quant_fp8_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [16, 32, 64, 128]:
        for BLOCK_K in [16, 32, 64]:
            for num_stages in [2, 3]:
                for num_warps in [4, 8]:
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": BLOCK_M,
                                "BLOCK_K": BLOCK_K,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


class TritonAutotuner(Autotuner):
    # pyre-ignore[2]
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if is_dev_mode():
            fn = args[0]
            logger.warn(f"Only keep first config for {fn} due to dev mode")
            del self.configs[1:]
        self.bench_time = -1
        self.best_config: List[triton.Config] = []

    @property
    def kernel_name(self) -> str:
        return (
            f"{self.fn.fn.__name__}"
            if isinstance(self.fn, Heuristics)
            else f"{self.fn.__name__}"
        )

    # pyre-ignore[2]
    def prune_configs(self, kwargs) -> List[triton.Config]:
        if is_dev_mode():
            return self.configs[:1]
        else:
            return super().prune_configs(kwargs)

    # pyre-ignore[2, 3]
    def run(self, *args, **kwargs):
        orig_num_warmups = self.num_warmups
        orig_num_reps = self.num_reps
        orig_bench_time = self.bench_time
        if is_dev_mode():
            self.num_warmups = 1
            self.num_reps = 1

        ret = super().run(*args, **kwargs)

        if is_dev_mode():
            self.num_warmups = orig_num_warmups
            self.num_reps = orig_num_reps
        if get_verbose_level() > 0 and self.bench_time != orig_bench_time:
            logger.info(
                f"FastMOE TritonAutotuner: {self.kernel_name}: {self.best_config}: takes {self.bench_time} secs"
            )

        return ret


# pyre-ignore
def triton_autotune(
    configs: List[triton.Config],
    key: List[str],
    # pyre-ignore
    prune_configs_by=None,
    # pyre-ignore
    reset_to_zero=None,
    # pyre-ignore
    restore_value=None,
):
    # pyre-ignore
    def decorator(fn):
        return TritonAutotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            prune_configs_by=prune_configs_by,
        )

    return decorator
