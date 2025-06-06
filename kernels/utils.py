# pyre-strict
from enum import Enum, unique
from typing import Tuple

import torch


@unique
class KernelType(Enum):
    TRITON = "TRITON"
    PYTORCH = "PYTORCH"
    TRITON_CC = "TRITON_CC"


gpu_unavailable: Tuple[bool, str] = (
    not torch.cuda.is_available()
    or torch.cuda.device_count() == 0
    or torch.version.hip,
    "CUDA is not available or no Nvidia GPUs detected",
)


def to_fp32_if_pytorch_kernel(
    x: torch.Tensor, kernel: KernelType = KernelType.PYTORCH
) -> torch.Tensor:
    if kernel == KernelType.PYTORCH:
        return x.to(torch.float32)
    return x


@torch.fx.wrap
def fx_torch_zeros_like(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros_like(t, dtype=dtype)


@torch.fx.wrap
def fx_infer_max_len(
    lengths: torch.Tensor,
) -> int:
    # Do not call ".item()" to avoid problems for lowering
    max_len = int(lengths.max())
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range [0, 10**9)
        torch._check_is_size(max_len)
        torch._check(max_len < 10**9)
        torch._check(max_len > 0)
    return max_len
