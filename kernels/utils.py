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
