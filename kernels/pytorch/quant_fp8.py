# pyre-strict

from typing import Tuple

import torch


def calculate_scale(
    a_max: torch.Tensor,
) -> torch.Tensor:
    MAX_FP8 = 448.0
    a_scale = torch.div(MAX_FP8, a_max).to(torch.float32)
    a_scale = torch.exp2(torch.floor(torch.log2(a_scale)))
    # Check for inf or nan
    is_nan = a_scale != a_scale
    is_inf = (a_scale == float("inf")) | (a_scale == float("-inf"))
    is_not_finite = is_nan | is_inf
    a_scale = torch.where(is_not_finite, 2.0**127, a_scale)
    return a_scale


def pytorch_rowwise_quant_fp8(
    a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = a.contiguous()
    a_max, _ = torch.max(torch.abs(a), dim=-1)
    a_scale = calculate_scale(a_max)
    a_fp8 = (a * a_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return a_fp8, a_scale


def pytorch_transpose_rowwise_quant_fp8(
    a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return pytorch_rowwise_quant_fp8(a.transpose(-2, -1))
