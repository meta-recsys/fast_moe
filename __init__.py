# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
FastMoE - High-Performance Mixture of Experts for PyTorch

FastMoE is a PyTorch library for efficient Mixture of Experts (MoE) layers,
featuring optimized GPU kernels implemented in both PyTorch and Triton.
"""

__version__ = "0.1.0"
__author__ = "Meta MRS"
__license__ = "MIT"

from fast_moe.kernels.utils import KernelType

# Import main classes for convenient access
from fast_moe.modules.fast_moe_module import FastMoELayer, GatingNetwork
from fast_moe.utils.enums import ExpertType, LossType, RouterChoice

__all__ = [
    "FastMoELayer",
    "GatingNetwork",
    "KernelType",
    "ExpertType",
    "RouterChoice",
    "LossType",
    "__version__",
]
