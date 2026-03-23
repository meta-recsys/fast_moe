# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""FastMoE Modules Package"""

from fast_moe.modules.fast_moe_module import FastMoELayer, GatingNetwork

__all__ = ["FastMoELayer", "GatingNetwork"]
