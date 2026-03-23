# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from enum import Enum, unique


@unique
class ExpertType(Enum):
    MLP = "MLP"
    FFN = "FFN"
    FFNBIAS = "FFN_WITH_BIAS"


@unique
class RouterChoice(Enum):
    TopK = "TOP_K"
    Vanilla = "VANILLA"


@unique
class LossType(Enum):
    LB = "LOAD_BALANCE"
    MI = "MUTAL_INFORMATION"
