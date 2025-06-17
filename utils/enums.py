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
