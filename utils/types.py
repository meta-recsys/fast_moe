# pyre-strict
from typing import NamedTuple, Optional

import torch


class MRNOutput(NamedTuple):
    x: torch.Tensor
    loss: Optional[torch.Tensor]
    load: Optional[torch.Tensor]
