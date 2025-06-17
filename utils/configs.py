# pyre-strict
from dataclasses import dataclass


@dataclass
class SGConfig:
    model_d: int
    num_experts: int
    num_activated_experts: int
    loss_coef: float = 1e-2
    # control whether an MRN layer will use the input activation dtype from
    # HSTU as its dtype or not. I.e., if HSTU uses BF16 computation, MRN will
    # follow that throughout forward and backward if this is set to True.
    # Otherwise, MRN will let autocast automatically decide the proper dtypes.
    use_input_dtype: bool = True
    # control whether we apply activation checkpointing to mrn_compute_output and
    # silu_jagged_bmm_combine.
    activation_checkpointing: bool = False
    enable_noisy_gating: bool = False
