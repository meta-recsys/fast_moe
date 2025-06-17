# pyre-strict
from typing import List, Tuple

import torch
import torch.nn.functional as F
from fast_moe.kernels.moe import mul_merge_k_add
from fast_moe.kernels.utils import KernelType
from torch.autograd.profiler import record_function
from torch.distributions import Normal


@torch.fx.wrap
def fx_torch_zeros(
    shape: List[int], device: torch.device, requires_grad: bool
) -> torch.Tensor:
    return torch.zeros(shape, device=device, requires_grad=requires_grad)


def _cv_squared(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)

    x_float = x.float()
    return x_float.var() / (x_float.mean().pow(2) + eps)


def _train_loss(
    gates: torch.Tensor, load: torch.Tensor, loss_coef: float
) -> torch.Tensor:
    return (_cv_squared(gates.sum(0)) + _cv_squared(load.sum(0))) * loss_coef


def _create_fused_mlp_weights(num_mlps: int, in_dim: int, out_dim: int) -> torch.Tensor:
    t = torch.empty(size=(num_mlps, in_dim, out_dim))
    for i in range(num_mlps):
        torch.nn.init.xavier_uniform_(t[i])
    return t


def _combine(
    expert_out: torch.Tensor,
    o2i_token_index: torch.Tensor,
    i2o_token_index: torch.Tensor,
    top_k_gates: torch.Tensor,
    gate_index: torch.Tensor,
    multiply_by_gates: bool = True,
    kernel: KernelType = KernelType.PYTORCH,
    stable_sorting: bool = False,
) -> torch.Tensor:
    with record_function("## combine ##"):
        return mul_merge_k_add(
            index=o2i_token_index,
            reverse_index=i2o_token_index,
            value=expert_out,
            stable_sorting=stable_sorting,
            weight=top_k_gates if multiply_by_gates else None,
            weight_index=gate_index if multiply_by_gates else None,
            kernel=kernel,
        )


def _prob_in_top_k(
    clean_values: torch.Tensor,
    noisy_values: torch.Tensor,
    noise_stddev: torch.Tensor,
    noisy_top_values: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    k: int,
) -> torch.Tensor:
    batch = clean_values.size(0)
    m = noisy_top_values.size(1)
    top_values_flat = noisy_top_values.flatten()

    threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + k
    threshold_if_in = torch.unsqueeze(
        torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
    )

    is_in = torch.gt(noisy_values, threshold_if_in)
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = torch.unsqueeze(
        torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
    )

    # Normal (super class Distribution) ctor forces D2H sync at
    # for every argument constrant https://fburl.com/gpfcsmc5  and input value
    # support https://fburl.com/9mnold55. The check enforces self.mean is a real
    # number and self.std is positive. The value support enforces the input is a
    # real number. We know these are always true for this model, so we skip those
    # checks to avoid blocking D2H syncs.
    normal = Normal(mean, std, validate_args=False)
    prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
    prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
    prob = torch.where(is_in, prob_if_in, prob_if_out)
    return prob


# We have to put these functions in here otherwise the fx wrap won't work
def _noisy_logits(
    x: torch.Tensor,
    clean_logits: torch.Tensor,
    weight_noise: torch.Tensor,
    noise: torch.Tensor,
    noise_eps: float = 1e-2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_noise_stddev = x @ weight_noise
    noise_stddev = F.softplus(raw_noise_stddev) + noise_eps
    noisy_logits = clean_logits + (noise * noise_stddev)
    return noisy_logits, noise_stddev, noise_stddev


def _dispatch(
    load: torch.Tensor,
    top_k_indices: torch.Tensor,
    stable_sorting: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # L: top_k_indices.shape[0]
    # K: top_k_indices.shape[1]
    # The dispatched result Y is a [L*K, D] tensor where L is the number of tokens in
    # the batch, K is the number of active experts, and D is the input dimension.
    #
    # expert_index [L*K]: for each row in Y, which expert it should be routed to
    # gate_index [L*K]: for each row in Y, which value in top_k_gates.view(-1) it corresponds to
    _, gate_index = top_k_indices.view(-1).sort(stable=stable_sorting)
    # shape [L*K]: for each row in Y, which token it is from
    token_index: torch.Tensor = gate_index // top_k_indices.size(1)
    # shape [E]
    num_tokens_per_expert: torch.Tensor = load.sum(0)

    return token_index, num_tokens_per_expert, gate_index


def _compute_top_logits(
    logits: torch.Tensor,
    k: int,
    k_plus: int,
    dtype: torch.dtype,
    post_softmax: bool = False,
    stable_sorting: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = F.normalize(logits, dim=1)
    sorted_logits, sorted_indices = logits.sort(
        descending=True, dim=1, stable=stable_sorting
    )
    top_logits = sorted_logits[:, :k_plus]
    top_k_indices = sorted_indices[:, :k].contiguous()
    if post_softmax and k > 1:
        top_k_logits = sorted_logits[:, :k]
        top_k_gates = F.softmax(top_k_logits, dim=1)
    else:
        top_gates = F.softmax(sorted_logits, dim=1)
        top_k_gates = top_gates[:, :k]

    # softplus and softmax forces FP32 outputs even under BF16 autocast context.
    # Converting back to input dtype so that subsequent computations will run in
    # desired precision.
    top_k_gates = top_k_gates.to(dtype).contiguous()
    return top_logits, top_k_gates, top_k_indices
