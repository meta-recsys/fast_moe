#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


# N.B. stable sorting is slow, it should only be set to true for testing
STABLE_SORTING: bool = False


class MRNOutput(NamedTuple):
    x: torch.Tensor
    loss: Optional[torch.Tensor]
    load: Optional[torch.Tensor]


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


def _create_fused_mlp_weights(num_mlps: int, in_dim: int, out_dim: int) -> torch.Tensor:
    t = torch.empty(size=(num_mlps, in_dim, out_dim))
    for i in range(num_mlps):
        torch.nn.init.xavier_uniform_(t[i])
    return t


class OrigDispatcherImpl(object):
    def __init__(self, num_experts: int, gates: torch.Tensor) -> None:
        self._gates = gates
        self._num_experts = num_experts
        non_zero_gate = torch.nonzero(gates)
        sorted_experts, index_sorted_experts = non_zero_gate.sort(0)
        # pyre-ignore
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # pyre-ignore
        self._batch_index = non_zero_gate[index_sorted_experts[:, 1], 0]
        # pyre-ignore
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        # pyre-ignore
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, expert_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        expert_input = expert_inputs[self._batch_index].squeeze(1)
        return torch.split(expert_input, self._part_sizes, dim=0)

    def combine(
        self,
        expert_out: List[torch.Tensor],
        output_size: int,
        multiply_by_gates: bool = True,
    ) -> torch.Tensor:
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        B: int = self._gates.size(0)
        zeros = torch.zeros(
            [B, output_size],
            device=stitched.device,
            # force accumulate in FP32 to avoid underflow
            dtype=torch.float32,
            requires_grad=True,
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float()).to(
            stitched.dtype
        )
        return combined


class OrigSGMoEImpl(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: SGConfig,
        custom_kernel: bool = True,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts: int = config.num_experts
        self.k: int = config.num_activated_experts
        self.loss_coef: float = config.loss_coef
        self.model_d: int = config.model_d
        self.output_size: int = output_size
        self.input_size: int = input_size
        self.custom_kernel: bool = custom_kernel
        self.is_train: bool = is_train
        self.use_triton_cc: bool = not torch.version.hip
        self.use_input_dtype: bool = config.use_input_dtype
        self.enable_noisy_gating: bool = config.enable_noisy_gating

        self._experts_hidden_w = torch.nn.Parameter(
            data=_create_fused_mlp_weights(
                self.num_experts,
                self.model_d,
                input_size,
            )
        )
        self._experts_hidden_bias = torch.nn.Parameter(
            data=torch.empty(
                [self.num_experts, self.model_d],
            ).fill_(0.0)
        )

        self._experts_w = torch.nn.Parameter(
            data=_create_fused_mlp_weights(
                self.num_experts,
                output_size,
                self.model_d,
            )
        )
        self._experts_bias = torch.nn.Parameter(
            data=torch.empty(
                [self.num_experts, output_size],
            ).fill_(0.0)
        )

        self._w_gate: torch.nn.Parameter = torch.nn.Parameter(
            (
                torch.zeros(
                    input_size,
                    self.num_experts,
                )
                if self.enable_noisy_gating
                else torch.empty(
                    input_size,
                    self.num_experts,
                )
            ),
            requires_grad=True,
        )
        if self.enable_noisy_gating:
            self._w_noise: torch.nn.Parameter = torch.nn.Parameter(
                torch.zeros(
                    input_size,
                    self.num_experts,
                ),
                requires_grad=True,
            )
        else:
            torch.nn.init.xavier_uniform_(self._w_gate)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self._output_weight = torch.nn.Parameter(
            torch.empty(
                (
                    input_size,
                    output_size,
                )
            ),
        )
        torch.nn.init.xavier_uniform_(self._output_weight)
        self._k_plus: int = min(self.k + 1, self.num_experts)

        assert self.k <= self.num_experts

    def _cv_squared(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device).to(x.dtype)
        # pyre-ignore
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(
        self,
        clean_values: torch.Tensor,
        noisy_values: torch.Tensor,
        noise_stddev: torch.Tensor,
        noisy_top_values: torch.Tensor,
    ) -> torch.Tensor:
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )

        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )

        normal = Normal(self.mean, self.std, validate_args=False)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _orig_noisy_gating(
        self,
        x: torch.Tensor,
        train: bool,
        noise_eps: float = 1e-2,
        post_softmax: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_logits = x @ self._w_gate
        if train and self.enable_noisy_gating:
            raw_noise_stddev = x @ self._w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_eps
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = F.normalize(logits, dim=1)
        sorted_logits, sorted_indices = logits.sort(
            descending=True,
            dim=1,
            stable=True,  # stable=mrn.STABLE_SORTING
        )
        top_logits = sorted_logits[:, : self._k_plus]
        top_k_indices = sorted_indices[:, : self.k]

        if post_softmax and self.k > 1:
            top_k_logits = sorted_logits[:, : self.k]
            top_k_gates = F.softmax(top_k_logits, dim=1)
        else:
            top_gates = F.softmax(sorted_logits, dim=1)
            top_k_gates = top_gates[:, : self.k]

        top_k_gates = top_k_gates.to(x.dtype)

        zeros = torch.zeros_like(logits, requires_grad=True).to(top_k_gates.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.k < self.num_experts and train and self.enable_noisy_gating:
            load = (
                self._prob_in_top_k(
                    clean_logits,
                    noisy_logits,  # pyre-ignore
                    noise_stddev,  # pyre-ignore
                    top_logits,
                )
            ).sum(0)
        else:
            load = (gates > 0).sum(0)

        if not train:
            loss = torch.zeros([], device=top_k_gates.device, requires_grad=False)
            return gates, loss
        else:
            importance = gates.sum(0)
            loss = self._cv_squared(importance) + self._cv_squared(load)
            return gates, loss

    def router_forward(self, x: torch.Tensor, train: bool) -> MRNOutput:
        gates, loss = self._orig_noisy_gating(x, train)
        loss *= self.loss_coef
        dispatcher = OrigDispatcherImpl(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_output_hidden = [
            F.silu(
                F.linear(
                    expert_inputs[i],
                    self._experts_hidden_w[i],
                    self._experts_hidden_bias[i],
                )
            )
            for i in range(self.num_experts)
        ]
        expert_outputs = [
            F.linear(expert_output_hidden[i], self._experts_w[i], self._experts_bias[i])
            for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs, self.output_size)
        return MRNOutput(y, loss, (gates > 0).sum(0) / (gates > 0).sum())

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        group_norm: bool,
        num_groups: int,
        linear_dim: int,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
    ) -> MRNOutput:
        if group_norm:
            x_norm = F.group_norm(
                x.view(-1, num_groups, linear_dim),
                num_groups=num_groups,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
            ).view(-1, self.input_size)
        else:
            x_norm = F.layer_norm(
                x,
                normalized_shape=(x.shape[-1],),
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
            )

        # norm is on the disallowed list for AMP autocast, where the output
        # will be FP32 even if BF16 autocast is enabled. So, explicitly cast
        # back to input dtype here.
        x_norm = x_norm.to(x.dtype) if self.use_input_dtype else x_norm
        output = self.router_forward(x_norm, self.training)
        y = torch.addmm(output[0], x, self._output_weight)
        return MRNOutput(y, output[1], output[2])
