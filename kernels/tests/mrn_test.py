# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# buck2 test -c fbcode.disable_re_tests=True @//mode/opt //fast_moe/kernels/tests:mrn_test -- --print-passing-details

# pyre-unsafe
import logging
import unittest
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fast_moe.dev_settings import set_dev_mode
from fast_moe.kernels.moe import index_select_jagged_bmm, silu_jagged_bmm_combine
from fast_moe.kernels.utils import (
    fx_infer_max_len,
    fx_torch_zeros_like,
    gpu_unavailable,
    KernelType,
)
from fast_moe.utils.configs import MRNOutput, SGConfig
from fast_moe.utils.enums import ExpertType, LossType, RouterChoice
from fast_moe.utils.utils import (
    _compute_top_logits,
    _create_fused_mlp_weights,
    _dispatch,
    _noisy_logits,
    _prob_in_top_k,
    _train_loss,
    fx_torch_zeros,
)
from hypothesis import given, settings, strategies as st, Verbosity
from pyre_extensions import none_throws
from torch.autograd.profiler import record_function
from torch.distributions.normal import Normal

logger: logging.Logger = logging.getLogger(__name__)

# N.B. stable sorting is slow, it should only be set to true for testing
STABLE_SORTING: bool = True

torch.fx.wrap("fx_infer_max_len")
torch.fx.wrap("fx_torch_zeros_like")


class SGMoE(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        group_norm: bool,
        num_groups: int,
        config: SGConfig,
        custom_kernel: bool = True,
        is_train: bool = True,
        test_mode: bool = False,
    ) -> None:
        """
        Args:
            test_mode: Currently for turning on stable sorting which is slow otherwise
        """
        super().__init__()
        self._is_inference: bool = not is_train
        self._stable_sorting = True if test_mode else False

        self._custom_kernel = custom_kernel

        self.num_experts: int = config.num_experts
        # TODO: use actual shared expert
        self.s: int = 1
        self.k: int = config.num_activated_experts

        self.loss_coef: float = 1e-2
        self.model_d: int = config.model_d
        self.output_hidden_dim: int = self.model_d * self.s
        self.output_size: int = output_size
        self.input_size: int = input_size
        self.custom_kernel: bool = custom_kernel
        self.is_train: bool = is_train
        self.use_input_dtype: bool = config.use_input_dtype
        self.activation_checkpointing: bool = False
        self.norm_eps: float = 1e-6
        # TODO: support expert_type
        self._expert_type: ExpertType = ExpertType.MLP
        # TODO: support router_choice
        self.router_choice: RouterChoice = RouterChoice.Vanilla

        post_norm = False
        self._post_norm: bool = post_norm
        self._loss_type: LossType = LossType.LB
        assert self.k <= self.num_experts, (
            f"Expect k <= num_experts, but got {self.k} > {self.num_experts}"
        )
        assert self.num_experts > 0 or self.s > 0

        self._norm_w: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((self.input_size,)),
        )
        self._norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((self.input_size,)),
        )

        if self._post_norm:
            self._output_norm_w: torch.nn.Parameter = torch.nn.Parameter(
                torch.ones((self.output_size,)),
            )
            self._output_norm_bias: torch.nn.Parameter = torch.nn.Parameter(
                torch.zeros((self.output_size,)),
            )

        self._dispatch: Callable[
            [torch.Tensor, torch.Tensor, bool],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = torch.compile(_dispatch) if self.is_train else _dispatch

        self.enable_noisy_gating: bool = config.enable_noisy_gating

        self.group_norm: bool = group_norm
        logger.info(
            f"SGMoE ({self.k}/{self.num_experts}), dims={self.input_size}-{self.model_d}-{self.output_size}"
        )

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

        self._prob_in_top_k: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ] = torch.compile(_prob_in_top_k) if is_train else _prob_in_top_k

        self._compute_top_logits: Callable[
            [torch.Tensor, int, int, torch.dtype, bool, bool],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = torch.compile(_compute_top_logits) if is_train else _compute_top_logits

        self._train_loss: Callable[
            [torch.Tensor, torch.Tensor, float],
            torch.Tensor,
        ] = torch.compile(_train_loss) if is_train else _train_loss

        self._noisy_logits: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = torch.compile(_noisy_logits) if is_train else _noisy_logits

    def _kernel_type(self) -> KernelType:
        if self._custom_kernel:
            return KernelType.TRITON
        else:
            return KernelType.PYTORCH

    def _noisy_gating(
        self,
        x: torch.Tensor,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_logits = x @ self._w_gate
        if train and self.enable_noisy_gating:
            noisy_logits, noise_stddev, noise_stddev = self._noisy_logits(
                x,
                clean_logits,
                self._w_noise,
                torch.randn_like(clean_logits),
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_k_gates, top_k_indices = self._compute_top_logits(
            logits,
            self.k,
            self._k_plus,
            x.dtype if self.use_input_dtype else logits.dtype,
            False,
            self._stable_sorting,
        )

        zeros = fx_torch_zeros_like(logits, dtype=top_k_gates.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.k < self.num_experts and train and self.enable_noisy_gating:
            load = self._prob_in_top_k(
                clean_logits,
                noisy_logits,  # pyre-ignore
                noise_stddev,  # pyre-ignore
                top_logits,
                # pyrefly: ignore [bad-argument-type]
                self.mean,
                # pyrefly: ignore [bad-argument-type]
                self.std,
                self.k,
            )
        else:
            load = gates > 0

        if not train:
            loss = fx_torch_zeros([], device=top_k_gates.device, requires_grad=False)
            return load, top_k_gates, top_k_indices, loss
        else:
            return (
                load,
                top_k_gates,
                top_k_indices,
                self._train_loss(gates, load, self.loss_coef),
            )

    def _gating(
        self,
        x: torch.Tensor,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._noisy_gating(x, train)

    def _expert_forward(
        self,
        x: torch.Tensor,
        token_index: torch.Tensor,
        lengths: torch.Tensor,
        gate_index: torch.Tensor,
        gating_scores: torch.Tensor,
    ) -> torch.Tensor:
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        reverse_token_index: torch.Tensor = token_index.sort(
            stable=self._stable_sorting
        )[1].view_as(gating_scores)
        max_seq_len = fx_infer_max_len(lengths)
        with record_function("## expert_forward##"):
            expert_hidden = index_select_jagged_bmm(
                max_seq_len=max_seq_len,
                offsets=offsets,
                index=gate_index.view(-1, self.k),
                jagged=x,
                weight=self._experts_hidden_w.permute(0, 2, 1).to(x.dtype),
                bias=self._experts_hidden_bias.to(x.dtype),
                kernel=self._kernel_type(),
            )

            y = silu_jagged_bmm_combine(
                max_seq_len=max_seq_len,
                offsets=offsets,
                jagged=expert_hidden,
                weight=self._experts_w.permute(0, 2, 1).to(x.dtype),
                bias=self._experts_bias.to(x.dtype),
                index=token_index,
                reverse_index=reverse_token_index,
                gating_scores=gating_scores,
                gates_index=gate_index,
                kernel=self._kernel_type(),
            )
        return y

    def _shared_expert_forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.addmm(y, x, self._output_weight)

    def router_forward(
        self, x: torch.Tensor, train: bool, shared_x: Optional[torch.Tensor] = None
    ) -> MRNOutput:
        if self.num_experts > 0:
            load, top_k_gates, top_k_indices, loss = self._gating(x, train=train)
            with record_function("## dispatch ##"):
                token_index, lengths, gate_index = self._dispatch(
                    load,
                    top_k_indices,
                    self._stable_sorting,
                )
                load = lengths / lengths.sum()
            y = self._expert_forward(x, token_index, lengths, gate_index, top_k_gates)
        else:
            y = x
            loss, load = None, None

        y = self._shared_expert_forward(shared_x if shared_x is not None else x, y)

        return MRNOutput(y, loss, load)

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
    ) -> MRNOutput:
        x_norm = F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=self._norm_w.to(x.dtype),
            bias=self._norm_bias.to(x.dtype),
            eps=self.norm_eps,
        )

        # norm is on the disallowed list for AMP autocast, where the output
        # will be FP32 even if BF16 autocast is enabled. So, explicitly cast
        # back to input dtype here.
        x_norm = x_norm.to(x.dtype) if self.use_input_dtype else x_norm
        return self.router_forward(x_norm, self.training, x)


class OrigDispatcherImpl(object):
    def __init__(self, num_experts: int, gates: torch.Tensor) -> None:
        self._gates = gates
        self._num_experts = num_experts
        non_zero_gate = torch.nonzero(gates)
        sorted_experts, index_sorted_experts = non_zero_gate.sort(0)

        _, self._expert_index = sorted_experts.split(1, dim=1)

        self._batch_index = non_zero_gate[index_sorted_experts[:, 1], 0]

        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]

        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, expert_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        expert_input = expert_inputs[self._batch_index].squeeze(1)
        # pyrefly: ignore [bad-return]
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

        # pyrefly: ignore [bad-argument-type]
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
            stable=STABLE_SORTING,
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


class MrnTest(unittest.TestCase):
    @given(
        B=st.sampled_from([16, 32]),
        max_seq_len=st.sampled_from([32, 64, 128]),
        input_d=st.sampled_from([64]),
        model_d=st.sampled_from([64]),
        num_experts=st.sampled_from([4, 8]),
        num_activated_experts=st.sampled_from([1, 2]),
        # group_norm=st.booleans(),
        group_norm=st.just(False),
        num_groups=st.sampled_from([1, 2]),
        # dtype=st.sampled_from([torch.bfloat16, torch.float32]),
        dtype=st.sampled_from([torch.bfloat16]),
        use_input_dtype=st.just(True),
        enable_noisy_gating=st.just(False),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    def test_mrn_numerics_parity_triton(self, *args, **kwargs) -> None:
        self._test_mrn_numerics_parity(
            *args,
            **kwargs,
            custom_kernel=True,
            train=True,
        )

    def _test_mrn_numerics_parity(
        self,
        B: int,
        max_seq_len: int,
        input_d: int,
        model_d: int,
        num_experts: int,
        num_activated_experts: int,
        group_norm: bool,
        num_groups: int,
        dtype: torch.dtype,
        train: bool,
        custom_kernel: bool,
        use_input_dtype: bool,
        enable_noisy_gating: bool,
    ) -> None:
        set_dev_mode(True)
        device = torch.device("cuda")

        assert input_d % num_groups == 0
        linear_dim = input_d // num_groups

        atol = 1e-2 if dtype == torch.bfloat16 else None
        rtol = 1e-2 if dtype == torch.bfloat16 else None

        torch.manual_seed(0)
        mrn_baseline = OrigSGMoEImpl(
            input_size=input_d,
            output_size=input_d,
            config=SGConfig(
                model_d=model_d,
                num_experts=num_experts,
                num_activated_experts=num_activated_experts,
                use_input_dtype=use_input_dtype,
                enable_noisy_gating=enable_noisy_gating,
            ),
        ).to(device)

        # The norm parameters is for original implementation only
        # In the new implementation, GroupNorm is a submodule of MRN
        norm_shape: int = num_groups if group_norm else input_d
        norm_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.ones((norm_shape,), device=device),
        )
        norm_bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((norm_shape,), device=device),
        )

        torch.manual_seed(0)
        mrn_new_impl = SGMoE(
            input_size=input_d,
            output_size=input_d,
            group_norm=group_norm,
            num_groups=num_groups,
            config=SGConfig(
                model_d=model_d,
                num_experts=num_experts,
                num_activated_experts=num_activated_experts,
                use_input_dtype=use_input_dtype,
                enable_noisy_gating=enable_noisy_gating,
            ),
            custom_kernel=custom_kernel,
            is_train=True,
            test_mode=True,
        ).to(device)

        lengths = torch.randint(low=1, high=max_seq_len + 1, size=(B,), device=device)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        x_baseline = torch.randn(
            int(lengths.sum().item()), input_d, device=device, dtype=dtype
        )
        x_new_impl = x_baseline.clone()

        x_baseline.requires_grad_()
        x_new_impl.requires_grad_()

        if not train:
            mrn_baseline.eval()
            mrn_new_impl.eval()

        torch.manual_seed(0)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(dtype == torch.bfloat16),
        ):
            y_baseline, loss_baseline, load_baseline = mrn_baseline.forward(
                x_baseline,
                x_offsets=offsets,
                max_seq_len=max_seq_len,
                group_norm=group_norm,
                num_groups=num_groups,
                linear_dim=linear_dim,
                norm_weight=norm_weight,
                norm_bias=norm_bias,
                norm_eps=mrn_new_impl.norm_eps,
            )

        torch.manual_seed(0)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(dtype == torch.bfloat16),
        ):
            y_new_impl, loss_new_impl, load_new_impl = mrn_new_impl(
                x_new_impl,
                x_offsets=offsets,
                max_seq_len=max_seq_len,
            )

        torch.testing.assert_close(
            y_baseline,
            y_new_impl,
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            loss_baseline,
            loss_new_impl,
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            load_baseline,
            load_new_impl,
            atol=atol,
            rtol=rtol,
        )

        if train:
            dout = torch.randn_like(y_baseline) * 0.01
            torch.manual_seed(0)
            y_baseline.backward(dout)
            torch.manual_seed(0)
            y_new_impl.backward(dout)

            for (n1, p1), (n2, p2) in zip(
                mrn_baseline.named_parameters(),
                [
                    (key, value)
                    for key, value in mrn_new_impl.named_parameters()
                    if "norm" not in key
                ],
                strict=True,
            ):
                self.assertEqual(n1, n2)
                torch.testing.assert_close(
                    none_throws(p1.grad),
                    none_throws(p2.grad),
                    atol=atol,
                    rtol=rtol,
                )

            torch.testing.assert_close(
                x_baseline.grad,
                x_new_impl.grad,
                atol=atol,
                rtol=rtol,
            )

            if group_norm:
                raise RuntimeError("group_norm is not supported yet")
            else:
                torch.testing.assert_close(
                    norm_weight.grad,
                    mrn_new_impl._norm_w.grad,
                    atol=atol,
                    rtol=rtol,
                )
                torch.testing.assert_close(
                    norm_bias.grad,
                    mrn_new_impl._norm_bias.grad,
                    atol=atol,
                    rtol=rtol,
                )
