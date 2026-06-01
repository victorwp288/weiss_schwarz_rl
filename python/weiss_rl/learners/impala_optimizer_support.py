"""IMPALA optimizer construction and parameter-group policy."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch.optim import Optimizer

_OPPONENT_CONTEXT_ADAPTER_NAMES = {
    "opponent_context_action_bias_adapter",
    "opponent_context_hidden_adapter",
    "opponent_context_recurrent_adapter",
}
_OPPONENT_CONTEXT_CANDIDATE_RESIDUAL_PREFIX = "opponent_context_candidate_residual_"


class ImpalaOptimizerSupportMixin:
    def _optimizer_for_step(self: Any) -> Optimizer:
        if self.optimizer is None:
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model before creating an optimizer")
            self.optimizer = cast(Any, torch.optim.Adam(self._optimizer_parameter_groups(), lr=self.learning_rate))
        return cast(Optimizer, self.optimizer)

    def _optimizer_parameter_groups(self: Any) -> Any:
        model = self.model
        if model is None:
            raise ValueError("ImpalaLearner requires a model before creating an optimizer")
        adapter_names = {
            name
            for name, _parameter in model.named_parameters()
            if name in _OPPONENT_CONTEXT_ADAPTER_NAMES or name.startswith(_OPPONENT_CONTEXT_CANDIDATE_RESIDUAL_PREFIX)
        }
        multiplier = float(getattr(model, "opponent_context_adapter_lr_multiplier", 1.0))
        if bool(getattr(model, "opponent_context_adapter_train_only", False)):
            if not adapter_names:
                raise ValueError(
                    "opponent_context_adapter_train_only requires at least one trainable opponent-context adapter"
                )
            adapter_params: list[Tensor] = []
            for name, parameter in model.named_parameters():
                trainable = name in adapter_names
                parameter.requires_grad_(trainable)
                if trainable:
                    adapter_params.append(parameter)
            if not adapter_params:
                raise ValueError(
                    "opponent_context_adapter_train_only found no trainable opponent-context adapter parameters"
                )
            return [{"params": adapter_params, "lr": float(self.learning_rate) * multiplier}]
        if not adapter_names or multiplier == 1.0:
            return model.parameters()
        scaled_adapter_params: list[Tensor] = []
        base_params: list[Tensor] = []
        for name, parameter in model.named_parameters():
            if name in adapter_names:
                scaled_adapter_params.append(parameter)
            else:
                base_params.append(parameter)
        if not scaled_adapter_params:
            return model.parameters()
        return [
            {"params": base_params},
            {"params": scaled_adapter_params, "lr": float(self.learning_rate) * multiplier},
        ]


__all__ = ["ImpalaOptimizerSupportMixin"]
