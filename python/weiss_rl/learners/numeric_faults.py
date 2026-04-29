"""Numeric fault-bundle helpers for learner updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from weiss_rl.replay.bundles import write_fault_bundle


def batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(key)
    return getattr(batch, key, None)


def nonfinite_indices(values: Tensor | np.ndarray) -> np.ndarray:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return np.argwhere(~np.isfinite(array)).astype(np.int64, copy=False)


def collect_nonfinite_gradients(model: nn.Module | None, grad_norm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    if model is None:
        raise ValueError("ImpalaLearner requires a model")

    bad_gradients = {
        name: parameter.grad.detach()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
    }
    grad_norm_tensor = torch.as_tensor(grad_norm)
    return bad_gradients, grad_norm_tensor


@dataclass(frozen=True, slots=True)
class LearnerNumericFaultReporter:
    fault_dir: Path | None
    checkpoint_dir: Path | None
    logs_dir: Path | None
    update_count: int
    policy_version: int
    pass_action_id: int | None

    def fault_dir_path(self) -> Path:
        if self.fault_dir is not None:
            return self.fault_dir
        if self.checkpoint_dir is not None:
            return self.checkpoint_dir / "faults"
        if self.logs_dir is not None:
            return self.logs_dir / "faults"
        return Path("faults")

    def batch_size(self, batch: Any) -> int:
        for key in ("rewards", "actions", "logits", "obs"):
            value = batch_value(batch, key)
            if value is not None:
                return int(np.asarray(value).size)
        return 1

    def batch_fault_snapshot(self, batch: Any) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for key in (
            "obs",
            "actions",
            "legal_mask",
            "to_play_seat",
            "actor",
            "initial_hidden_state",
            "vtrace_rho_bar",
            "vtrace_c_bar",
        ):
            value = batch_value(batch, key)
            if value is not None:
                snapshot[key] = value
        vtrace_result = batch_value(batch, "vtrace_result")
        if vtrace_result is not None:
            snapshot["vtrace_result"] = vtrace_result
        return snapshot

    def write_numeric_fault_bundle(self, *, stage: str, batch: Any, context: dict[str, Any]) -> Path:
        return write_fault_bundle(
            fault_dir=self.fault_dir_path(),
            prefix="learner_numeric_fault",
            payload={
                "format": "numeric_fault_bundle",
                "component": "impala_learner",
                "stage": stage,
                "update_count": self.update_count,
                "policy_version": self.policy_version,
                "batch_size": self.batch_size(batch),
                "pass_action_id": self.pass_action_id,
                "batch": self.batch_fault_snapshot(batch),
                "context": context,
            },
        )

    def ensure_finite_tensor(
        self,
        name: str,
        tensor: Tensor,
        *,
        batch: Any,
        context: dict[str, Any],
    ) -> None:
        if bool(torch.isfinite(tensor).all().item()):
            return
        fault_context = dict(context)
        fault_context[name] = tensor.detach()
        fault_context[f"{name}_nonfinite_indices"] = nonfinite_indices(tensor)
        fault_path = self.write_numeric_fault_bundle(stage=name, batch=batch, context=fault_context)
        raise RuntimeError(f"non-finite learner {name}; wrote fault bundle to {fault_path}")

    def raise_for_nonfinite_gradients(
        self,
        *,
        batch: Any,
        context: dict[str, Any],
        grad_norm_tensor: Tensor,
        bad_gradients: dict[str, Tensor],
    ) -> None:
        fault_context = dict(context)
        fault_context["grad_norm"] = grad_norm_tensor.detach()
        fault_context["grad_norm_nonfinite_indices"] = nonfinite_indices(grad_norm_tensor)
        if bad_gradients:
            fault_context["bad_gradient_names"] = sorted(bad_gradients)
            fault_context["bad_gradients"] = bad_gradients
        fault_path = self.write_numeric_fault_bundle(stage="gradients", batch=batch, context=fault_context)
        raise RuntimeError(f"non-finite learner gradients; wrote fault bundle to {fault_path}")
