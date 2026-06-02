"""IMPALA numeric fault bundle support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor

from weiss_rl.learners.faults import (
    batch_fault_snapshot,
    collect_nonfinite_gradients,
    ensure_finite_gradients,
    ensure_finite_tensor,
    fault_dir_path,
    learner_batch_size,
    raise_for_nonfinite_gradients,
    write_numeric_fault_bundle,
)


def _batch_value(batch: Any, key: str) -> Any:
    # Resolve through impala_learner so the historical helper remains the compatibility hook.
    from weiss_rl.learners import impala_learner as learner_module

    return learner_module._batch_value(batch, key)


class ImpalaFaultSupportMixin:
    def _batch_size(self: Any, batch: Any) -> int:
        return learner_batch_size(batch, batch_value=_batch_value)

    def _fault_dir_path(self: Any) -> Path:
        return fault_dir_path(fault_dir=self.fault_dir, checkpoint_dir=self.checkpoint_dir, logs_dir=self.logs_dir)

    def _batch_fault_snapshot(self: Any, batch: Any) -> dict[str, Any]:
        return batch_fault_snapshot(batch, batch_value=_batch_value)

    def _write_numeric_fault_bundle(self: Any, *, stage: str, batch: Any, context: dict[str, Any]) -> Path:
        return write_numeric_fault_bundle(
            fault_dir=self._fault_dir_path(),
            stage=stage,
            update_count=self.update_count,
            policy_version=self.policy_version,
            batch_size=self._batch_size(batch),
            pass_action_id=self.pass_action_id,
            batch_snapshot=self._batch_fault_snapshot(batch),
            context=context,
        )

    def _ensure_finite_tensor(
        self: Any,
        name: str,
        tensor: Tensor,
        *,
        batch: Any,
        context: dict[str, Any],
    ) -> None:
        ensure_finite_tensor(
            name, tensor, batch=batch, context=context, write_bundle=self._write_fault_bundle_for_stage
        )

    def _collect_nonfinite_gradients(self: Any, grad_norm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        return collect_nonfinite_gradients(self.model, grad_norm)

    def _ensure_finite_gradients(self: Any, *, batch: Any, context: dict[str, Any], grad_norm: Tensor) -> None:
        bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
        ensure_finite_gradients(
            batch=batch,
            context=context,
            grad_norm_tensor=grad_norm_tensor,
            bad_gradients=bad_gradients,
            write_bundle=self._write_fault_bundle_for_stage,
        )

    def _raise_for_nonfinite_gradients(
        self: Any,
        *,
        batch: Any,
        context: dict[str, Any],
        grad_norm_tensor: Tensor,
        bad_gradients: dict[str, Tensor],
    ) -> None:
        raise_for_nonfinite_gradients(
            batch=batch,
            context=context,
            grad_norm_tensor=grad_norm_tensor,
            bad_gradients=bad_gradients,
            write_bundle=self._write_fault_bundle_for_stage,
        )

    def _write_fault_bundle_for_stage(self: Any, stage: str, batch: Any, context: dict[str, Any]) -> Path:
        return self._write_numeric_fault_bundle(stage=stage, batch=batch, context=context)


__all__ = ["ImpalaFaultSupportMixin"]
