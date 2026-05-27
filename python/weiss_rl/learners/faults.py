"""Numeric fault bundle helpers for learners."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from weiss_rl.learners.tensor_ops import nonfinite_indices
from weiss_rl.replay.bundles import write_fault_bundle

BatchValueGetter = Callable[[Any, str], Any]
FaultBundleWriter = Callable[[str, Any, dict[str, Any]], Path]

FAULT_BATCH_KEYS = (
    "obs",
    "actions",
    "legal_mask",
    "to_play_seat",
    "actor",
    "initial_hidden_state",
    "vtrace_rho_bar",
    "vtrace_c_bar",
)


def learner_batch_size(batch: Any, *, batch_value: BatchValueGetter) -> int:
    """Infer the fault payload batch size from common learner batch fields."""
    for key in ("rewards", "actions", "logits", "obs"):
        value = batch_value(batch, key)
        if value is not None:
            return int(np.asarray(value).size)
    return 1


def fault_dir_path(*, fault_dir: Path | None, checkpoint_dir: Path | None, logs_dir: Path | None) -> Path:
    """Resolve the learner numeric fault output directory."""
    if fault_dir is not None:
        return fault_dir
    if checkpoint_dir is not None:
        return checkpoint_dir / "faults"
    if logs_dir is not None:
        return logs_dir / "faults"
    return Path("faults")


def batch_fault_snapshot(batch: Any, *, batch_value: BatchValueGetter) -> dict[str, Any]:
    """Collect the batch fields persisted in numeric fault bundles."""
    snapshot: dict[str, Any] = {}
    for key in FAULT_BATCH_KEYS:
        value = batch_value(batch, key)
        if value is not None:
            snapshot[key] = value
    vtrace_result = batch_value(batch, "vtrace_result")
    if vtrace_result is not None:
        snapshot["vtrace_result"] = vtrace_result
    return snapshot


def write_numeric_fault_bundle(
    *,
    fault_dir: Path,
    stage: str,
    update_count: int,
    policy_version: int,
    batch_size: int,
    pass_action_id: int | None,
    batch_snapshot: dict[str, Any],
    context: dict[str, Any],
) -> Path:
    """Write an IMPALA learner numeric fault bundle."""
    return write_fault_bundle(
        fault_dir=fault_dir,
        prefix="learner_numeric_fault",
        payload={
            "format": "numeric_fault_bundle",
            "component": "impala_learner",
            "stage": stage,
            "update_count": update_count,
            "policy_version": policy_version,
            "batch_size": batch_size,
            "pass_action_id": pass_action_id,
            "batch": batch_snapshot,
            "context": context,
        },
    )


def ensure_finite_tensor(
    name: str,
    tensor: Tensor,
    *,
    batch: Any,
    context: dict[str, Any],
    write_bundle: FaultBundleWriter,
) -> None:
    """Raise with a persisted fault bundle if a tensor contains NaN or infinity."""
    if bool(torch.isfinite(tensor).all().item()):
        return
    fault_context = dict(context)
    fault_context[name] = tensor.detach()
    fault_context[f"{name}_nonfinite_indices"] = nonfinite_indices(tensor)
    fault_path = write_bundle(name, batch, fault_context)
    raise RuntimeError(f"non-finite learner {name}; wrote fault bundle to {fault_path}")


def collect_nonfinite_gradients(model: nn.Module | None, grad_norm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    """Return nonfinite parameter gradients and the grad-norm tensor."""
    if model is None:
        raise ValueError("ImpalaLearner requires a model")

    bad_gradients = {
        name: parameter.grad.detach()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
    }
    grad_norm_tensor = torch.as_tensor(grad_norm)
    return bad_gradients, grad_norm_tensor


def ensure_finite_gradients(
    *,
    batch: Any,
    context: dict[str, Any],
    grad_norm_tensor: Tensor,
    bad_gradients: dict[str, Tensor],
    write_bundle: FaultBundleWriter,
) -> None:
    """Raise with a persisted fault bundle if gradients or grad norm are nonfinite."""
    if not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item()):
        return
    raise_for_nonfinite_gradients(
        batch=batch,
        context=context,
        grad_norm_tensor=grad_norm_tensor,
        bad_gradients=bad_gradients,
        write_bundle=write_bundle,
    )


def raise_for_nonfinite_gradients(
    *,
    batch: Any,
    context: dict[str, Any],
    grad_norm_tensor: Tensor,
    bad_gradients: dict[str, Tensor],
    write_bundle: FaultBundleWriter,
) -> None:
    """Persist gradient fault context and raise the learner gradient error."""
    fault_context = dict(context)
    fault_context["grad_norm"] = grad_norm_tensor.detach()
    fault_context["grad_norm_nonfinite_indices"] = nonfinite_indices(grad_norm_tensor)
    if bad_gradients:
        fault_context["bad_gradient_names"] = sorted(bad_gradients)
        fault_context["bad_gradients"] = bad_gradients
    fault_path = write_bundle("gradients", batch, fault_context)
    raise RuntimeError(f"non-finite learner gradients; wrote fault bundle to {fault_path}")
