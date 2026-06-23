"""Normal IMPALA update input validation and V-trace diagnostics."""

from __future__ import annotations

from typing import Any

from weiss_rl.learners.impala.batch_access import batch_value as _batch_value
from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.learners.vtrace_diagnostics import summarize_vtrace_diagnostics

_MISSING_TRAINING_INPUTS_MESSAGE = (
    "batch must include obs, actions, legality, and either vtrace_result or raw vtrace inputs for learner updates"
)


def has_impala_training_inputs(batch: Any) -> bool:
    return _batch_value(batch, "obs") is not None


def resolve_impala_update_vtrace_result(batch: Any) -> Any:
    return _batch_value(batch, "vtrace_result")


def missing_impala_training_input_fields(*, learner: Any, batch: Any) -> list[str]:
    missing = [key for key in ("obs", "actions") if _batch_value(batch, key) is None]
    if not learner._has_legal_actions(batch):
        missing.append("legal_actions")
    has_vtrace_targets = isinstance(_batch_value(batch, "vtrace_result"), VTraceTargets)
    has_raw_vtrace_inputs = learner._has_raw_vtrace_inputs(batch)
    if not has_vtrace_targets and not has_raw_vtrace_inputs:
        missing.append("vtrace_result_or_raw_inputs")
    return missing


def validate_impala_training_inputs(*, learner: Any, batch: Any) -> None:
    missing = missing_impala_training_input_fields(learner=learner, batch=batch)
    if missing:
        missing_fields = ", ".join(missing)
        raise ValueError(f"{_MISSING_TRAINING_INPUTS_MESSAGE}; missing {missing_fields}")


def summarize_precomputed_vtrace_update_metrics(
    *,
    learner: Any,
    batch: Any,
    vtrace_result: Any,
) -> dict[str, float]:
    if not isinstance(vtrace_result, VTraceTargets):
        return {}
    rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
    c_bar_value = _batch_value(batch, "vtrace_c_bar")
    rho_bar = learner.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
    c_bar = learner.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
    return summarize_vtrace_diagnostics(vtrace_result, rho_bar=rho_bar, c_bar=c_bar)


__all__ = [
    "has_impala_training_inputs",
    "missing_impala_training_input_fields",
    "resolve_impala_update_vtrace_result",
    "summarize_precomputed_vtrace_update_metrics",
    "validate_impala_training_inputs",
]
