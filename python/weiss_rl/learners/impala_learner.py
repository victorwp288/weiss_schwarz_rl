"""IMPALA learner helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask


VTRACE_RHO_PERCENTILES = (50, 90, 95, 99)


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def summarize_vtrace_diagnostics(
    result: VTraceTargets,
    *,
    rho_bar: float,
    c_bar: float,
) -> dict[str, float]:
    flat_rhos = np.asarray(result.rhos, dtype=np.float64).reshape(-1)
    if flat_rhos.size == 0:
        raise ValueError("result.rhos must not be empty")

    metrics = {
        f"vtrace_rho_p{percentile}": float(np.percentile(flat_rhos, percentile))
        for percentile in VTRACE_RHO_PERCENTILES
    }
    metrics["vtrace_rho_clip_rate"] = float(np.mean(flat_rhos > rho_bar))
    metrics["vtrace_c_clip_rate"] = float(np.mean(flat_rhos > c_bar))
    return metrics


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


@dataclass(slots=True)
class ImpalaLearner:
    learning_rate: float = 2e-4

    def update(self, batch: Any) -> dict[str, float]:
        """Learner update hook."""
        metrics = {"loss": 0.0}
        vtrace_result = _batch_value(batch, "vtrace_result")
        if isinstance(vtrace_result, VTraceTargets):
            rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
            c_bar_value = _batch_value(batch, "vtrace_c_bar")
            rho_bar = 1.0 if rho_bar_value is None else float(rho_bar_value)
            c_bar = 1.0 if c_bar_value is None else float(c_bar_value)
            metrics.update(summarize_vtrace_diagnostics(vtrace_result, rho_bar=rho_bar, c_bar=c_bar))
        return metrics
