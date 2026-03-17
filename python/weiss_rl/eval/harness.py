"""Deterministic evaluation harness scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from weiss_rl.masking import assert_strictly_increasing_legal_ids, masked_logp_from_legal_ids, masked_logp_from_mask

_CDF_RENORMALIZE_TOL = 1e-6


class _FloatRng(Protocol):
    def next_float(self) -> float: ...


@dataclass(slots=True)
class EvalSamplerAnomalies:
    cdf_renormalizations: int = 0


def eval_sampler_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def eval_sampler_logp_from_legal_ids(
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


def sample_action_pinned(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    rng: _FloatRng,
    pass_action_id: int | None = None,
    anomalies: EvalSamplerAnomalies | None = None,
) -> tuple[int, np.float32]:
    """Sample one action from a single packed legal-id row with pinned CPU CDF math."""
    logits_array = _coerce_eval_logits(logits)
    legal_ids_array = _coerce_eval_legal_ids(legal_ids, action_space=logits_array.shape[0])

    if legal_ids_array.size == 0:
        action = _require_pass_action(pass_action_id, action_space=logits_array.shape[0])
        logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=action)
        return action, logp

    assert_strictly_increasing_legal_ids(legal_ids_array)
    probs64 = _legal_probs_for_cdf(logits_array, legal_ids_array, anomalies=anomalies)
    action_index = _sample_cdf_index(probs64, rng=rng)
    action = int(legal_ids_array[action_index])
    logp = _selected_logp(logits_array, legal_ids_array, action, pass_action_id=pass_action_id)
    return action, logp


@dataclass(slots=True)
class MatchupSummary:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    truncations: int = 0


def summarize_pair_outcomes(outcomes: list[str]) -> MatchupSummary:
    out = MatchupSummary()
    for token in outcomes:
        key = token.strip().lower()
        if key == "w":
            out.wins += 1
        elif key == "l":
            out.losses += 1
        elif key == "d":
            out.draws += 1
        elif key == "t":
            out.truncations += 1
    return out


def _fault_env_indices(engine_status: Any) -> list[int]:
    return np.flatnonzero(np.atleast_1d(np.asarray(engine_status)) != 0).astype(int).tolist()


def _json_ready_array(value: Any) -> int | list[int]:
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array)
    return array.astype(int).tolist()


def _json_ready_episode_key(episode_key: Any) -> object:
    if isinstance(episode_key, (bytes, bytearray)):
        return repr(bytes(episode_key))

    array = np.asarray(episode_key)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (bytes, bytearray)):
            return repr(bytes(scalar))
        return scalar
    return array.tolist()


def abort_on_engine_fault_eval(
    *,
    run_dir: Path,
    engine_status: Any,
    decision_id: Any | None = None,
    episode_key: Any | None = None,
    note: str = "engine_status!=0 during evaluation",
) -> None:
    """Hard-fail evaluation on engine faults after writing a local artifact."""
    fault_env_indices = _fault_env_indices(engine_status)
    if not fault_env_indices:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    fault_path = run_dir / "eval_engine_fault.json"
    payload: dict[str, object] = {
        "note": note,
        "fault_env_indices": fault_env_indices,
        "engine_status": _json_ready_array(engine_status),
    }
    if decision_id is not None:
        payload["decision_id"] = _json_ready_array(decision_id)
    if episode_key is not None:
        payload["episode_key"] = _json_ready_episode_key(episode_key)

    fault_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    raise RuntimeError(f"{note}; wrote {fault_path}")


def _coerce_eval_logits(logits: np.ndarray) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim != 1:
        raise ValueError("logits must be a 1D array")
    return logits_array


def _coerce_eval_legal_ids(legal_ids: np.ndarray, *, action_space: int) -> np.ndarray:
    legal_ids_array = np.asarray(legal_ids)
    if legal_ids_array.ndim != 1:
        raise ValueError("legal_ids must be 1D")
    if legal_ids_array.dtype == np.bool_ or not np.issubdtype(legal_ids_array.dtype, np.integer):
        raise ValueError("legal_ids must be an integer array")

    signed = legal_ids_array.astype(np.int64, copy=False)
    if np.any(signed < 0):
        raise ValueError("legal_ids must be >= 0")
    if np.any(signed >= action_space):
        raise ValueError(f"legal_ids must be < action_space ({action_space})")
    return signed.astype(np.intp, copy=False)


def _require_pass_action(pass_action_id: int | None, *, action_space: int) -> int:
    if pass_action_id is None:
        raise ValueError("pass_action_id is required when legal_ids is empty")
    if pass_action_id < 0 or pass_action_id >= action_space:
        raise ValueError(f"pass_action_id must be in [0, {action_space})")
    return int(pass_action_id)


def _legal_probs_for_cdf(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    *,
    anomalies: EvalSamplerAnomalies | None = None,
) -> np.ndarray:
    legal_logits = logits[legal_ids]
    if not np.all(np.isfinite(legal_logits)):
        raise ValueError("legal logits must be finite")

    row_max = np.max(legal_logits)
    shifted = legal_logits - row_max
    weights = np.exp(shifted)
    denom = np.sum(weights, dtype=np.float32)
    probs64 = np.asarray(weights / denom, dtype=np.float64)
    return _normalize_cdf_probs(probs64, anomalies=anomalies)


def _normalize_cdf_probs(
    probs64: np.ndarray,
    *,
    anomalies: EvalSamplerAnomalies | None = None,
) -> np.ndarray:
    prob_sum = float(np.sum(probs64, dtype=np.float64))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise ValueError("legal probabilities must sum to a finite positive value")
    if abs(prob_sum - 1.0) > _CDF_RENORMALIZE_TOL:
        probs64 = probs64 / prob_sum
        if anomalies is not None:
            anomalies.cdf_renormalizations += 1
    return probs64


def _sample_cdf_index(probs64: np.ndarray, *, rng: _FloatRng) -> int:
    cdf = np.cumsum(probs64, dtype=np.float64)
    cdf[-1] = 1.0
    draw = float(rng.next_float())
    if not np.isfinite(draw) or draw < 0.0 or draw > 1.0:
        raise ValueError("rng.next_float() must return a finite value in [0.0, 1.0]")
    return min(int(np.searchsorted(cdf, draw, side="right")), cdf.size - 1)


def _selected_logp(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    action: int,
    *,
    pass_action_id: int | None,
) -> np.float32:
    legal_offsets = np.array([0, legal_ids.size], dtype=np.int64)
    actions = np.array([action], dtype=np.int64)
    logp = masked_logp_from_legal_ids(
        logits[np.newaxis, :],
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )
    return np.float32(logp[0])
