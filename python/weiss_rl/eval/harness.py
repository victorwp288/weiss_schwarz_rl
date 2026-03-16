"""Deterministic evaluation harness scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import numpy as np

from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask


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
