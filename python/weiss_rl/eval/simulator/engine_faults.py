"""Evaluation engine-fault artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def fault_env_indices(engine_status: Any) -> list[int]:
    return np.flatnonzero(np.atleast_1d(np.asarray(engine_status)) != 0).astype(int).tolist()


def json_ready_array(value: Any) -> int | list[int]:
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array)
    return array.astype(int).tolist()


def json_ready_episode_key(episode_key: Any) -> object:
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
    fault_indices = fault_env_indices(engine_status)
    if not fault_indices:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    fault_path = run_dir / "eval_engine_fault.json"
    payload: dict[str, object] = {
        "note": note,
        "fault_env_indices": fault_indices,
        "engine_status": json_ready_array(engine_status),
    }
    if decision_id is not None:
        payload["decision_id"] = json_ready_array(decision_id)
    if episode_key is not None:
        payload["episode_key"] = json_ready_episode_key(episode_key)

    fault_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    raise RuntimeError(f"{note}; wrote {fault_path}")


__all__ = [
    "abort_on_engine_fault_eval",
    "fault_env_indices",
    "json_ready_array",
    "json_ready_episode_key",
]
