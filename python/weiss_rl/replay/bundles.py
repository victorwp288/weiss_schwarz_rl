"""Replay bundle serialization scaffold."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from weiss_rl.repro import (
    derive_replay_key256,
    key256_to_hex,
    key256_to_short64,
    resolve_episode_key256,
)

torch: ModuleType | None
try:  # pragma: no cover - torch is optional here
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(slots=True)
class ReplayRecord:
    episode_key: str
    episode_key64: int
    replay_key256: str
    replay_key64: int
    decision_id: int
    action: int
    reward: float
    terminated: bool
    truncated: bool


def make_replay_record(
    *,
    simulator_episode_key: int | bytes | None,
    run_id256: bytes,
    spec_hash256: bytes,
    actor_id: int,
    env_id: int,
    episode_index: int,
    episode_seed64: int,
    decision_id: int,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> ReplayRecord:
    episode_key256 = resolve_episode_key256(
        simulator_episode_key=simulator_episode_key,
        run_id256=run_id256,
        actor_id=actor_id,
        env_id=env_id,
        episode_index=episode_index,
        episode_seed64=episode_seed64,
    )
    replay_key256 = derive_replay_key256(episode_key256=episode_key256, spec_hash256=spec_hash256)

    return ReplayRecord(
        episode_key=key256_to_hex(episode_key256),
        episode_key64=key256_to_short64(episode_key256),
        replay_key256=key256_to_hex(replay_key256),
        replay_key64=key256_to_short64(replay_key256),
        decision_id=decision_id,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )


def write_jsonl(records: list[ReplayRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")


def write_fault_bundle(*, fault_dir: Path, prefix: str, payload: dict[str, Any]) -> Path:
    fault_dir.mkdir(parents=True, exist_ok=True)
    path = fault_dir / f"{prefix}_{time.time_ns()}.json"
    path.write_text(json.dumps(_json_ready(payload), allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _nonfinite_token(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _json_ready(value.tolist()),
        }
    if torch is not None and isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().numpy())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return repr(value)


def _nonfinite_token(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value > 0:
        return "inf"
    return "-inf"
