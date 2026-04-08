"""Replay bundle serialization (deterministic replay zip).

M5-07: save deterministic replay zip (actions + legal_fingerprint + episode keys).
Minimum bundle:
  - meta.json (episode identity + spec hash + replay key)
  - steps.jsonl (action sequence + actor seat + decision_id + engine_status + legality fingerprint)
Optional:
  - fault.json (invariant / engine fault metadata)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
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
        decision_id=int(decision_id),
        action=int(action),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def write_jsonl(records: list[ReplayRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")


# ----------------------------
# Fingerprinting (RL-layer)
# ----------------------------
_FINGERPRINT_PERSON = b"wslegal1"  # bump if algorithm changes


def compute_legal_fingerprint64(*, decision_id: int, legal_ids: np.ndarray) -> int:
    """Compute legality fingerprint (u64) from decision_id + packed legal_ids.

    This must be deterministic across platforms/runs for the same legality.
    """
    did = int(decision_id) & 0xFFFF_FFFF
    ids = np.asarray(legal_ids, dtype=np.uint16)
    h = hashlib.blake2b(digest_size=8, person=_FINGERPRINT_PERSON)
    h.update(did.to_bytes(4, byteorder="little", signed=False))
    # Always little-endian bytes for stability.
    h.update(ids.astype("<u2", copy=False).tobytes())
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def _legal_slice(legal_ids: np.ndarray, legal_offsets: np.ndarray, row: int) -> np.ndarray:
    offs = np.asarray(legal_offsets, dtype=np.uint32)
    ids = np.asarray(legal_ids, dtype=np.uint16)
    start = int(offs[row])
    end = int(offs[row + 1])
    if start < 0 or end < start or end > ids.shape[0]:
        raise ValueError("legal_offsets out of bounds for legal_ids")
    return ids[start:end]


# ----------------------------
# Bundle schema
# ----------------------------
@dataclass(frozen=True, slots=True)
class ReplayBundleMeta:
    schema_version: int
    created_utc_ns: int

    # IDs
    episode_key256: str
    episode_key64: int
    replay_key256: str
    replay_key64: int

    # Provenance
    run_id256: str
    spec_hash256: str
    actor_id: int
    env_id: int
    episode_index: int
    episode_seed64: int


@dataclass(frozen=True, slots=True)
class ReplayStep:
    t: int
    decision_id: int
    actor: int
    action: int
    reward: float
    terminated: bool
    truncated: bool
    engine_status: int
    legal_fingerprint64: int


# ----------------------------
# High-level helpers
# ----------------------------
def make_replay_bundle_meta(
    *,
    simulator_episode_key: int | bytes | None,
    run_id256: bytes,
    spec_hash256: bytes,
    actor_id: int,
    env_id: int,
    episode_index: int,
    episode_seed64: int,
) -> ReplayBundleMeta:
    episode_key256_bytes = resolve_episode_key256(
        simulator_episode_key=simulator_episode_key,
        run_id256=run_id256,
        actor_id=actor_id,
        env_id=env_id,
        episode_index=episode_index,
        episode_seed64=episode_seed64,
    )
    replay_key256_bytes = derive_replay_key256(episode_key256=episode_key256_bytes, spec_hash256=spec_hash256)
    return ReplayBundleMeta(
        schema_version=1,
        created_utc_ns=time.time_ns(),
        episode_key256=key256_to_hex(episode_key256_bytes),
        episode_key64=key256_to_short64(episode_key256_bytes),
        replay_key256=key256_to_hex(replay_key256_bytes),
        replay_key64=key256_to_short64(replay_key256_bytes),
        run_id256=key256_to_hex(run_id256),
        spec_hash256=key256_to_hex(spec_hash256),
        actor_id=int(actor_id),
        env_id=int(env_id),
        episode_index=int(episode_index),
        episode_seed64=int(episode_seed64),
    )


def write_replay_bundle(
    *,
    out_dir: Path,
    meta: ReplayBundleMeta,
    steps: list[ReplayStep],
    fault_payload: dict[str, Any] | None = None,
) -> Path:
    """Write replay bundle zip and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"replay_{meta.replay_key64:016x}.zip"

    meta_bytes = (json.dumps(asdict(meta), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    def iter_steps_jsonl() -> bytes:
        lines = []
        for s in steps:
            lines.append(json.dumps(asdict(s), sort_keys=True, separators=(",", ":")))
        return ("\n".join(lines) + "\n").encode("utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", meta_bytes)
        zf.writestr("steps.jsonl", iter_steps_jsonl())
        if fault_payload is not None:
            fault_text = json.dumps(_json_ready(fault_payload), allow_nan=False, sort_keys=True) + "\n"
            zf.writestr("fault.json", fault_text.encode("utf-8"))

    return zip_path


def load_replay_bundle(path: Path) -> tuple[ReplayBundleMeta, list[ReplayStep], dict[str, Any] | None]:
    """Load meta + steps (+ optional fault) from replay zip."""
    with zipfile.ZipFile(path, "r") as zf:
        meta_raw = json.loads(zf.read("meta.json").decode("utf-8"))
        steps_raw = zf.read("steps.jsonl").decode("utf-8").splitlines()
        fault = None
        if "fault.json" in zf.namelist():
            fault = json.loads(zf.read("fault.json").decode("utf-8"))

    meta = ReplayBundleMeta(**meta_raw)
    steps: list[ReplayStep] = [ReplayStep(**json.loads(line)) for line in steps_raw if line.strip()]
    return meta, steps, fault


# ----------------------------
# Deterministic rerun (fast layout)
# ----------------------------
def rerun_replay_bundle_fast(
    *,
    bundle_path: Path,
    max_decisions: int,
    max_ticks: int,
    observation_visibility: str,
) -> None:
    """Re-run a replay bundle deterministically and assert fingerprint matches.

    Uses EnvPool fast layout i16_legal_ids:
      reset_into_i16_legal_ids(out)
      step_into_i16_legal_ids(actions, out)   # note arg order
    """
    meta, steps, _fault = load_replay_bundle(bundle_path)

    # Use weiss_rl pool factory to get the EnvPool API that supports reset_into_*.
    from weiss_rl.envs.pool_factory import make_env_pool_from_config
    import weiss_sim

    pool, layout = make_env_pool_from_config(
        {
            "max_decisions": int(max_decisions),
            "max_ticks": int(max_ticks),
            "observation_visibility": str(observation_visibility),
            "seed": int(meta.episode_seed64 & 0xFFFF_FFFF),
        },
        profile="fast",
        num_envs=1,
    )
    if layout != "i16_legal_ids":
        raise RuntimeError(f"rerun requires i16_legal_ids layout, got {layout!r}")

    out = weiss_sim.BatchOutMinimalI16LegalIds(num_envs=1)
    pool.reset_into_i16_legal_ids(out)

    actions = np.ascontiguousarray(np.zeros((1,), dtype=np.uint32))

    for t, s in enumerate(steps):
        did = int(out.decision_id[0])
        actor = int(out.actor[0])

        if did != int(s.decision_id):
            raise RuntimeError(f"decision_id mismatch at t={t}: sim={did} bundle={s.decision_id}")
        if actor != int(s.actor):
            raise RuntimeError(f"actor mismatch at t={t}: sim={actor} bundle={s.actor}")

        row_legal = _legal_slice(out.legal_ids, out.legal_offsets, 0)
        fp = compute_legal_fingerprint64(decision_id=did, legal_ids=row_legal)
        if fp != int(s.legal_fingerprint64):
            raise RuntimeError(f"legal_fingerprint64 mismatch at t={t}: sim={fp} bundle={s.legal_fingerprint64}")

        actions[0] = np.uint32(int(s.action))
        pool.step_into_i16_legal_ids(actions, out)  # IMPORTANT: (actions, out)

        r = float(out.rewards[0])
        term = bool(out.terminated[0])
        trunc = bool(out.truncated[0])
        eng = int(out.engine_status[0])

        if not _close_float(r, float(s.reward)):
            raise RuntimeError(f"reward mismatch at t={t}: sim={r} bundle={s.reward}")
        if term != bool(s.terminated):
            raise RuntimeError(f"terminated mismatch at t={t}: sim={term} bundle={s.terminated}")
        if trunc != bool(s.truncated):
            raise RuntimeError(f"truncated mismatch at t={t}: sim={trunc} bundle={s.truncated}")
        if eng != int(s.engine_status):
            raise RuntimeError(f"engine_status mismatch at t={t}: sim={eng} bundle={s.engine_status}")

        if term or trunc:
            break


def _close_float(a: float, b: float, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


# ----------------------------
# Existing fault JSON helper
# ----------------------------
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
