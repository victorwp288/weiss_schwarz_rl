"""Replay bundle serialization (deterministic replay zip).

M5-07: save deterministic replay zip (actions + legal_fingerprint + episode keys).
Minimum bundle:
  - meta.json (episode identity + spec hash + replay key)
  - steps.jsonl (action sequence + actor seat + decision_id + engine_status + legality fingerprint)
Optional:
  - fault.json (invariant / engine fault metadata)
"""

from __future__ import annotations

import json
import math
import time
import zipfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from weiss_rl.artifacts.reproducibility import (
    derive_replay_key256,
    key256_to_hex,
    key256_to_short64,
    legal_fingerprint_v1,
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


@dataclass(frozen=True, slots=True)
class ReplayRerunContract:
    version: int
    observation_visibility: str
    max_decisions: int
    max_ticks: int
    reward_json: str | None = None
    curriculum_json: str | None = None
    deck: str | None = None
    opponent_deck: str | None = None


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
    episode_identity_source: str
    simulator_episode_key_kind: str | None = None
    simulator_episode_key_u64: int | None = None
    simulator_episode_key_hex: str | None = None
    rerun_contract: ReplayRerunContract | None = None
    rerun_supported: bool = False
    rerun_blocker: str | None = None


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
def compute_legal_fingerprint64(*, spec_hash256: bytes, decision_id: int, legal_ids: np.ndarray) -> int:
    """Compute the canonical replay/eval legality fingerprint.

    This is a thin wrapper around the normative ``legal_fingerprint_v1`` contract
    so replay bundles stay aligned with the repo's paper-grade reproducibility
    rules, including malformed-input rejection.
    """
    return int(
        legal_fingerprint_v1(
            spec_hash256=spec_hash256,
            decision_id=int(decision_id),
            legal_ids=np.asarray(legal_ids),
        )
    )


def _legal_slice(legal_ids: np.ndarray, legal_offsets: np.ndarray, row: int) -> np.ndarray:
    offs = np.asarray(legal_offsets, dtype=np.uint32)
    ids = np.asarray(legal_ids, dtype=np.uint16)
    start = int(offs[row])
    end = int(offs[row + 1])
    if start < 0 or end < start or end > ids.shape[0]:
        raise ValueError("legal_offsets out of bounds for legal_ids")
    return ids[start:end]


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
    rerun_contract: ReplayRerunContract | None = None,
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
    simulator_episode_key_kind, simulator_episode_key_u64, simulator_episode_key_hex = _describe_simulator_episode_key(
        simulator_episode_key
    )
    rerun_supported = rerun_contract is not None
    return ReplayBundleMeta(
        schema_version=3 if rerun_supported else 2,
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
        episode_identity_source="simulator" if simulator_episode_key_kind is not None else "derived",
        simulator_episode_key_kind=simulator_episode_key_kind,
        simulator_episode_key_u64=simulator_episode_key_u64,
        simulator_episode_key_hex=simulator_episode_key_hex,
        rerun_contract=rerun_contract,
        rerun_supported=rerun_supported,
        rerun_blocker=None if rerun_supported else _missing_rerun_contract_blocker(),
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

    meta = _parse_replay_bundle_meta(meta_raw)
    steps: list[ReplayStep] = [ReplayStep(**json.loads(line)) for line in steps_raw if line.strip()]
    return meta, steps, fault


# ----------------------------
# Deterministic rerun (fast layout)
# ----------------------------
def rerun_replay_bundle_fast(
    *,
    bundle_path: Path,
    max_decisions: int | None = None,
    max_ticks: int | None = None,
    observation_visibility: str | None = None,
    report_path: Path | None = None,
) -> None:
    """Rerun replay verification using the contract persisted in the bundle.

    Caller-supplied simulator reconstruction args are rejected. Replay
    verification must use the bundle's persisted rerun contract or fail fast.
    """
    from weiss_rl.replay.runner import verify_replay_bundle

    if any(value is not None for value in (max_decisions, max_ticks, observation_visibility)):
        raise TypeError(
            "rerun_replay_bundle_fast() no longer accepts simulator override args; "
            "persist the rerun contract in the replay bundle instead"
        )

    verify_replay_bundle(bundle_path=bundle_path, report_path=report_path)


def _describe_simulator_episode_key(
    simulator_episode_key: int | bytes | None,
) -> tuple[str | None, int | None, str | None]:
    if simulator_episode_key is None or simulator_episode_key == b"":
        return None, None, None
    if isinstance(simulator_episode_key, bytes):
        return "bytes", None, simulator_episode_key.hex()
    return "u64", int(simulator_episode_key), None


def _parse_replay_bundle_meta(meta_raw: dict[str, Any]) -> ReplayBundleMeta:
    parsed = dict(meta_raw)
    contract_raw = parsed.get("rerun_contract")
    if isinstance(contract_raw, dict):
        parsed["rerun_contract"] = ReplayRerunContract(**contract_raw)
    return ReplayBundleMeta(**parsed)


def _missing_rerun_contract_blocker() -> str:
    return (
        "Replay bundle does not include the full deterministic rerun contract yet. "
        "Stored identity is authoritative, but simulator config/spec inputs required "
        "for reconstruction are intentionally not reconstructed from caller-supplied args."
    )


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
