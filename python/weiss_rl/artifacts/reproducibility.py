"""Determinism and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_U64_MASK = (1 << 64) - 1
_U32_MASK = (1 << 32) - 1
_PYTHONHASHSEED_MAX = (1 << 32) - 1


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize with stable separators and key ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_seed_bytes(seeds: list[int]) -> bytes:
    """Serialize parsed seed values in a platform-stable form."""
    return canonical_json_bytes(seeds)


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return sha256_bytes(data).hex()


def stable_hash64(data: bytes) -> int:
    digest = sha256_bytes(data)
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def fixed_python_hash_seed() -> int | None:
    raw_value = os.environ.get("PYTHONHASHSEED", "").strip()
    if not raw_value or raw_value.lower() == "random":
        return None
    try:
        seed = int(raw_value)
    except ValueError:
        return None
    if 0 <= seed <= _PYTHONHASHSEED_MAX:
        return seed
    return None


def require_fixed_python_hash_seed(context: str) -> int:
    seed = fixed_python_hash_seed()
    if seed is None:
        raise RuntimeError(
            f"{context} requires a fixed PYTHONHASHSEED for reproducible simulator/action ordering. "
            "In PowerShell, run `$env:PYTHONHASHSEED='0'` before launching the command."
        )
    return seed


def _u32_le(value: int) -> bytes:
    if value < 0 or value > _U32_MASK:
        raise ValueError(f"u32 value out of range: {value}")
    return value.to_bytes(4, byteorder="little", signed=False)


def _u64_le(value: int) -> bytes:
    if value < 0 or value > _U64_MASK:
        raise ValueError(f"u64 value out of range: {value}")
    return value.to_bytes(8, byteorder="little", signed=False)


def _ensure_32_bytes(value: bytes, name: str) -> bytes:
    if len(value) != 32:
        raise ValueError(f"{name} must be 32 bytes, got {len(value)}")
    return value


def _tagged_bytes(tag: str, payload: bytes) -> bytes:
    tag_bytes = tag.encode("utf-8")
    return _u32_le(len(tag_bytes)) + tag_bytes + _u32_le(len(payload)) + payload


def _tagged_payload(tag: bytes, payload: bytes) -> bytes:
    return tag + _u32_le(len(payload)) + payload


def _git_commit_bytes(git_commit: str | bytes | None) -> bytes:
    if git_commit is None:
        return b""
    if isinstance(git_commit, bytes):
        return git_commit
    normalized = git_commit.strip()
    if len(normalized) == 40:
        try:
            return bytes.fromhex(normalized)
        except ValueError:
            pass
    return normalized.encode("ascii")


def serialize_run_identity(
    spec_hash256: str,
    config_hash256: str,
    git_commit: str | bytes | None,
    start_nonce: int,
) -> bytes:
    return b"".join(
        (
            _tagged_bytes("run", b""),
            _tagged_bytes("spec", bytes.fromhex(spec_hash256)),
            _tagged_bytes("config", bytes.fromhex(config_hash256)),
            _tagged_bytes("git", _git_commit_bytes(git_commit)),
            _tagged_bytes("nonce", _u64_le(start_nonce)),
        )
    )


def compute_run_id256(
    spec_hash256: str,
    config_hash256: str,
    git_commit: str | bytes | None,
    start_nonce: int,
) -> str:
    return sha256_hex(serialize_run_identity(spec_hash256, config_hash256, git_commit, start_nonce))


def compute_run_id64(
    spec_hash256: str,
    config_hash256: str,
    git_commit: str | bytes | None,
    start_nonce: int,
) -> int:
    return stable_hash64(serialize_run_identity(spec_hash256, config_hash256, git_commit, start_nonce))


def derive_actor_seed(base_seed64: int, actor_id: int) -> int:
    payload = f"actor|{base_seed64}|{actor_id}".encode()
    return stable_hash64(payload) & _U64_MASK


def derive_episode_seed(actor_seed64: int, env_id: int, episode_index: int) -> int:
    payload = f"episode|{actor_seed64}|{env_id}|{episode_index}".encode()
    return stable_hash64(payload) & _U64_MASK


def legal_fingerprint_v1(spec_hash256: bytes, decision_id: int, legal_ids: list[int] | np.ndarray) -> int:
    legal_ids_array = legal_ids if isinstance(legal_ids, np.ndarray) else np.asarray(legal_ids, dtype=np.uint32)
    if legal_ids_array.size > 1 and np.any(legal_ids_array[1:] <= legal_ids_array[:-1]):
        raise ValueError(f"legal_ids must be strictly increasing; got {legal_ids_array}")

    parts = [b"legal_fp_v1", _ensure_32_bytes(spec_hash256, "spec_hash256")]
    parts.append(decision_id.to_bytes(4, byteorder="little", signed=False))
    parts.append(len(legal_ids_array).to_bytes(4, byteorder="little", signed=False))
    for legal_id in legal_ids_array:
        parts.append(int(legal_id).to_bytes(4, byteorder="little", signed=False))
    return stable_hash64(b"".join(parts)) & _U64_MASK


def key256_to_short64(key256: bytes) -> int:
    key256 = _ensure_32_bytes(key256, "key256")
    return int.from_bytes(key256[:8], byteorder="little", signed=False)


def derive_episode_key256(
    *,
    run_id256: bytes,
    actor_id: int,
    env_id: int,
    episode_index: int,
    episode_seed64: int,
) -> bytes:
    run_id256 = _ensure_32_bytes(run_id256, "run_id256")
    payload = b"".join(
        (
            b"episode",
            run_id256,
            _u32_le(actor_id),
            _u32_le(env_id),
            _u32_le(episode_index),
            _u64_le(episode_seed64),
        )
    )
    return sha256_bytes(payload)


def derive_replay_key256(*, episode_key256: bytes, spec_hash256: bytes) -> bytes:
    episode_key256 = _ensure_32_bytes(episode_key256, "episode_key256")
    spec_hash256 = _ensure_32_bytes(spec_hash256, "spec_hash256")
    return sha256_bytes(b"".join((b"replay", episode_key256, spec_hash256)))


def normalize_simulator_episode_key256(simulator_episode_key: int | bytes) -> bytes:
    if isinstance(simulator_episode_key, int):
        return sha256_bytes(_tagged_payload(b"episode_u64", _u64_le(simulator_episode_key)))
    if len(simulator_episode_key) == 32:
        return simulator_episode_key
    return sha256_bytes(_tagged_payload(b"episode_bytes", simulator_episode_key))


def resolve_episode_key256(
    *,
    simulator_episode_key: int | bytes | None,
    run_id256: bytes,
    actor_id: int,
    env_id: int,
    episode_index: int,
    episode_seed64: int,
) -> bytes:
    if simulator_episode_key is None or simulator_episode_key == b"":
        return derive_episode_key256(
            run_id256=run_id256,
            actor_id=actor_id,
            env_id=env_id,
            episode_index=episode_index,
            episode_seed64=episode_seed64,
        )
    return normalize_simulator_episode_key256(simulator_episode_key)


def key256_to_hex(key256: bytes) -> str:
    return _ensure_32_bytes(key256, "key256").hex()


def parse_seed_file(path: Path) -> list[int]:
    """Parse a seed file with strict format: one u64 per line, no comments or blanks."""
    seeds = []
    for line_num, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            raise ValueError(f"Blank line {line_num} in {path}; seed files must contain one u64 per line")
        if line.startswith("#"):
            raise ValueError(f"Comment on line {line_num} in {path}; seed files do not allow comments")
        try:
            seed = int(line)
        except ValueError as err:
            raise ValueError(f"Invalid seed on line {line_num} in {path}: {line!r}") from err
        if not (0 <= seed < (1 << 64)):
            raise ValueError(f"Seed out of u64 range on line {line_num} in {path}: {seed}")
        seeds.append(seed)
    return seeds


def hash_seed_file(path: Path) -> str:
    """Hash parsed seed contents so equivalent files stay stable across checkouts."""
    seeds = parse_seed_file(path)
    return sha256_hex(canonical_seed_bytes(seeds))


def compute_seed_hashes(seed_sets: dict[str, Path]) -> dict[str, str]:
    """Compute hashes for all seed sets."""
    return {name: hash_seed_file(path) for name, path in seed_sets.items()}
