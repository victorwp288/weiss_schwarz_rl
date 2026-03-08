"""Determinism and reproducibility helpers."""

from __future__ import annotations

import hashlib
import struct
import json

from typing import Any

_U64_MASK = (1 << 64) - 1


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize with stable separators and key ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash64(data: bytes) -> int:
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def derive_actor_seed(base_seed64: int, actor_id: int) -> int:
    payload = f"actor|{base_seed64}|{actor_id}".encode("utf-8")
    return stable_hash64(payload) & _U64_MASK


def derive_episode_seed(actor_seed64: int, env_id: int, episode_index: int) -> int:
    payload = f"episode|{actor_seed64}|{env_id}|{episode_index}".encode("utf-8")
    return stable_hash64(payload) & _U64_MASK


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _u32_le(x: int) -> bytes:
    if x < 0 or x > 0xFFFFFFFF:
        raise ValueError(f"u32 out of range: {x}")
    return struct.pack("<I", x)


def _u64_le(x: int) -> bytes:
    if x < 0 or x > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"u64 out of range: {x}")
    return struct.pack("<Q", x)


def _ensure_32_bytes(x: bytes, name: str) -> bytes:
    if len(x) != 32:
        raise ValueError(f"{name} must be 32 bytes, got {len(x)}")
    return x


def key256_to_short64(key256: bytes) -> int:
    """Deterministic short id for filenames: first 8 bytes as little-endian uint64."""
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
    """§13.4 fallback: SHA-256('episode' || run_id256 || actor_id || env_id || episode_index || episode_seed64)."""
    run_id256 = _ensure_32_bytes(run_id256, "run_id256")
    payload = b"".join(
        [
            b"episode",
            run_id256,
            _u32_le(actor_id),
            _u32_le(env_id),
            _u32_le(episode_index),
            _u64_le(episode_seed64),
        ]
    )
    return sha256_bytes(payload)


def derive_replay_key256(*, episode_key256: bytes, spec_hash256: bytes) -> bytes:
    """§13.4: SHA-256('replay' || episode_key256 || spec_hash256)."""
    episode_key256 = _ensure_32_bytes(episode_key256, "episode_key256")
    spec_hash256 = _ensure_32_bytes(spec_hash256, "spec_hash256")
    payload = b"".join([b"replay", episode_key256, spec_hash256])
    return sha256_bytes(payload)


def resolve_episode_key256(
    *,
    simulator_episode_key: Optional[bytes],
    run_id256: bytes,
    actor_id: int,
    env_id: int,
    episode_index: int,
    episode_seed64: int,
) -> bytes:
    """Store simulator-provided deterministic episode key if present; else derive fallback per §13.4."""
    if simulator_episode_key is not None and len(simulator_episode_key) > 0:
        # Contract for *256*: must be 32 bytes.
        return _ensure_32_bytes(simulator_episode_key, "simulator_episode_key")
    return derive_episode_key256(
        run_id256=run_id256,
        actor_id=actor_id,
        env_id=env_id,
        episode_index=episode_index,
        episode_seed64=episode_seed64,
    )


def key256_to_hex(key256: bytes) -> str:
    key256 = _ensure_32_bytes(key256, "key256")
    return key256.hex()

