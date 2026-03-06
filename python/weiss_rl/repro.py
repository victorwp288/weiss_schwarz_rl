"""Determinism and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

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


def legal_fingerprint_v1(
    spec_hash256: bytes,
    decision_id: int,
    legal_ids: list[int] | np.ndarray,
) -> int:
    """
    Compute legality fingerprint to detect determinism drift + serialization bugs.

    Per §16.6: combines spec_hash256, decision_id, and legal_ids into a uint64
    fingerprint used in evaluation, replay, and training diagnostics.

    Args:
        spec_hash256: 32-byte SHA256 hash of simulator spec.
        decision_id: uint32 decision identifier from simulator.
        legal_ids: strictly increasing sequence of uint32 action ids.

    Returns:
        uint64 fingerprint value.

    Raises:
        ValueError: if legal_ids are not strictly increasing (paper-grade hard fail).
    """
    # Validate strictly increasing (paper-grade invariant).
    if isinstance(legal_ids, np.ndarray):
        legal_ids_array = legal_ids
    else:
        legal_ids_array = np.asarray(legal_ids, dtype=np.uint32)

    if legal_ids_array.size > 1:
        if np.any(legal_ids_array[1:] <= legal_ids_array[:-1]):
            raise ValueError(
                f"legal_ids must be strictly increasing; got {legal_ids_array}"
            )

    # Build canonical bytes per spec format:
    # b"legal_fp_v1" || spec_hash256 ||
    # u32_le(decision_id) || u32_le(len(legal_ids)) ||
    # for id in legal_ids: u32_le(id)
    parts = [b"legal_fp_v1", spec_hash256]

    # Add decision_id as uint32 little-endian.
    parts.append(decision_id.to_bytes(4, byteorder="little", signed=False))

    # Add length of legal_ids as uint32 little-endian.
    parts.append(len(legal_ids_array).to_bytes(4, byteorder="little", signed=False))

    # Add each legal_id as uint32 little-endian.
    for id_val in legal_ids_array:
        parts.append(int(id_val).to_bytes(4, byteorder="little", signed=False))

    canonical_bytes = b"".join(parts)
    return stable_hash64(canonical_bytes) & _U64_MASK
