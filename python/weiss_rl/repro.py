"""Determinism and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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


def parse_seed_file(path: Path) -> list[int]:
    """Parse a seed file with strict format: one u64 per line, no comments or blanks."""
    seeds = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            seed = int(line)
        except ValueError:
            raise ValueError(f"Invalid seed on line {line_num} in {path}: {line!r}")
        if not (0 <= seed < (1 << 64)):
            raise ValueError(f"Seed out of u64 range on line {line_num} in {path}: {seed}")
        seeds.append(seed)
    return seeds


def hash_seed_file(path: Path) -> str:
    """Compute SHA256 hash of the seed file content."""
    content = path.read_text(encoding="utf-8")
    return sha256_hex(content.encode("utf-8"))


def compute_seed_hashes(seed_sets: dict[str, Path]) -> dict[str, str]:
    """Compute hashes for all seed sets."""
    return {name: hash_seed_file(path) for name, path in seed_sets.items()}
