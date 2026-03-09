"""Determinism and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_U64_MASK = (1 << 64) - 1
_U32_MASK = (1 << 32) - 1


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


def _u32_le(value: int) -> bytes:
    if value < 0 or value > _U32_MASK:
        raise ValueError(f"u32 value out of range: {value}")
    return value.to_bytes(4, byteorder="little", signed=False)


def _u64_le(value: int) -> bytes:
    if value < 0 or value > _U64_MASK:
        raise ValueError(f"u64 value out of range: {value}")
    return value.to_bytes(8, byteorder="little", signed=False)


def _tagged_bytes(tag: str, payload: bytes) -> bytes:
    tag_bytes = tag.encode("utf-8")
    return _u32_le(len(tag_bytes)) + tag_bytes + _u32_le(len(payload)) + payload


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
    payload = f"actor|{base_seed64}|{actor_id}".encode("utf-8")
    return stable_hash64(payload) & _U64_MASK


def derive_episode_seed(actor_seed64: int, env_id: int, episode_index: int) -> int:
    payload = f"episode|{actor_seed64}|{env_id}|{episode_index}".encode("utf-8")
    return stable_hash64(payload) & _U64_MASK


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
