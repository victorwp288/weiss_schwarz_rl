from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SHA256_HEX_LENGTH = 64
_GIT_COMMIT_HEX_LENGTH = 40


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def _expected_sha256(value: str, *, flag_name: str) -> str:
    if not value.strip():
        return ""
    normalized = _normalize_sha256(value)
    if not normalized:
        raise ValueError(f"{flag_name} must be a 64-character lowercase or uppercase SHA-256 hex string")
    return normalized


def _require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    if expected and expected != actual:
        raise RuntimeError(f"{flag_name} mismatch: expected {expected}, observed {actual}")


def _normalize_git_commit(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _GIT_COMMIT_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
