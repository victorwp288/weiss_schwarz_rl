from __future__ import annotations

import sys
from typing import Any

SHA256_HEX_LENGTH = 64


def normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != SHA256_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def expected_sha256(value: str, *, flag_name: str) -> str:
    if not value.strip():
        return ""
    normalized = normalize_sha256(value)
    if not normalized:
        raise ValueError(f"{flag_name} must be a 64-character lowercase or uppercase SHA-256 hex string")
    return normalized


def require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    if expected and expected != actual:
        raise RuntimeError(f"{flag_name} mismatch: expected {expected}, observed {actual}")


def resolve_run_label(parser: Any, run_label: str, run_id_alias: str) -> str:
    normalized_label = run_label.strip()
    normalized_alias = run_id_alias.strip()
    if normalized_label and normalized_alias and normalized_label != normalized_alias:
        parser.error("--run-label and deprecated --run-id must match when both are provided")
    if normalized_alias:
        print("Warning: --run-id is deprecated; use --run-label instead.", file=sys.stderr)
    return normalized_label or normalized_alias


def require_positive_int(name: str, value: int) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return number


def spec_mismatch_policy(stack: Any) -> str:
    return "hard_fail"
