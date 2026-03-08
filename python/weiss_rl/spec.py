"""Simulator spec-bundle compatibility helpers."""

from __future__ import annotations

from typing import Any

HARD_FAIL_SPEC_MISMATCH_POLICY = "hard_fail"


def assert_spec_compatibility(
    expected_spec_hash: int | str,
    observed_bundle: dict[str, Any],
) -> None:
    """Raise when the simulator spec hash does not match the expected value."""
    observed = observed_bundle.get("spec_hash")
    if str(observed) != str(expected_spec_hash):
        raise RuntimeError(
            f"Spec mismatch: expected {expected_spec_hash}, observed {observed}. "
            "Refuse to continue with mixed contracts."
        )


def normalize_spec_mismatch_policy(value: str | None, *, source: str) -> str:
    """Allow only the master-plan fail-fast policy."""
    token = (value or HARD_FAIL_SPEC_MISMATCH_POLICY).strip().lower()
    if token != HARD_FAIL_SPEC_MISMATCH_POLICY:
        raise ValueError(
            f"{source} must be '{HARD_FAIL_SPEC_MISMATCH_POLICY}' to satisfy the fail-fast contract; "
            f"got {value!r}"
        )
    return token


def require_fail_on_spec_mismatch(value: bool | None, *, source: str) -> str:
    """Require fail-fast boolean config flags to stay enabled."""
    if value is False:
        raise ValueError(f"{source} must stay true to satisfy the fail-fast contract")
    return HARD_FAIL_SPEC_MISMATCH_POLICY
