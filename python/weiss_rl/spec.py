"""Simulator spec-bundle compatibility helpers."""

from __future__ import annotations

from typing import Any


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
