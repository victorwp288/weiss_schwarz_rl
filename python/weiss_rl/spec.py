"""Simulator spec-bundle compatibility helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any
import warnings


class SpecMismatchPolicy(Enum):
    """Policy for handling spec bundle mismatches."""

    HARD_FAIL = "hard_fail"
    """Raise RuntimeError immediately on mismatch (paper-grade evaluation default)."""

    WARN = "warn"
    """Log a warning but continue execution (training profiles allowed)."""

    IGNORE = "ignore"
    """Silently continue without checking."""


def assert_spec_compatibility(
    expected_spec_hash: int | str,
    observed_bundle: dict[str, Any],
    policy: SpecMismatchPolicy = SpecMismatchPolicy.HARD_FAIL,
) -> None:
    """Handle spec bundle mismatch according to configured policy.

    Args:
        expected_spec_hash: The expected spec hash value.
        observed_bundle: The bundle dict containing the observed spec_hash.
        policy: Mismatch policy (HARD_FAIL, WARN, or IGNORE).

    Raises:
        RuntimeError: If policy is HARD_FAIL and hashes don't match.
    """
    observed = observed_bundle.get("spec_hash")
    if str(observed) == str(expected_spec_hash):
        return

    mismatch_msg = (
        f"Spec mismatch: expected {expected_spec_hash}, observed {observed}. "
        "Mixed contracts may produce unreproducible artifacts."
    )

    if policy == SpecMismatchPolicy.HARD_FAIL:
        raise RuntimeError(mismatch_msg)
    elif policy == SpecMismatchPolicy.WARN:
        warnings.warn(mismatch_msg, RuntimeWarning, stacklevel=2)
    elif policy == SpecMismatchPolicy.IGNORE:
        pass
