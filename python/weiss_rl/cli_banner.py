"""CLI banner utilities for contract reporting at startup."""

from __future__ import annotations

from .spec import HARD_FAIL_SPEC_MISMATCH_POLICY


def print_startup_banner(
    spec_hash: str,
    config_hash: str,
    run_id: str,
    spec_mismatch_policy: str = HARD_FAIL_SPEC_MISMATCH_POLICY,
) -> None:
    """Print a startup contract banner with hashes, run ID, and policy."""
    print("=" * 80)
    print("STARTUP CONTRACT")
    print("=" * 80)
    print(f"spec_hash:              {spec_hash}")
    print(f"config_hash:            {config_hash}")
    print(f"run_id:                 {run_id}")
    print(f"spec_mismatch_policy:   {spec_mismatch_policy}")
    print("=" * 80)
