"""CLI banner utilities for contract reporting at startup."""

from __future__ import annotations

from weiss_rl.core.spec import HARD_FAIL_SPEC_MISMATCH_POLICY


def print_startup_banner(
    spec_hash: str,
    config_hash: str,
    *,
    run_id64: str = "",
    run_id256: str = "",
    run_label: str = "",
    run_dir_name: str = "",
    spec_mismatch_policy: str = HARD_FAIL_SPEC_MISMATCH_POLICY,
) -> None:
    print("=" * 80)
    print("STARTUP CONTRACT")
    print("=" * 80)
    print(f"spec_hash:              {spec_hash}")
    print(f"config_hash:            {config_hash}")
    if run_id64:
        print(f"computed_run_id64:      {run_id64}")
    if run_id256:
        print(f"computed_run_id256:     {run_id256}")
    print(f"run_label:              {run_label or '(default)'}")
    if run_dir_name:
        print(f"run_dir_name:           {run_dir_name}")
    print(f"spec_mismatch_policy:   {spec_mismatch_policy}")
    print("=" * 80)
