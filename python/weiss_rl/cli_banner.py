"""CLI banner utilities for contract reporting at startup."""

from __future__ import annotations


def print_startup_banner(spec_hash: str, config_hash: str, run_id: str) -> None:
    """Print a startup contract banner with hashes and run ID.
    
    Args:
        spec_hash: SHA-256 hash of the spec bundle
        config_hash: SHA-256 hash of the config
        run_id: Unique run identifier
    """
    print("=" * 80)
    print("STARTUP CONTRACT")
    print("=" * 80)
    print(f"spec_hash:   {spec_hash}")
    print(f"config_hash: {config_hash}")
    print(f"run_id:      {run_id}")
    print("=" * 80)
