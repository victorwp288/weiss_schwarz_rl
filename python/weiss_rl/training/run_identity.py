"""Run identity helpers for new and resumed training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.artifacts.manifest import default_run_dir_name
from weiss_rl.artifacts.reproducibility import compute_run_id64, compute_run_id256


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id256: str
    run_id64: str
    run_dir_name: str


def new_run_identity(
    *,
    spec_hash256: str,
    config_hash256: str,
    git_commit: str,
    start_nonce: int,
    run_label: str,
) -> RunIdentity:
    """Compute the stable identifiers and directory name for a new run."""

    run_id256 = compute_run_id256(spec_hash256, config_hash256, git_commit or None, start_nonce)
    run_id64 = f"{compute_run_id64(spec_hash256, config_hash256, git_commit or None, start_nonce):016x}"
    return RunIdentity(
        run_id256=run_id256,
        run_id64=run_id64,
        run_dir_name=run_label or default_run_dir_name(run_id64),
    )


def resume_run_identity(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    run_dir_name: str,
    expected_spec_hash256: str,
    expected_config_hash256: str,
) -> RunIdentity:
    """Load run identity from a resume manifest and validate immutable hashes."""

    run_id256 = str(manifest.get("run_id256", "")).strip().lower()
    run_id64 = str(manifest.get("run_id64", "")).strip().lower()
    existing_spec_hash = str(manifest.get("spec_hash256", "")).strip().lower()
    existing_config_hash = str(manifest.get("config_hash256", "")).strip().lower()
    if existing_spec_hash != expected_spec_hash256:
        raise RuntimeError(
            f"resume run spec hash mismatch: expected {expected_spec_hash256}, "
            f"found {existing_spec_hash} in {manifest_path}"
        )
    if existing_config_hash != expected_config_hash256:
        raise RuntimeError(
            f"resume run config hash mismatch: expected {expected_config_hash256}, "
            f"found {existing_config_hash} in {manifest_path}"
        )
    return RunIdentity(run_id256=run_id256, run_id64=run_id64, run_dir_name=run_dir_name)
