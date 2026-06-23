from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.experiments.baselines import (
    NOLEAGUE_BASELINE_NAME,
    NOLEAGUE_BASELINE_POLICY_ID,
    SELECTED_CANDIDATE_POLICY_ID,
)
from weiss_rl.workflows.snapshots import resolve_b1_seed_checkpoint_path

from .workflow_snapshots_test_support import write_checkpoint, write_registry


def test_resolve_b1_seed_checkpoint_path_auto_prefers_canonical_alias(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    canonical_checkpoint = write_checkpoint(run_dir, 7)
    write_checkpoint(run_dir, 8)
    write_checkpoint(run_dir, 9)
    write_registry(
        run_dir,
        {
            "snapshots": [
                {"policy_id": SELECTED_CANDIDATE_POLICY_ID, "update": 9},
                {"policy_id": NOLEAGUE_BASELINE_NAME, "update": 8},
                {"policy_id": NOLEAGUE_BASELINE_POLICY_ID, "update": 7},
            ],
        },
    )

    checkpoint_path, policy_id = resolve_b1_seed_checkpoint_path(
        repo_root=tmp_path,
        run_dir=run_dir,
        init_policy_id="auto",
    )

    assert checkpoint_path == canonical_checkpoint
    assert policy_id == NOLEAGUE_BASELINE_POLICY_ID


def test_resolve_b1_seed_checkpoint_path_auto_falls_back_to_legacy_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    legacy_checkpoint = write_checkpoint(run_dir, 8)
    write_registry(
        run_dir,
        {
            "snapshots": [
                {"policy_id": SELECTED_CANDIDATE_POLICY_ID, "update": 9},
                {"policy_id": NOLEAGUE_BASELINE_NAME, "update": 8},
            ],
        },
    )

    checkpoint_path, policy_id = resolve_b1_seed_checkpoint_path(
        repo_root=tmp_path,
        run_dir=run_dir,
        init_policy_id="",
    )

    assert checkpoint_path == legacy_checkpoint
    assert policy_id == NOLEAGUE_BASELINE_NAME


def test_resolve_b1_seed_checkpoint_path_reports_all_auto_policy_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = write_registry(
        run_dir,
        {"snapshots": [{"policy_id": "policy_000001", "update": 7}]},
    )

    with pytest.raises(SystemExit) as exc_info:
        resolve_b1_seed_checkpoint_path(repo_root=tmp_path, run_dir=run_dir, init_policy_id="auto")

    assert str(exc_info.value) == (
        "Could not resolve a B1 seed checkpoint from --b1-run. "
        "Tried policy ids: "
        f"{NOLEAGUE_BASELINE_POLICY_ID}, {NOLEAGUE_BASELINE_NAME}, {SELECTED_CANDIDATE_POLICY_ID}. "
        f"Last error: snapshot policy id not found in {registry_path}: {SELECTED_CANDIDATE_POLICY_ID}"
    )


def test_resolve_b1_seed_checkpoint_path_uses_explicit_policy_id_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = write_registry(
        run_dir,
        {"snapshots": [{"policy_id": NOLEAGUE_BASELINE_POLICY_ID, "update": 7}]},
    )

    with pytest.raises(SystemExit) as exc_info:
        resolve_b1_seed_checkpoint_path(repo_root=tmp_path, run_dir=run_dir, init_policy_id=" policy_000099 ")

    assert str(exc_info.value) == (
        "Could not resolve a B1 seed checkpoint from --b1-run. "
        "Tried policy ids: policy_000099. "
        f"Last error: snapshot policy id not found in {registry_path}: policy_000099"
    )
