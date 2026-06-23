from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.workflows.snapshots import resolve_single_snapshot_checkpoint_path

from .workflow_snapshots_test_support import write_registry


def test_resolve_single_snapshot_checkpoint_path_uses_its_missing_registry_message(tmp_path: Path) -> None:
    run_dir = Path("runs") / "missing"
    registry_path = tmp_path / run_dir / "training" / "snapshots" / "registry.json"

    with pytest.raises(SystemExit) as exc_info:
        resolve_single_snapshot_checkpoint_path(repo_root=tmp_path, run_dir=run_dir)

    assert str(exc_info.value) == f"smoke profile fallback requires a snapshot registry: {registry_path}"


def test_resolve_single_snapshot_checkpoint_path_requires_exactly_one_policy_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = write_registry(
        run_dir,
        {
            "snapshots": [
                {"policy_id": "policy_000001", "update": 7},
                {"policy_id": "policy_000002", "update": 8},
                {"policy_id": "", "update": 9},
                "ignored",
            ],
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        resolve_single_snapshot_checkpoint_path(repo_root=tmp_path, run_dir=run_dir)

    assert str(exc_info.value) == f"smoke profile fallback requires exactly one snapshot in {registry_path}; found 2"
