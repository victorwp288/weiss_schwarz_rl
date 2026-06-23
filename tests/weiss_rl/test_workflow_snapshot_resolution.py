from __future__ import annotations

from pathlib import Path

import pytest
from weiss_rl.workflows.snapshots import resolve_snapshot_checkpoint_path

from .workflow_snapshots_test_support import write_checkpoint, write_registry


def test_resolve_snapshot_checkpoint_path_reports_missing_registry(tmp_path: Path) -> None:
    run_dir = Path("runs") / "missing"
    registry_path = tmp_path / run_dir / "training" / "snapshots" / "registry.json"

    with pytest.raises(SystemExit) as exc_info:
        resolve_snapshot_checkpoint_path(
            repo_root=tmp_path,
            run_dir=run_dir,
            policy_id="policy_000001",
        )

    assert str(exc_info.value) == f"--init-from-run-dir snapshot registry not found: {registry_path}"


def test_resolve_snapshot_checkpoint_path_requires_snapshots_list(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = write_registry(run_dir, {"schema_version": 1})

    with pytest.raises(SystemExit) as exc_info:
        resolve_snapshot_checkpoint_path(
            repo_root=tmp_path,
            run_dir=run_dir,
            policy_id="policy_000001",
        )

    assert str(exc_info.value) == f"snapshot registry must contain a snapshots list: {registry_path}"


def test_resolve_snapshot_checkpoint_path_reports_missing_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = write_registry(
        run_dir,
        {"snapshots": [{"policy_id": "policy_000001", "update": 7}]},
    )
    checkpoint_path = run_dir / "training" / "checkpoints" / "checkpoint_7.pt"

    with pytest.raises(SystemExit) as exc_info:
        resolve_snapshot_checkpoint_path(
            repo_root=tmp_path,
            run_dir=run_dir,
            policy_id="policy_000001",
        )

    assert str(exc_info.value) == (
        f"checkpoint for snapshot 'policy_000001' was not found: {checkpoint_path}. "
        "Use --init-from-checkpoint if the source checkpoint was moved."
    )
    assert registry_path.is_file()


def test_resolve_snapshot_checkpoint_path_reports_missing_policy_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = write_registry(
        run_dir,
        {"snapshots": [{"policy_id": "policy_000002", "update": 7}]},
    )

    with pytest.raises(SystemExit) as exc_info:
        resolve_snapshot_checkpoint_path(
            repo_root=tmp_path,
            run_dir=run_dir,
            policy_id="policy_000001",
        )

    assert str(exc_info.value) == f"snapshot policy id not found in {registry_path}: policy_000001"


def test_resolve_snapshot_checkpoint_path_accepts_update_count_compatibility_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    checkpoint_path = write_checkpoint(run_dir, 11)
    write_registry(
        run_dir,
        {"snapshots": [{"policy_id": "policy_000001", "update_count": 11}]},
    )

    resolved = resolve_snapshot_checkpoint_path(
        repo_root=tmp_path,
        run_dir=run_dir,
        policy_id="policy_000001",
    )

    assert resolved == checkpoint_path


def test_resolve_snapshot_checkpoint_path_prefers_update_over_update_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    checkpoint_path = write_checkpoint(run_dir, 7)
    write_checkpoint(run_dir, 11)
    write_registry(
        run_dir,
        {"snapshots": [{"policy_id": "policy_000001", "update": 7, "update_count": 11}]},
    )

    resolved = resolve_snapshot_checkpoint_path(
        repo_root=tmp_path,
        run_dir=run_dir,
        policy_id="policy_000001",
    )

    assert resolved == checkpoint_path


@pytest.mark.parametrize("snapshot", [{"policy_id": "policy_000001"}, {"policy_id": "policy_000001", "update": "7"}])
def test_resolve_snapshot_checkpoint_path_reports_missing_integer_update(
    tmp_path: Path,
    snapshot: dict[str, object],
) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = write_registry(run_dir, {"snapshots": [snapshot]})

    with pytest.raises(SystemExit) as exc_info:
        resolve_snapshot_checkpoint_path(
            repo_root=tmp_path,
            run_dir=run_dir,
            policy_id="policy_000001",
        )

    assert str(exc_info.value) == f"snapshot 'policy_000001' is missing an integer update in {registry_path}"
