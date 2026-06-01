from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.experiments.baselines import (
    NOLEAGUE_BASELINE_NAME,
    NOLEAGUE_BASELINE_POLICY_ID,
    SELECTED_CANDIDATE_POLICY_ID,
)
from weiss_rl.workflows.snapshots import (
    resolve_b1_seed_checkpoint_path,
    resolve_single_snapshot_checkpoint_path,
    resolve_snapshot_checkpoint_path,
)


def _write_registry(run_dir: Path, payload: object) -> Path:
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return registry_path


def _write_checkpoint(run_dir: Path, update: int) -> Path:
    checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{update}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    return checkpoint_path


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
    registry_path = _write_registry(run_dir, {"schema_version": 1})

    with pytest.raises(SystemExit) as exc_info:
        resolve_snapshot_checkpoint_path(
            repo_root=tmp_path,
            run_dir=run_dir,
            policy_id="policy_000001",
        )

    assert str(exc_info.value) == f"snapshot registry must contain a snapshots list: {registry_path}"


def test_resolve_snapshot_checkpoint_path_reports_missing_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = _write_registry(
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
    registry_path = _write_registry(
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
    checkpoint_path = _write_checkpoint(run_dir, 11)
    _write_registry(
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
    checkpoint_path = _write_checkpoint(run_dir, 7)
    _write_checkpoint(run_dir, 11)
    _write_registry(
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
    registry_path = _write_registry(run_dir, {"snapshots": [snapshot]})

    with pytest.raises(SystemExit) as exc_info:
        resolve_snapshot_checkpoint_path(
            repo_root=tmp_path,
            run_dir=run_dir,
            policy_id="policy_000001",
        )

    assert str(exc_info.value) == f"snapshot 'policy_000001' is missing an integer update in {registry_path}"


def test_resolve_single_snapshot_checkpoint_path_uses_its_missing_registry_message(tmp_path: Path) -> None:
    run_dir = Path("runs") / "missing"
    registry_path = tmp_path / run_dir / "training" / "snapshots" / "registry.json"

    with pytest.raises(SystemExit) as exc_info:
        resolve_single_snapshot_checkpoint_path(repo_root=tmp_path, run_dir=run_dir)

    assert str(exc_info.value) == f"smoke profile fallback requires a snapshot registry: {registry_path}"


def test_resolve_single_snapshot_checkpoint_path_requires_exactly_one_policy_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    registry_path = _write_registry(
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


def test_resolve_b1_seed_checkpoint_path_auto_prefers_canonical_alias(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    canonical_checkpoint = _write_checkpoint(run_dir, 7)
    _write_checkpoint(run_dir, 8)
    _write_checkpoint(run_dir, 9)
    _write_registry(
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
    legacy_checkpoint = _write_checkpoint(run_dir, 8)
    _write_registry(
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
    registry_path = _write_registry(
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
    registry_path = _write_registry(
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
