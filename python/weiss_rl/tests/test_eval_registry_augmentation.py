from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.eval_registry_augmentation import augment_eval_snapshot_registry
from weiss_rl.league.registry import SNAPSHOT_METADATA_FILENAME, SnapshotRegistry, snapshot_weights_relpath


def _write_snapshot(run_dir: Path, registry: SnapshotRegistry, policy_id: str, *, update: int = 0) -> None:
    weights_path = run_dir / snapshot_weights_relpath(policy_id)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(f"weights:{policy_id}".encode())
    (weights_path.parent / SNAPSHOT_METADATA_FILENAME).write_text(
        json.dumps({"policy_id": policy_id}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    registry.add_snapshot(
        policy_id=policy_id,
        update=update,
        weights_sha256="",
        path=snapshot_weights_relpath(policy_id),
    )


def test_augment_eval_snapshot_registry_copies_source_champions_without_replacing_training_registry(
    tmp_path: Path,
) -> None:
    target_run = tmp_path / "runs" / "target"
    source_run = tmp_path / "runs" / "source"
    target_registry = SnapshotRegistry()
    _write_snapshot(target_run, target_registry, "candidate", update=5)
    target_registry.pin_snapshot("candidate")
    target_registry.save(target_run / "training" / "snapshots" / "registry.json")
    source_registry = SnapshotRegistry(champion_size=2)
    _write_snapshot(source_run, source_registry, "seed_a")
    _write_snapshot(source_run, source_registry, "seed_b")
    source_registry.add_champion("seed_a")
    source_registry.add_champion("seed_b")
    source_registry.save(source_run / "training" / "snapshots" / "registry.json")

    summary = augment_eval_snapshot_registry(
        target_run_dir=target_run,
        source_registry_json=source_run / "training" / "snapshots" / "registry.json",
        include_source_champions=True,
    )

    original = SnapshotRegistry.load(target_run / "training" / "snapshots" / "registry.json")
    augmented = SnapshotRegistry.load(target_run / "training" / "snapshots" / "registry_with_imported_champions.json")
    assert original.pinned_snapshots == ["candidate"]
    assert original.champion_snapshots == []
    assert [snapshot.policy_id for snapshot in augmented.snapshots] == ["seed_a", "seed_b", "candidate"]
    assert augmented.champion_snapshots == ["seed_a", "seed_b"]
    assert (target_run / "training" / "snapshots" / "seed_a" / "weights.pt").read_bytes() == b"weights:seed_a"
    assert summary["copied_policy_ids"] == ["seed_a", "seed_b"]
    assert Path(summary["summary_json"]).is_file()


def test_augment_eval_snapshot_registry_reports_missing_requested_policy(tmp_path: Path) -> None:
    target_run = tmp_path / "runs" / "target"
    source_run = tmp_path / "runs" / "source"
    SnapshotRegistry().save(target_run / "training" / "snapshots" / "registry.json")
    SnapshotRegistry().save(source_run / "training" / "snapshots" / "registry.json")

    summary = augment_eval_snapshot_registry(
        target_run_dir=target_run,
        source_registry_json=source_run / "training" / "snapshots" / "registry.json",
        include_policy_ids=("missing",),
    )

    assert summary["missing_policy_ids"] == ["missing"]
    augmented = SnapshotRegistry.load(target_run / "training" / "snapshots" / "registry_with_imported_champions.json")
    assert augmented.snapshots == []
