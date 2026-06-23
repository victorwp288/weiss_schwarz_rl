from __future__ import annotations

import json
from pathlib import Path

import torch
from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath

from .snapshot_registry_test_support import (
    _load_train_script_module,
    _retention_stack,
)


def test_train_snapshot_persistence_writes_artifact_bundle_and_registry_entry(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = _retention_stack(recent_size=24, champion_size=4)
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    checkpoint_path = training_paths.checkpoints_dir / "checkpoint_7.pt"
    torch.save({"format": "checkpoint_stub"}, checkpoint_path)

    model = torch.nn.Linear(3, 2)
    train_script._persist_snapshot_registry_entry(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        model_state_dict=model.state_dict(),
        config_hash256="ab" * 32,
        device=torch.device("cpu"),
        update=7,
        policy_version=7,
    )

    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert len(registry.snapshots) == 1

    snapshot = registry.snapshots[0]
    expected_weights_relpath = snapshot_weights_relpath("policy_000007")
    weights_path = run_dir / expected_weights_relpath
    metadata_path = training_paths.snapshots_dir / "policy_000007" / "policy_meta.json"

    assert snapshot.policy_id == "policy_000007"
    assert snapshot.update == 7
    assert snapshot.path == expected_weights_relpath
    assert snapshot.path != checkpoint_path.relative_to(run_dir).as_posix()
    assert weights_path.is_file()
    assert snapshot.weights_sha256 == train_script._sha256_file(weights_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "format": "minimal_train_snapshot_metadata_v1",
        "policy_id": "policy_000007",
        "source_checkpoint_path": "training/checkpoints/checkpoint_7.pt",
        "update": 7,
        "weights_path": expected_weights_relpath,
        "weights_sha256": snapshot.weights_sha256,
    }

    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert payload["format"] == "minimal_train_snapshot_weights_v1"
    assert payload["policy_id"] == "policy_000007"
    assert payload["update"] == 7
    assert payload["config_hash256"] == "ab" * 32
    assert payload["device"] == "cpu"
    assert set(payload["model_state_dict"]) == set(model.state_dict())


def test_train_snapshot_retention_prunes_old_snapshot_artifacts(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = _retention_stack(recent_size=1, champion_size=0)
    run_dir = tmp_path / "run"
    training_paths = train_script._training_paths(run_dir)
    model = torch.nn.Linear(3, 2)

    for policy_version in (1, 2, 3):
        checkpoint_path = training_paths.checkpoints_dir / f"checkpoint_{policy_version}.pt"
        torch.save({"format": "checkpoint_stub"}, checkpoint_path)
        train_script._persist_snapshot_registry_entry(
            stack=stack,
            training_paths=training_paths,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            model_state_dict=model.state_dict(),
            config_hash256="ab" * 32,
            device=torch.device("cpu"),
            update=policy_version,
            policy_version=policy_version,
        )

    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")

    assert [snapshot.policy_id for snapshot in registry.snapshots] == ["policy_000003"]
    assert not (training_paths.snapshots_dir / "policy_000001").exists()
    assert not (training_paths.snapshots_dir / "policy_000002").exists()
    assert (training_paths.snapshots_dir / "policy_000003").is_dir()
