from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from weiss_rl.league.registry import SNAPSHOT_METADATA_FILENAME, SnapshotRegistry, snapshot_weights_relpath
from weiss_rl.training.checkpointing.publish import (
    CHECKPOINT_SNAPSHOT_METADATA_FORMAT,
    publish_checkpoint_snapshot,
)
from weiss_rl.training.checkpointing.publish_reporting import checkpoint_publish_output_text
from weiss_rl.training.checkpointing.publish_runtime import run_checkpoint_publish


def test_checkpoint_publish_entrypoint_exposes_only_cli_boundary() -> None:
    import weiss_rl.training.checkpointing.publish_entrypoint as checkpoint_publish_entrypoint

    assert hasattr(checkpoint_publish_entrypoint, "parse_args")
    assert hasattr(checkpoint_publish_entrypoint, "main")
    assert not hasattr(checkpoint_publish_entrypoint, "_build_parser")
    assert not hasattr(checkpoint_publish_entrypoint, "publish_checkpoint_snapshot")
    assert not hasattr(checkpoint_publish_entrypoint, "run_checkpoint_publish")


def test_checkpoint_publish_is_not_a_training_root_alias() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "checkpoint_publish")
    assert not hasattr(training, "checkpoint_publish_cli")
    assert not hasattr(training, "checkpoint_publish_entrypoint")
    assert not hasattr(training, "checkpoint_publish_runtime")


def test_checkpoint_publish_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.training.checkpointing.publish_cli import build_checkpoint_publish_parser

    args = build_checkpoint_publish_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--checkpoint-path",
            str(tmp_path / "run" / "training" / "checkpoints" / "checkpoint_25.pt"),
        ]
    )

    assert args.run_dir == tmp_path / "run"
    assert args.checkpoint_path == tmp_path / "run" / "training" / "checkpoints" / "checkpoint_25.pt"
    assert args.policy_id is None
    assert args.pin is False
    assert args.replace is False


def test_checkpoint_publish_runtime_maps_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import weiss_rl.training.checkpointing.publish_runtime as checkpoint_publish_runtime

    observed: dict[str, object] = {}

    def _fake_publish(**kwargs):
        observed.update(kwargs)
        return {"policy_id": "checkpoint_000025"}

    monkeypatch.setattr(checkpoint_publish_runtime, "publish_checkpoint_snapshot", _fake_publish)
    args = SimpleNamespace(
        run_dir=tmp_path / "run",
        checkpoint_path=tmp_path / "run" / "training" / "checkpoints" / "checkpoint_25.pt",
        policy_id="candidate_u25",
        pin=True,
        replace=True,
    )

    result = run_checkpoint_publish(args)

    assert observed == {
        "run_dir": tmp_path / "run",
        "checkpoint_path": tmp_path / "run" / "training" / "checkpoints" / "checkpoint_25.pt",
        "policy_id": "candidate_u25",
        "pin": True,
        "replace": True,
    }
    assert result.result == {"policy_id": "checkpoint_000025"}


def test_checkpoint_publish_reporting_preserves_pretty_json() -> None:
    assert checkpoint_publish_output_text({"policy_id": "checkpoint_000025", "pinned": False}) == (
        '{\n  "pinned": false,\n  "policy_id": "checkpoint_000025"\n}'
    )


def _write_manifest(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config_canonical": {
                    "config": {
                        "model": {
                            "structured_policy_contract": "factorized_v1",
                        }
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(run_dir: Path, *, filename: str = "checkpoint_25.pt", update: int = 25) -> Path:
    checkpoint_path = run_dir / "training" / "checkpoints" / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Linear(3, 2)
    torch.save(
        {
            "format": "minimal_train_checkpoint_v1",
            "update_count": update,
            "policy_version": 2,
            "device": "cpu",
            "config_hash256": "ab" * 32,
            "spec_hash256": "cd" * 32,
            "model_state_dict": model.state_dict(),
            "public_heuristic_logit_bias_scale": 0.125,
            "public_heuristic_actor_logit_bias_scale": 0.25,
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_publish_checkpoint_snapshot_preserves_latest_as_chronological(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir)
    checkpoint_path = _write_checkpoint(run_dir)

    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000003",
        update=30,
        weights_sha256="3" * 64,
        path=snapshot_weights_relpath("policy_000003"),
    )
    registry.save(registry_path)

    result = publish_checkpoint_snapshot(run_dir=run_dir, checkpoint_path=checkpoint_path)

    assert result["policy_id"] == "checkpoint_000025"
    assert result["update"] == 25
    assert result["source_checkpoint_path"] == "training/checkpoints/checkpoint_25.pt"
    assert result["already_exists"] is False

    reloaded = SnapshotRegistry.load(registry_path)
    assert reloaded.latest_ids(1) == ["policy_000003"]
    assert [snapshot.policy_id for snapshot in reloaded.snapshots] == [
        "checkpoint_000025",
        "policy_000003",
    ]

    metadata_path = run_dir / "training" / "snapshots" / "checkpoint_000025" / SNAPSHOT_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"] == CHECKPOINT_SNAPSHOT_METADATA_FORMAT
    assert metadata["source_checkpoint_update"] == 25
    assert metadata["source_checkpoint_policy_version"] == 2
    assert metadata["source_checkpoint_path"] == "training/checkpoints/checkpoint_25.pt"

    weights_payload = torch.load(
        run_dir / "training" / "snapshots" / "checkpoint_000025" / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert weights_payload["format"] == "minimal_train_snapshot_weights_v1"
    assert weights_payload["policy_id"] == "checkpoint_000025"
    assert weights_payload["structured_policy_contract"] == "factorized_v1"
    assert weights_payload["public_heuristic_logit_bias_scale"] == 0.125
    assert weights_payload["public_heuristic_actor_logit_bias_scale"] == 0.25


def test_publish_checkpoint_snapshot_accepts_repo_relative_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "runs" / "run"
    _write_manifest(run_dir)
    _write_checkpoint(run_dir)
    SnapshotRegistry().save(run_dir / "training" / "snapshots" / "registry.json")
    monkeypatch.chdir(repo_root)

    result = publish_checkpoint_snapshot(
        run_dir=Path("runs/run"),
        checkpoint_path=Path("runs/run/training/checkpoints/checkpoint_25.pt"),
        policy_id="devbest_u25",
    )

    assert result["policy_id"] == "devbest_u25"
    assert result["source_checkpoint_path"] == "training/checkpoints/checkpoint_25.pt"


def test_publish_checkpoint_snapshot_requires_numbered_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir)
    checkpoint_path = _write_checkpoint(run_dir, filename="latest.pt")

    with pytest.raises(ValueError, match="numbered"):
        publish_checkpoint_snapshot(run_dir=run_dir, checkpoint_path=checkpoint_path)
