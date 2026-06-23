from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from weiss_rl.league.registry import SNAPSHOT_METADATA_FILENAME, SnapshotRegistry
from weiss_rl.training.checkpointing.interpolation import interpolate_model_state_dicts
from weiss_rl.training.checkpointing.interpolation_reporting import checkpoint_interpolation_output_line
from weiss_rl.training.checkpointing.interpolation_runtime import (
    CheckpointInterpolationRunResult,
    run_checkpoint_interpolation,
)


def test_checkpoint_interpolation_entrypoint_exposes_only_cli_boundary() -> None:
    import weiss_rl.training.checkpointing.interpolation_entrypoint as checkpoint_interpolation_entrypoint

    assert hasattr(checkpoint_interpolation_entrypoint, "parse_args")
    assert hasattr(checkpoint_interpolation_entrypoint, "main")
    assert not hasattr(checkpoint_interpolation_entrypoint, "_build_parser")
    assert not hasattr(checkpoint_interpolation_entrypoint, "interpolate_model_state_dicts")
    assert not hasattr(checkpoint_interpolation_entrypoint, "run_checkpoint_interpolation")
    assert not hasattr(checkpoint_interpolation_entrypoint, "_load_checkpoint")
    assert not hasattr(checkpoint_interpolation_entrypoint, "_publish_snapshot")


def test_checkpoint_interpolation_is_not_a_training_root_alias() -> None:
    import weiss_rl.training as training

    assert not hasattr(training, "checkpoint_interpolation")
    assert not hasattr(training, "checkpoint_interpolation_cli")
    assert not hasattr(training, "checkpoint_interpolation_entrypoint")
    assert not hasattr(training, "checkpoint_interpolation_runtime")


def test_checkpoint_interpolation_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.training.checkpointing.interpolation_cli import build_checkpoint_interpolation_parser

    args = build_checkpoint_interpolation_parser().parse_args(
        [
            "--first-checkpoint",
            str(tmp_path / "first.pt"),
            "--second-checkpoint",
            str(tmp_path / "second.pt"),
            "--first-run-dir",
            str(tmp_path / "first_run"),
            "--second-run-dir",
            str(tmp_path / "second_run"),
            "--second-weight",
            "0.25",
            "--output-run-dir",
            str(tmp_path / "mixed"),
        ]
    )

    assert args.first_checkpoint == tmp_path / "first.pt"
    assert args.second_checkpoint == tmp_path / "second.pt"
    assert args.first_run_dir == tmp_path / "first_run"
    assert args.second_run_dir == tmp_path / "second_run"
    assert args.second_weight == 0.25
    assert args.output_run_dir == tmp_path / "mixed"
    assert args.policy_id == "trajectory_bc_latest"
    assert args.allow_config_hash_mismatch is False


def test_checkpoint_interpolation_reporting_preserves_console_line(tmp_path: Path) -> None:
    assert checkpoint_interpolation_output_line(
        checkpoint_path=tmp_path / "mixed" / "training" / "checkpoints" / "checkpoint_interpolated.pt",
        summary_path=tmp_path / "mixed" / "eval" / "diagnostics" / "checkpoint_interpolation_summary.json",
        second_weight=0.25,
    ) == (
        f"Interpolated checkpoint written to "
        f"{tmp_path / 'mixed' / 'training' / 'checkpoints' / 'checkpoint_interpolated.pt'} "
        f"with second_weight=0.250; summary written to "
        f"{tmp_path / 'mixed' / 'eval' / 'diagnostics' / 'checkpoint_interpolation_summary.json'}"
    )


def test_checkpoint_interpolation_runtime_writes_checkpoint_summary_and_snapshot(tmp_path: Path) -> None:
    first_run = tmp_path / "first_run"
    second_run = tmp_path / "second_run"
    output_run = tmp_path / "mixed_run"
    first_checkpoint = _write_interpolation_checkpoint(first_run, value=1.0)
    second_checkpoint = _write_interpolation_checkpoint(second_run, value=5.0)
    (second_run / "manifest.json").write_text(json.dumps({"source": "second"}, sort_keys=True), encoding="utf-8")

    result = run_checkpoint_interpolation(
        SimpleNamespace(
            first_checkpoint=first_checkpoint,
            second_checkpoint=second_checkpoint,
            first_run_dir=first_run,
            second_run_dir=second_run,
            second_weight=0.25,
            output_run_dir=output_run,
            policy_id="mixed_policy",
            allow_config_hash_mismatch=False,
        )
    )

    assert isinstance(result, CheckpointInterpolationRunResult)
    assert result.checkpoint_path == output_run / "training" / "checkpoints" / "checkpoint_interpolated.pt"
    assert result.summary_path == output_run / "eval" / "diagnostics" / "checkpoint_interpolation_summary.json"
    assert result.summary["format"] == "checkpoint_interpolation_summary_v1"
    assert result.summary["snapshot"]["policy_id"] == "mixed_policy"
    assert result.summary_path.is_file()
    assert (output_run / "training" / "checkpoints" / "latest.pt").is_file()

    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["model_state_dict"]["weight"].tolist() == pytest.approx([2.0])
    assert payload["policy_anchor_model_state_dict"] is None
    assert payload["optimizer_state_dict"] is None
    assert payload["grad_scaler_state_dict"] is None
    assert payload["interpolation"]["second_weight"] == 0.25

    registry = SnapshotRegistry.load(output_run / "training" / "snapshots" / "registry.json")
    assert registry.pinned_snapshots == ["mixed_policy"]
    metadata_path = output_run / "training" / "snapshots" / "mixed_policy" / SNAPSHOT_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "interpolated_snapshot_meta_v1"
    assert metadata["policy_id"] == "mixed_policy"


def test_interpolate_model_state_dicts_lerps_float_tensors_and_copies_equal_int_tensors() -> None:
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor([7], dtype=torch.int64),
    }
    second = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor([7], dtype=torch.int64),
    }

    mixed = interpolate_model_state_dicts(first, second, second_weight=0.25)

    assert mixed["weight"].tolist() == pytest.approx([2.0, 4.0])
    assert mixed["counter"].tolist() == [7]
    assert mixed["counter"] is not first["counter"]


def test_interpolate_model_state_dicts_rejects_incompatible_keys() -> None:
    with pytest.raises(ValueError, match="state dict keys do not match"):
        interpolate_model_state_dicts(
            {"left": torch.tensor([1.0])},
            {"right": torch.tensor([1.0])},
            second_weight=0.5,
        )


def test_interpolate_model_state_dicts_rejects_changed_nonfloating_tensor() -> None:
    with pytest.raises(ValueError, match="non-floating tensor differs"):
        interpolate_model_state_dicts(
            {"counter": torch.tensor([1], dtype=torch.int64)},
            {"counter": torch.tensor([2], dtype=torch.int64)},
            second_weight=0.5,
        )


def _write_interpolation_checkpoint(run_dir: Path, *, value: float) -> Path:
    checkpoint_path = run_dir / "training" / "checkpoints" / "checkpoint_10.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "minimal_train_checkpoint_v1",
            "config_hash256": "aa" * 32,
            "spec_hash256": "bb" * 32,
            "algorithm": "impala",
            "recurrent_core": "lstm",
            "update_count": 10,
            "model_state_dict": {
                "weight": torch.tensor([value], dtype=torch.float32),
                "counter": torch.tensor([7], dtype=torch.int64),
            },
            "policy_anchor_model_state_dict": {"old": torch.tensor([1.0])},
            "optimizer_state_dict": {"old": 1},
            "grad_scaler_state_dict": {"old": 2},
        },
        checkpoint_path,
    )
    return checkpoint_path
