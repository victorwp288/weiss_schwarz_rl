from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from weiss_rl.config import load_stack_config
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.learners.impala import ImpalaLearner
from weiss_rl.model import PolicyValueModel

from .snapshot_registry_test_support import (
    REPO_ROOT,
    _load_train_script_module,
    _make_policy_value_model,
)


def test_rollback_to_best_checkpoint_preserves_latest_alias_and_logs_event(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "presets" / "typed_local.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = SimpleNamespace(run_dir=run_dir)
    training_paths = train_script._training_paths(run_dir)
    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()

    learner.update_count = 160
    learner.policy_version = 8
    learner.total_samples_processed = 5120
    best_checkpoint = training_paths.checkpoints_dir / "checkpoint_160.pt"
    train_script._write_checkpoint(
        checkpoint_path=best_checkpoint,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=best_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.5},
        dev_eval_summary={"aggregate_score": 0.65625, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )

    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="policy_000160",
        update=160,
        weights_sha256="8" * 64,
        path="training/snapshots/policy_000160/weights.pt",
    )
    registry.add_snapshot(
        policy_id="policy_000220",
        update=220,
        weights_sha256="1" * 64,
        path="training/snapshots/policy_000220/weights.pt",
    )
    registry.add_champion("policy_000160")
    registry.add_champion("policy_000220")
    registry.save(training_paths.snapshots_dir / "registry.json")

    learner.update_count = 220
    learner.policy_version = 11
    learner.total_samples_processed = 7040
    current_checkpoint = training_paths.checkpoints_dir / "checkpoint_220.pt"
    train_script._write_checkpoint(
        checkpoint_path=current_checkpoint,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=current_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.48, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )
    latest_before_rollback = train_script._load_checkpoint_tracker(training_paths)["latest"]

    runtime = SimpleNamespace(reset_count=0, refresh_count=0, publish_calls=[])
    runtime.reset_outcome_tracker = lambda: setattr(runtime, "reset_count", runtime.reset_count + 1)
    runtime.refresh_opponent_pool = lambda: setattr(runtime, "refresh_count", runtime.refresh_count + 1)

    def _publish_snapshot(**kwargs: object) -> dict[str, float]:
        runtime.publish_calls.append(kwargs)
        return {"snapshot_publish_latency_ms": 1.25, "snapshot_apply_latency_ms": 2.5}

    runtime.maybe_publish_snapshot = _publish_snapshot

    event = train_script._maybe_rollback_to_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        model=cast(PolicyValueModel, learner.model),
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.48, "stall_monitor": {"worst_truncation_rate": 0.0}},
        last_rollback_update=None,
    )

    assert event is not None
    assert event["action"] == "rollback_to_best"
    assert event["reasons"] == ["score_drop"]
    assert event["best_update_count"] == 160
    assert event["demoted_champions"] == ["policy_000220"]
    assert runtime.reset_count == 1
    assert runtime.refresh_count == 1
    assert len(runtime.publish_calls) == 1
    tracker = train_script._load_checkpoint_tracker(training_paths)
    assert tracker["latest"] == latest_before_rollback
    assert tracker["latest"]["metric_kind"] == "dev_eval_mean"
    assert tracker["latest"]["metric_value"] == pytest.approx(0.48)
    assert tracker["latest"]["source_checkpoint_path"].endswith("training/checkpoints/checkpoint_220.pt")
    assert tracker["latest"]["update_count"] == 220
    assert tracker["latest"]["policy_version"] == 11
    assert training_paths.latest_checkpoint_path.read_bytes() != training_paths.best_checkpoint_path.read_bytes()
    assert training_paths.latest_checkpoint_path.read_bytes() == current_checkpoint.read_bytes()
    reloaded = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert reloaded.champion_snapshots == ["policy_000160"]
    log_event = json.loads((training_paths.logs_dir / "checkpoint_guard.jsonl").read_text(encoding="utf-8"))
    assert log_event["action"] == "rollback_to_best"
    assert log_event["rolled_back_checkpoint_path"].endswith("training/checkpoints/best.pt")
    assert log_event["demoted_champions"] == ["policy_000220"]


def test_finalize_from_best_checkpoint_keeps_latest_alias_chronological(tmp_path: Path) -> None:
    train_script = _load_train_script_module()
    stack = load_stack_config(REPO_ROOT / "configs" / "presets" / "typed_local.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = SimpleNamespace(run_dir=run_dir)
    training_paths = train_script._training_paths(run_dir)
    learner = ImpalaLearner(
        model=_make_policy_value_model(stack),
        checkpoint_dir=training_paths.checkpoints_dir,
        logs_dir=training_paths.logs_dir,
        pass_action_id=0,
    )
    learner._optimizer_for_step()

    learner.update_count = 160
    learner.policy_version = 8
    learner.total_samples_processed = 5120
    best_checkpoint = training_paths.checkpoints_dir / "checkpoint_160.pt"
    train_script._write_checkpoint(
        checkpoint_path=best_checkpoint,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=best_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.5},
        dev_eval_summary={"aggregate_score": 0.65625, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )

    learner.update_count = 220
    learner.policy_version = 11
    learner.total_samples_processed = 7040
    current_checkpoint = training_paths.checkpoints_dir / "checkpoint_220.pt"
    train_script._write_checkpoint(
        checkpoint_path=current_checkpoint,
        learner=learner,
        stack=stack,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
    )
    tracker = train_script._publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=current_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.28125, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )
    assert tracker["best"]["source_checkpoint_path"].endswith("training/checkpoints/checkpoint_160.pt")

    runtime = SimpleNamespace(
        reset_count=0,
        refresh_count=0,
    )
    runtime.reset_outcome_tracker = lambda: setattr(runtime, "reset_count", runtime.reset_count + 1)
    runtime.refresh_opponent_pool = lambda: setattr(runtime, "refresh_count", runtime.refresh_count + 1)

    event = train_script._maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        device=torch.device("cpu"),
        spec_hash256="ab" * 32,
        algorithm="impala_vtrace_gru",
        latest_metrics={"loss": 0.1},
        dev_eval_summary={"aggregate_score": 0.28125, "stall_monitor": {"worst_truncation_rate": 0.0}},
    )

    assert event is not None
    assert runtime.reset_count == 1
    assert runtime.refresh_count == 1
    tracker = train_script._load_checkpoint_tracker(training_paths)
    assert tracker["latest"]["metric_kind"] == "dev_eval_mean"
    assert tracker["latest"]["metric_value"] == pytest.approx(0.28125)
    assert tracker["latest"]["source_checkpoint_path"].endswith("training/checkpoints/checkpoint_220.pt")
    assert tracker["latest"]["update_count"] == 220
    assert tracker["latest"]["policy_version"] == 11
    assert training_paths.latest_checkpoint_path.read_bytes() == current_checkpoint.read_bytes()
    assert training_paths.latest_checkpoint_path.read_bytes() != training_paths.best_checkpoint_path.read_bytes()
