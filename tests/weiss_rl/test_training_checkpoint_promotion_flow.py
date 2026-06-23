from __future__ import annotations

from pathlib import Path

from .training_checkpoint_promotion_test_support import (
    RecordingRuntime,
    RecordingTensorBoardLogger,
    make_learner,
    recording_hooks,
    run_checkpoint_promotion,
)


def test_checkpoint_promotion_writes_aliases_registry_and_refreshes_on_promotion(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    stack = object()
    device = object()
    algorithm = object()
    latest_metrics = {"loss": 1.0, "league_effective_update": 42.0}
    dev_eval_summary = {"aggregate_score": 0.75}
    learner = make_learner(calls, update_count=6, policy_version=11)
    tracker_payload = {"best": {"update": 6}}

    result = run_checkpoint_promotion(
        tmp_path=tmp_path,
        learner=learner,
        stack=stack,
        runtime=RecordingRuntime(calls),
        device=device,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        last_dev_eval_summary=dev_eval_summary,
        tensorboard_logger=RecordingTensorBoardLogger(calls),
        hooks=recording_hooks(calls, tracker_payload=tracker_payload, promotion_passed=True),
    )

    assert result == tracker_payload
    assert [call[0] for call in calls] == [
        "write",
        "aliases",
        "guard",
        "tensorboard",
        "state_dict",
        "persist",
        "refresh",
        "promotion",
        "refresh",
    ]
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_6.pt"
    assert calls[0][1] == {
        "checkpoint_path": checkpoint_path,
        "learner": learner,
        "stack": stack,
        "device": device,
        "spec_hash256": "spec-hash",
        "algorithm": algorithm,
    }
    assert calls[1][1]["checkpoint_path"] == checkpoint_path
    assert calls[1][1]["latest_metrics"] is latest_metrics
    assert calls[2][1]["dev_eval_summary"] is dev_eval_summary
    assert calls[3][1] == {"payload": tracker_payload, "step": 6}
    assert calls[5][1]["model_state_dict"] == {"weight": 1}
    assert calls[5][1]["policy_version"] == 11
    assert calls[7][1]["candidate_policy_id"] == "candidate_policy"
    assert calls[7][1]["league_reference_update"] == 42
    assert calls[7][1]["policy_version"] == 11
    assert calls[7][1]["run_id256"] == "run-id"
    assert calls[7][1]["config_hash256"] == "config-hash"
    assert calls[7][1]["spec_hash256"] == "spec-hash"


def test_checkpoint_promotion_refreshes_once_when_gate_does_not_promote(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    tracker_payload = {"latest": {"update": 6}}

    result = run_checkpoint_promotion(
        tmp_path=tmp_path,
        learner=make_learner(calls, update_count=6, policy_version=11),
        runtime=RecordingRuntime(calls),
        hooks=recording_hooks(calls, tracker_payload=tracker_payload, promotion_passed=False),
    )

    assert result == tracker_payload
    assert [call[0] for call in calls] == [
        "write",
        "aliases",
        "guard",
        "state_dict",
        "persist",
        "refresh",
        "promotion",
    ]
