from __future__ import annotations

import json

import pytest
import weiss_rl.training.checkpointing.guards.structured_guard as checkpoint_structured_guard
import weiss_rl.training.checkpointing.lifecycle.lifecycle as checkpoint_lifecycle
from weiss_rl.training.checkpoints import (
    append_checkpoint_guard_event,
    extract_structured_guard_b2_anchor_score,
    maybe_log_structured_mainmove_guard,
)

from .training_checkpoint_test_support import (
    _Learner,
    _TrainingPaths,
)


def test_append_checkpoint_guard_event_writes_sorted_jsonl(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")

    append_checkpoint_guard_event(paths, {"z": 2, "a": 1})
    append_checkpoint_guard_event(paths, {"event": "second"})

    event_path = paths.logs_dir / "checkpoint_guard.jsonl"
    assert event_path.read_text(encoding="utf-8").splitlines() == [
        '{"a": 1, "z": 2}',
        '{"event": "second"}',
    ]


def test_maybe_log_structured_mainmove_guard_writes_warning_when_learning_is_weak(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    learner = _Learner()

    payload = maybe_log_structured_mainmove_guard(
        training_paths=paths,
        learner=learner,
        latest_metrics={
            "structured_main_move_0_2_top1_rate": 0.4,
            "structured_main_move_share_when_play_available": 0.5,
        },
        dev_eval_summary={
            "aggregate_score": 0.2,
            "anchor_scores": {"B2 HeuristicPublic": 0.0},
        },
    )

    assert payload is not None
    assert payload["event_kind"] == "structured_mainmove_warning_v1"
    assert payload["b2_anchor_score"] == pytest.approx(0.0)
    assert payload["dev_eval_aggregate_score"] == pytest.approx(0.2)
    log_payload = json.loads((paths.logs_dir / "checkpoint_guard.jsonl").read_text(encoding="utf-8"))
    assert log_payload == payload
    assert extract_structured_guard_b2_anchor_score({"anchor_scores": {"B2": 0.125}}) == pytest.approx(0.125)


def test_structured_mainmove_guard_reexports_canonical_payload_boundary() -> None:
    assert checkpoint_lifecycle.extract_structured_guard_b2_anchor_score is (
        checkpoint_structured_guard.extract_structured_guard_b2_anchor_score
    )
    assert checkpoint_structured_guard.structured_mainmove_guard_warning_payload.__module__ == (
        "weiss_rl.training.checkpointing.guards.structured_guard"
    )


def test_structured_mainmove_guard_payload_thresholds_and_fallback_score_gate() -> None:
    learner = _Learner()

    assert (
        checkpoint_structured_guard.structured_mainmove_guard_warning_payload(
            learner=learner,
            latest_metrics=None,
            dev_eval_summary={"aggregate_score": 0.2},
        )
        is None
    )
    assert (
        checkpoint_structured_guard.structured_mainmove_guard_warning_payload(
            learner=learner,
            latest_metrics={
                "structured_main_move_0_2_top1_rate": 0.1,
                "structured_main_move_share_when_play_available": 0.2,
            },
            dev_eval_summary={"aggregate_score": 0.2},
        )
        is None
    )
    assert (
        checkpoint_structured_guard.structured_mainmove_guard_warning_payload(
            learner=learner,
            latest_metrics={
                "structured_main_move_0_2_top1_rate": 0.4,
                "structured_main_move_share_when_play_available": 0.5,
            },
            dev_eval_summary={"aggregate_score": 0.41, "anchor_scores": {"B1": 0.9}},
        )
        is None
    )

    payload = checkpoint_structured_guard.structured_mainmove_guard_warning_payload(
        learner=learner,
        latest_metrics={
            "structured_main_move_0_2_top1_rate": 0.4,
            "structured_main_move_share_when_play_available": 0.5,
        },
        dev_eval_summary={"aggregate_score": 0.2, "anchor_scores": {"B1": 0.9}},
    )

    assert payload is not None
    assert payload["event_kind"] == "structured_mainmove_warning_v1"
    assert payload["dev_eval_aggregate_score"] == pytest.approx(0.2)
    assert payload["b2_anchor_score"] is None
    assert payload["structured_main_move_0_2_top1_rate"] == pytest.approx(0.4)
    assert payload["structured_main_move_share_when_play_available"] == pytest.approx(0.5)


def test_maybe_log_structured_mainmove_guard_suppresses_when_b2_score_is_healthy(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")

    payload = maybe_log_structured_mainmove_guard(
        training_paths=paths,
        learner=_Learner(),
        latest_metrics={
            "structured_main_move_0_2_top1_rate": 0.4,
            "structured_main_move_share_when_play_available": 0.5,
        },
        dev_eval_summary={
            "aggregate_score": 0.2,
            "anchor_scores": {"B2 HeuristicPublic": 0.11},
        },
    )

    assert payload is None
    assert not (paths.logs_dir / "checkpoint_guard.jsonl").exists()
