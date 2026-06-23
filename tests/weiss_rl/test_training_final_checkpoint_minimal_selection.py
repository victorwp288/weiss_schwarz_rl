from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from weiss_rl.training.minimal.finalization import _finalize_training_checkpoint_selection

from .final_checkpoint_selection_test_support import (
    CheckpointHookRecorder,
    RecordingTensorBoardLogger,
    guard_event,
)


def test_finalize_training_checkpoint_selection_publishes_current_checkpoint_without_guard_reload(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=9)
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    latest_metrics = {"loss": 1.0}
    dev_eval_summary = {"aggregate_score": 0.5}
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_9.pt"
    recorder = CheckpointHookRecorder(
        events=events,
        checkpoint_path=checkpoint_path,
        alias_payload={"current": {"update": 9}},
        fail_on_load=True,
    )

    tracker_payload = _finalize_training_checkpoint_selection(
        learner=learner,
        stack=object(),
        artifacts=artifacts,
        training_paths=training_paths,
        runtime=object(),
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics=latest_metrics,
        last_dev_eval_summary=dev_eval_summary,
        last_dev_eval_update_count=9,
        tensorboard_logger=RecordingTensorBoardLogger(events),
        hooks=recorder.hooks(),
    )

    assert tracker_payload == {"current": {"update": 9}}
    assert [event[0] for event in events] == ["ensure", "aliases", "finalize", "tensorboard"]
    assert events[0][1]["learner"] is learner
    assert events[0][1]["spec_hash256"] == "spec-hash"
    assert events[1][1]["checkpoint_path"] == checkpoint_path
    assert events[1][1]["latest_metrics"] is latest_metrics
    assert events[1][1]["dev_eval_summary"] is dev_eval_summary
    assert events[2][1]["dev_eval_summary"] is dev_eval_summary
    assert events[3][1] == {"payload": tracker_payload, "step": 9}


def test_finalize_training_checkpoint_selection_reloads_tracker_after_best_finalization(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=10)
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    reloaded_tracker = {"best": {"update": 7}}
    recorder = CheckpointHookRecorder(
        events=events,
        checkpoint_path=tmp_path / "checkpoints" / "checkpoint_10.pt",
        alias_payload={"current": {"update": 10}},
        guard_event=guard_event(update_count=10, best_update_count=7, current_score=0.2, best_score=0.6),
        loaded_tracker=reloaded_tracker,
    )

    tracker_payload = _finalize_training_checkpoint_selection(
        learner=learner,
        stack=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=training_paths,
        runtime=object(),
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary={"aggregate_score": 0.5},
        last_dev_eval_update_count=9,
        tensorboard_logger=RecordingTensorBoardLogger(events),
        hooks=recorder.hooks(),
    )

    assert tracker_payload is reloaded_tracker
    assert [event[0] for event in events] == ["ensure", "aliases", "finalize", "load_tracker", "tensorboard"]
    assert events[1][1]["dev_eval_summary"] is None
    assert events[2][1]["dev_eval_summary"] is None
    assert events[3][1] == {"paths": training_paths}
    assert events[4][1] == {"payload": reloaded_tracker, "step": 10}
    stdout = capsys.readouterr().out
    assert "Checkpoint guard final selection: update=10 best_update=7 current_score=0.2000 best_score=0.6000" in stdout
