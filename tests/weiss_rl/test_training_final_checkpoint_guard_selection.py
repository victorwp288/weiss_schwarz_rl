from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import weiss_rl.training.checkpointing.lifecycle.finalization as checkpoint_finalization

from .final_checkpoint_selection_test_support import CheckpointHookRecorder, guard_event


def test_final_checkpoint_selection_helper_reloads_tracker_only_after_guard_event(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=12)
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    publication = checkpoint_finalization.FinalCheckpointPublication(
        checkpoint_path=tmp_path / "checkpoints" / "checkpoint_12.pt",
        dev_eval_summary={"aggregate_score": 0.3},
        tracker_payload={"latest": {"update": 12}},
    )
    reloaded_tracker = {"best": {"update": 9}}
    guard = guard_event(update_count=12, best_update_count=9, current_score=0.3, best_score=0.7)
    recorder = CheckpointHookRecorder(
        events=events,
        checkpoint_path=publication.checkpoint_path,
        alias_payload=publication.tracker_payload,
        guard_event=guard,
        loaded_tracker=reloaded_tracker,
    )

    selection = checkpoint_finalization.select_final_checkpoint_tracker_payload(
        hooks=recorder.hooks(),
        learner=learner,
        stack=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=training_paths,
        runtime=object(),
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        publication=publication,
    )

    assert selection == checkpoint_finalization.FinalCheckpointSelection(
        source="best_guard_checkpoint",
        tracker_payload=reloaded_tracker,
        guard_event=guard,
    )
    assert [event[0] for event in events] == ["finalize", "load_tracker"]
    assert events[0][1]["dev_eval_summary"] is publication.dev_eval_summary
    assert events[1][1] == {"paths": training_paths}
    stdout = capsys.readouterr().out
    assert "Checkpoint guard final selection: update=12 best_update=9 current_score=0.3000 best_score=0.7000" in stdout
