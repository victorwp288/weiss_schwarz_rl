from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import weiss_rl.training.checkpointing.finalization as checkpoint_finalization

from .final_checkpoint_selection_test_support import CheckpointHookRecorder


def test_final_checkpoint_publication_helper_uses_only_current_dev_eval_summary(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    learner = SimpleNamespace(update_count=12)
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_12.pt"
    current_summary = {"aggregate_score": 0.8}
    recorder = CheckpointHookRecorder(
        events=events,
        checkpoint_path=checkpoint_path,
        alias_payload={"latest": {"update": 12}},
    )

    publication = checkpoint_finalization.publish_final_checkpoint_aliases(
        hooks=recorder.hooks(),
        learner=learner,
        stack=object(),
        artifacts=artifacts,
        training_paths=training_paths,
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary=current_summary,
        last_dev_eval_update_count=12,
        update_count=12,
    )

    assert publication == checkpoint_finalization.FinalCheckpointPublication(
        checkpoint_path=checkpoint_path,
        dev_eval_summary=current_summary,
        tracker_payload={"latest": {"update": 12}},
    )
    assert [event[0] for event in events] == ["ensure", "aliases"]
    assert events[1][1]["dev_eval_summary"] is current_summary

    stale_publication = checkpoint_finalization.publish_final_checkpoint_aliases(
        hooks=recorder.hooks(),
        learner=learner,
        stack=object(),
        artifacts=artifacts,
        training_paths=training_paths,
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary={"aggregate_score": 0.2},
        last_dev_eval_update_count=11,
        update_count=12,
    )

    assert stale_publication.dev_eval_summary is None
    assert events[-1][1]["dev_eval_summary"] is None
