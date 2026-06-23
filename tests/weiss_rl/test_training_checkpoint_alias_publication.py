from __future__ import annotations

from pathlib import Path

import pytest
import weiss_rl.training.checkpointing.alias_publication as checkpoint_alias_publication
import weiss_rl.training.checkpointing.aliases as checkpoint_aliases
import weiss_rl.training.checkpointing.io as checkpoint_io
from weiss_rl.training.checkpointing.alias_mutation import (
    CheckpointAliasMutation,
    alias_record_for_mutation,
    apply_checkpoint_alias_mutation,
)
from weiss_rl.training.checkpoints import (
    OBSERVED_BEST_CHECKPOINT_FILENAME,
)

from .training_checkpoint_test_support import (
    _Learner,
    _TrainingPaths,
)


def test_checkpoints_reexports_canonical_checkpoint_io_boundary() -> None:
    from weiss_rl.training import checkpoints

    assert checkpoints.write_scalars_record is checkpoint_io.write_scalars_record
    assert checkpoints.current_focal_policy_id is checkpoint_io.current_focal_policy_id
    assert checkpoints.checkpoint_path_for_update is checkpoint_io.checkpoint_path_for_update
    assert checkpoints.ensure_current_checkpoint is checkpoint_io.ensure_current_checkpoint
    assert checkpoints.write_minimal_train_checkpoint is checkpoint_io.write_minimal_train_checkpoint
    assert checkpoints.restore_minimal_train_checkpoint is checkpoint_io.restore_minimal_train_checkpoint
    assert checkpoints.initialize_model_from_checkpoint is checkpoint_io.initialize_model_from_checkpoint
    assert checkpoint_io.write_minimal_train_checkpoint.__module__ == "weiss_rl.training.checkpointing.io"


def test_checkpoint_alias_mutation_copies_checkpoint_and_updates_tracker(tmp_path: Path) -> None:
    run_dir = tmp_path
    checkpoint_path = tmp_path / "checkpoint_25.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    alias_path = tmp_path / "training" / "checkpoints" / "latest.pt"
    alias_path.parent.mkdir(parents=True)
    candidate = checkpoint_aliases.CheckpointAliasCandidate(
        metric_kind="dev_eval_mean",
        metric_value=0.75,
        observed_score=0.8,
        dev_eval_candidate={"score": 0.8, "eligible_for_best": True},
    )
    learner = _Learner()
    tracker: dict[str, object] = {}

    record = apply_checkpoint_alias_mutation(
        tracker=tracker,
        mutation=CheckpointAliasMutation(
            alias_name="latest",
            alias_path=alias_path,
            source_checkpoint_path=checkpoint_path,
            metric_kind=candidate.metric_kind,
            metric_value=candidate.metric_value,
            include_dev_eval_candidate=True,
        ),
        run_dir=run_dir,
        learner=learner,
        candidate=candidate,
    )

    assert alias_path.read_bytes() == b"checkpoint"
    assert tracker["latest"] is record
    assert record == {
        "alias": "latest",
        "alias_path": "training/checkpoints/latest.pt",
        "source_checkpoint_path": "checkpoint_25.pt",
        "update_count": 3,
        "policy_version": 7,
        "metric_kind": "dev_eval_mean",
        "metric_value": 0.75,
        "dev_eval_candidate": {"score": 0.8, "eligible_for_best": True},
    }


def test_checkpoint_alias_mutation_record_omits_dev_eval_candidate_when_requested(tmp_path: Path) -> None:
    candidate = checkpoint_aliases.CheckpointAliasCandidate(
        metric_kind="dev_eval_mean",
        metric_value=0.75,
        observed_score=0.8,
        dev_eval_candidate={"score": 0.8},
    )

    record = alias_record_for_mutation(
        mutation=CheckpointAliasMutation(
            alias_name="best",
            alias_path=tmp_path / "training" / "checkpoints" / "best.pt",
            source_checkpoint_path=tmp_path / "checkpoint_25.pt",
            metric_kind=candidate.metric_kind,
            metric_value=candidate.metric_value,
            include_dev_eval_candidate=False,
        ),
        run_dir=tmp_path,
        learner=_Learner(),
        candidate=candidate,
    )

    assert record["alias"] == "best"
    assert record["metric_kind"] == "dev_eval_mean"
    assert "dev_eval_candidate" not in record


def test_checkpoint_alias_publication_reports_records_and_keeps_chronological_latest(tmp_path: Path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoint_tracker_path.parent.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint_3.pt"
    checkpoint.write_bytes(b"checkpoint")
    tracker = checkpoint_aliases.default_checkpoint_tracker_payload()
    candidate = checkpoint_aliases.CheckpointAliasCandidate(
        metric_kind="training_loss",
        metric_value=1.25,
        observed_score=None,
        dev_eval_candidate=None,
    )

    publication = checkpoint_aliases.apply_checkpoint_alias_publication(
        tracker=tracker,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=checkpoint,
        learner=_Learner(),
        candidate=candidate,
    )

    assert publication.tracker is tracker
    assert publication.candidate is candidate
    assert publication.latest_record is tracker["latest"]
    assert publication.best_record is tracker["best"]
    assert publication.observed_best_record is None
    assert tracker["observed_best"] is None
    assert publication.latest_record["alias"] == "latest"
    assert publication.latest_record["source_checkpoint_path"] == "checkpoint_3.pt"
    assert publication.best_record["alias"] == "best"
    assert paths.latest_checkpoint_path.read_bytes() == b"checkpoint"
    assert paths.best_checkpoint_path.read_bytes() == b"checkpoint"


def test_checkpoint_alias_publication_reports_skipped_observed_best_without_mutating_alias(tmp_path: Path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoint_tracker_path.parent.mkdir(parents=True)
    observed_best_path = paths.checkpoint_tracker_path.parent / OBSERVED_BEST_CHECKPOINT_FILENAME
    observed_best_path.write_bytes(b"existing-observed")
    checkpoint = tmp_path / "checkpoint_4.pt"
    checkpoint.write_bytes(b"new-checkpoint")
    tracker = {
        "format": "checkpoint_tracker_v1",
        "latest": None,
        "best": None,
        "observed_best": {"metric_value": 0.75, "source_checkpoint_path": "old.pt"},
    }
    candidate = checkpoint_aliases.CheckpointAliasCandidate(
        metric_kind=None,
        metric_value=None,
        observed_score=0.5,
        dev_eval_candidate={"score": 0.5, "eligible_for_best": False},
    )

    publication = checkpoint_aliases.apply_checkpoint_alias_publication(
        tracker=tracker,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=checkpoint,
        learner=_Learner(),
        candidate=candidate,
    )

    assert publication.observed_best_record is None
    assert publication.best_record is None
    assert tracker["observed_best"] == {"metric_value": 0.75, "source_checkpoint_path": "old.pt"}
    assert observed_best_path.read_bytes() == b"existing-observed"
    assert publication.latest_record["dev_eval_candidate"] == {"score": 0.5, "eligible_for_best": False}


def test_checkpoint_alias_publication_applies_latest_observed_and_best_mutations_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    tracker = checkpoint_aliases.default_checkpoint_tracker_payload()
    checkpoint_path = tmp_path / "checkpoint_25.pt"
    candidate = checkpoint_aliases.CheckpointAliasCandidate(
        metric_kind="dev_eval_mean",
        metric_value=0.7,
        observed_score=0.8,
        dev_eval_candidate={"score": 0.8, "eligible_for_best": True},
    )
    calls: list[tuple[str, Path, str | None, float | None, bool]] = []

    def fake_apply_checkpoint_alias_mutation(**kwargs: object) -> dict[str, object]:
        mutation = kwargs["mutation"]
        assert isinstance(mutation, CheckpointAliasMutation)
        calls.append(
            (
                mutation.alias_name,
                mutation.alias_path,
                mutation.metric_kind,
                mutation.metric_value,
                mutation.include_dev_eval_candidate,
            )
        )
        record: dict[str, object] = {
            "alias": mutation.alias_name,
            "metric_kind": mutation.metric_kind,
            "metric_value": mutation.metric_value,
        }
        tracker_arg = kwargs["tracker"]
        assert isinstance(tracker_arg, dict)
        tracker_arg[mutation.alias_name] = record
        return record

    monkeypatch.setattr(
        checkpoint_alias_publication,
        "apply_checkpoint_alias_mutation",
        fake_apply_checkpoint_alias_mutation,
    )

    publication = checkpoint_alias_publication.apply_checkpoint_alias_publication(
        tracker=tracker,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=checkpoint_path,
        learner=_Learner(),
        candidate=candidate,
    )

    assert calls == [
        ("latest", paths.latest_checkpoint_path, "dev_eval_mean", 0.7, True),
        (
            "observed_best",
            paths.checkpoint_tracker_path.parent / OBSERVED_BEST_CHECKPOINT_FILENAME,
            "dev_eval_observed_mean",
            0.8,
            True,
        ),
        ("best", paths.best_checkpoint_path, "dev_eval_mean", 0.7, False),
    ]
    assert publication.latest_record is tracker["latest"]
    assert publication.observed_best_record is tracker["observed_best"]
    assert publication.best_record is tracker["best"]
