from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from weiss_rl.training.checkpoints import (
    OBSERVED_BEST_CHECKPOINT_FILENAME,
    best_checkpoint_record,
    publish_checkpoint_aliases,
)

from .training_checkpoint_test_support import (
    _Learner,
    _TrainingPaths,
)


def _checkpoint_guard_stack() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            evaluation=SimpleNamespace(periodic_dev_eval_interval_updates=25),
            curriculum=SimpleNamespace(
                stall_monitor=SimpleNamespace(enabled=False, truncation_rate_threshold=0.25),
                checkpoint_guard=SimpleNamespace(
                    enabled=True,
                    promote_min_prob_gt_half=0.60,
                    promote_max_ci_half_width=0.24,
                ),
            ),
        )
    )


def _dev_eval_summary(*, score: float, prob_gt_half: float) -> dict[str, object]:
    return {
        "aggregate_score": score,
        "anchors": {
            "B2 HeuristicPublic": {
                "summary": {
                    "games": 32,
                    "truncations": 0,
                    "no_progress_timeouts": 0,
                    "natural_timeouts": 0,
                },
                "uncertainty": {
                    "prob_gt_half": prob_gt_half,
                    "prob_lt_half": 1.0 - prob_gt_half,
                    "ci_half_width": 0.12,
                },
            }
        },
    }


def test_publish_checkpoint_aliases_updates_latest_and_promotes_lower_training_loss(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoint_tracker_path.parent.mkdir(parents=True)
    stack = SimpleNamespace(config=SimpleNamespace(evaluation=None))
    learner = _Learner()
    run_dir = tmp_path

    checkpoint_a = tmp_path / "checkpoint_a.pt"
    checkpoint_a.write_bytes(b"checkpoint-a")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_a,
        learner=learner,
        latest_metrics={"loss": 1.5},
    )

    assert paths.latest_checkpoint_path.read_bytes() == b"checkpoint-a"
    assert paths.best_checkpoint_path.read_bytes() == b"checkpoint-a"
    assert tracker["latest"]["metric_kind"] == "training_loss"
    assert tracker["latest"]["metric_value"] == pytest.approx(1.5)
    assert tracker["best"]["source_checkpoint_path"] == "checkpoint_a.pt"

    learner.update_count = 4
    checkpoint_b = tmp_path / "checkpoint_b.pt"
    checkpoint_b.write_bytes(b"checkpoint-b")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_b,
        learner=learner,
        latest_metrics={"loss": 2.0},
    )

    assert paths.latest_checkpoint_path.read_bytes() == b"checkpoint-b"
    assert paths.best_checkpoint_path.read_bytes() == b"checkpoint-a"
    assert tracker["latest"]["source_checkpoint_path"] == "checkpoint_b.pt"
    assert tracker["best"]["source_checkpoint_path"] == "checkpoint_a.pt"

    learner.update_count = 5
    checkpoint_c = tmp_path / "checkpoint_c.pt"
    checkpoint_c.write_bytes(b"checkpoint-c")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_c,
        learner=learner,
        latest_metrics={"loss": 1.0},
    )

    assert paths.latest_checkpoint_path.read_bytes() == b"checkpoint-c"
    assert paths.best_checkpoint_path.read_bytes() == b"checkpoint-c"
    assert tracker["best"]["source_checkpoint_path"] == "checkpoint_c.pt"
    assert json.loads(paths.checkpoint_tracker_path.read_text(encoding="utf-8")) == tracker
    assert best_checkpoint_record(paths) == tracker["best"]


def test_publish_checkpoint_aliases_records_dev_eval_ineligibility_on_latest(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoint_tracker_path.parent.mkdir(parents=True)
    stack = _checkpoint_guard_stack()
    learner = _Learner()
    checkpoint = tmp_path / "checkpoint_25.pt"
    checkpoint.write_bytes(b"checkpoint")

    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.5},
        dev_eval_summary=_dev_eval_summary(score=0.62, prob_gt_half=0.58),
    )

    assert tracker["latest"]["metric_kind"] is None
    assert tracker["best"] is None
    assert tracker["observed_best"]["alias"] == "observed_best"
    assert tracker["observed_best"]["metric_kind"] == "dev_eval_observed_mean"
    assert tracker["observed_best"]["metric_value"] == pytest.approx(0.62)
    assert tracker["observed_best"]["source_checkpoint_path"] == "checkpoint_25.pt"
    observed_best_path = paths.checkpoint_tracker_path.parent / OBSERVED_BEST_CHECKPOINT_FILENAME
    assert observed_best_path.read_bytes() == b"checkpoint"
    candidate = tracker["latest"]["dev_eval_candidate"]
    assert candidate["score"] == pytest.approx(0.62)
    assert candidate["eligible_for_best"] is False
    assert candidate["ineligibility_reasons"] == ["confidence_prob"]
    assert candidate["confidence"]["min_prob_gt_half"] == pytest.approx(0.58)
    assert tracker["observed_best"]["dev_eval_candidate"] == candidate

    learner.update_count = 50
    lower_checkpoint = tmp_path / "checkpoint_50.pt"
    lower_checkpoint.write_bytes(b"lower-checkpoint")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=lower_checkpoint,
        learner=learner,
        latest_metrics={"loss": 0.25},
        dev_eval_summary=_dev_eval_summary(score=0.59, prob_gt_half=0.57),
    )

    assert tracker["latest"]["source_checkpoint_path"] == "checkpoint_50.pt"
    assert tracker["observed_best"]["source_checkpoint_path"] == "checkpoint_25.pt"
    assert observed_best_path.read_bytes() == b"checkpoint"


def test_publish_checkpoint_aliases_separates_observed_best_from_guarded_best(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoint_tracker_path.parent.mkdir(parents=True)
    stack = _checkpoint_guard_stack()
    learner = _Learner()
    checkpoint_25 = tmp_path / "checkpoint_25.pt"
    checkpoint_25.write_bytes(b"eligible")

    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=checkpoint_25,
        learner=learner,
        latest_metrics={"loss": 0.5},
        dev_eval_summary=_dev_eval_summary(score=0.61, prob_gt_half=0.85),
    )

    assert tracker["best"]["source_checkpoint_path"] == "checkpoint_25.pt"
    assert tracker["observed_best"]["source_checkpoint_path"] == "checkpoint_25.pt"

    learner.update_count = 50
    checkpoint_50 = tmp_path / "checkpoint_50.pt"
    checkpoint_50.write_bytes(b"higher-but-ineligible")
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=tmp_path,
        checkpoint_path=checkpoint_50,
        learner=learner,
        latest_metrics={"loss": 0.25},
        dev_eval_summary=_dev_eval_summary(score=0.64, prob_gt_half=0.55),
    )

    assert paths.best_checkpoint_path.read_bytes() == b"eligible"
    assert tracker["best"]["metric_kind"] == "dev_eval_mean"
    assert tracker["best"]["metric_value"] == pytest.approx(0.61)
    assert tracker["best"]["source_checkpoint_path"] == "checkpoint_25.pt"
    assert tracker["observed_best"]["metric_kind"] == "dev_eval_observed_mean"
    assert tracker["observed_best"]["metric_value"] == pytest.approx(0.64)
    assert tracker["observed_best"]["source_checkpoint_path"] == "checkpoint_50.pt"
    assert (paths.checkpoint_tracker_path.parent / OBSERVED_BEST_CHECKPOINT_FILENAME).read_bytes() == (
        b"higher-but-ineligible"
    )
    observed_candidate = tracker["observed_best"]["dev_eval_candidate"]
    assert observed_candidate["eligible_for_best"] is False
    assert observed_candidate["ineligibility_reasons"] == ["confidence_prob"]
