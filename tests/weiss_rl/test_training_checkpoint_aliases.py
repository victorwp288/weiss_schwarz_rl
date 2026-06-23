from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import weiss_rl.training.checkpointing.alias_candidates as checkpoint_alias_candidates
import weiss_rl.training.checkpointing.alias_publication as checkpoint_alias_publication
import weiss_rl.training.checkpointing.aliases as checkpoint_aliases
import weiss_rl.training.checkpointing.tracker as checkpoint_tracker


def test_checkpoints_reexports_canonical_checkpoint_alias_boundary() -> None:
    from weiss_rl.training import checkpoints

    assert checkpoints.load_checkpoint_tracker is checkpoint_aliases.load_checkpoint_tracker
    assert checkpoints.write_checkpoint_tracker is checkpoint_aliases.write_checkpoint_tracker
    assert checkpoints.build_checkpoint_record is checkpoint_aliases.build_checkpoint_record
    assert checkpoints.publish_checkpoint_aliases is checkpoint_aliases.publish_checkpoint_aliases
    assert checkpoints.observed_best_checkpoint_path is checkpoint_aliases.observed_best_checkpoint_path
    assert checkpoints.CHECKPOINT_TRACKER_FILENAME == checkpoint_aliases.CHECKPOINT_TRACKER_FILENAME
    assert checkpoint_aliases.publish_checkpoint_aliases.__module__ == "weiss_rl.training.checkpointing.aliases"


def test_checkpoint_aliases_reexport_canonical_candidate_boundary() -> None:
    assert checkpoint_aliases.CheckpointAliasCandidate is checkpoint_alias_candidates.CheckpointAliasCandidate
    assert checkpoint_aliases.checkpoint_alias_candidate is checkpoint_alias_candidates.checkpoint_alias_candidate
    assert (
        checkpoint_aliases.dev_eval_candidate_diagnostics is checkpoint_alias_candidates.dev_eval_candidate_diagnostics
    )
    assert checkpoint_aliases.should_update_observed_best is checkpoint_alias_candidates.should_update_observed_best
    assert checkpoint_alias_candidates.checkpoint_alias_candidate.__module__ == (
        "weiss_rl.training.checkpointing.alias_candidates"
    )


def test_checkpoint_aliases_reexport_canonical_publication_boundary() -> None:
    assert checkpoint_aliases.CheckpointAliasPublication is checkpoint_alias_publication.CheckpointAliasPublication
    assert (
        checkpoint_aliases.apply_checkpoint_alias_publication
        is checkpoint_alias_publication.apply_checkpoint_alias_publication
    )
    assert (
        checkpoint_aliases.latest_checkpoint_alias_mutation
        is checkpoint_alias_publication.latest_checkpoint_alias_mutation
    )
    assert (
        checkpoint_aliases.best_checkpoint_alias_mutation is checkpoint_alias_publication.best_checkpoint_alias_mutation
    )
    assert (
        checkpoint_aliases.observed_best_checkpoint_alias_mutation
        is checkpoint_alias_publication.observed_best_checkpoint_alias_mutation
    )
    assert (
        checkpoint_aliases.maybe_publish_observed_best_checkpoint_alias
        is checkpoint_alias_publication.maybe_publish_observed_best_checkpoint_alias
    )
    assert (
        checkpoint_aliases.maybe_publish_best_checkpoint_alias
        is checkpoint_alias_publication.maybe_publish_best_checkpoint_alias
    )
    assert checkpoint_aliases.tracker_record is checkpoint_alias_publication.tracker_record
    assert (
        checkpoint_aliases.observed_best_checkpoint_path is checkpoint_alias_publication.observed_best_checkpoint_path
    )
    assert checkpoint_alias_publication.apply_checkpoint_alias_publication.__module__ == (
        "weiss_rl.training.checkpointing.alias_publication"
    )


def test_checkpoint_aliases_reexport_canonical_tracker_boundary() -> None:
    assert (
        checkpoint_aliases.default_checkpoint_tracker_payload is checkpoint_tracker.default_checkpoint_tracker_payload
    )
    assert checkpoint_aliases.load_checkpoint_tracker is checkpoint_tracker.load_checkpoint_tracker
    assert checkpoint_aliases.write_checkpoint_tracker is checkpoint_tracker.write_checkpoint_tracker
    assert checkpoint_aliases.best_checkpoint_record is checkpoint_tracker.best_checkpoint_record
    assert checkpoint_aliases.CheckpointTrainingPaths is checkpoint_tracker.CheckpointTrainingPaths
    assert checkpoint_aliases.CHECKPOINT_TRACKER_FILENAME == checkpoint_tracker.CHECKPOINT_TRACKER_FILENAME
    assert checkpoint_aliases.CHECKPOINT_TRACKER_FORMAT == checkpoint_tracker.CHECKPOINT_TRACKER_FORMAT
    assert checkpoint_tracker.load_checkpoint_tracker.__module__ == "weiss_rl.training.checkpointing.tracker"


def test_checkpoint_tracker_loads_defaults_and_rejects_non_object(tmp_path: Path) -> None:
    paths = SimpleNamespace(checkpoint_tracker_path=tmp_path / "checkpoint_tracker.json")

    missing_tracker = checkpoint_tracker.load_checkpoint_tracker(paths)
    assert missing_tracker == {
        "format": "checkpoint_tracker_v1",
        "latest": None,
        "best": None,
        "observed_best": None,
    }
    fresh_tracker = checkpoint_tracker.default_checkpoint_tracker_payload()
    fresh_tracker["latest"] = {"alias": "latest"}
    assert checkpoint_tracker.default_checkpoint_tracker_payload()["latest"] is None

    paths.checkpoint_tracker_path.write_text(json.dumps({"best": {"alias": "best"}}), encoding="utf-8")
    loaded_tracker = checkpoint_tracker.load_checkpoint_tracker(paths)
    assert loaded_tracker["format"] == "checkpoint_tracker_v1"
    assert loaded_tracker["latest"] is None
    assert loaded_tracker["best"] == {"alias": "best"}
    assert loaded_tracker["observed_best"] is None
    assert checkpoint_tracker.best_checkpoint_record(paths) == {"alias": "best"}

    paths.checkpoint_tracker_path.write_text(json.dumps({"best": ["not-a-record"]}), encoding="utf-8")
    assert checkpoint_tracker.best_checkpoint_record(paths) is None

    paths.checkpoint_tracker_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint tracker must be a JSON object"):
        checkpoint_tracker.load_checkpoint_tracker(paths)


def test_checkpoint_tracker_write_uses_stable_sorted_json(tmp_path: Path) -> None:
    paths = SimpleNamespace(checkpoint_tracker_path=tmp_path / "checkpoint_tracker.json")
    payload = {"observed_best": None, "latest": {"z": 2, "a": 1}, "format": "checkpoint_tracker_v1", "best": None}

    checkpoint_tracker.write_checkpoint_tracker(paths, payload)

    assert paths.checkpoint_tracker_path.read_text(encoding="utf-8") == (
        "{\n"
        '  "best": null,\n'
        '  "format": "checkpoint_tracker_v1",\n'
        '  "latest": {\n'
        '    "a": 1,\n'
        '    "z": 2\n'
        "  },\n"
        '  "observed_best": null\n'
        "}\n"
    )
    assert checkpoint_tracker.load_checkpoint_tracker(paths) == payload


def test_checkpoint_alias_candidate_collects_dev_eval_diagnostics_and_observed_best_rules() -> None:
    stack = SimpleNamespace(
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
    dev_eval_summary = {
        "aggregate_score": 0.62,
        "anchors": {
            "B2 HeuristicPublic": {
                "summary": {"games": 32, "truncations": 2, "no_progress_timeouts": 3, "natural_timeouts": 1},
                "uncertainty": {"prob_gt_half": 0.58, "prob_lt_half": 0.42, "ci_half_width": 0.12},
            }
        },
    }

    candidate = checkpoint_alias_candidates.checkpoint_alias_candidate(
        stack=stack,
        latest_metrics={"loss": 0.5},
        dev_eval_summary=dev_eval_summary,
    )

    assert candidate.metric_kind is None
    assert candidate.metric_value is None
    assert candidate.observed_score == pytest.approx(0.62)
    assert candidate.dev_eval_candidate is not None
    assert candidate.dev_eval_candidate["score"] == pytest.approx(0.62)
    assert candidate.dev_eval_candidate["eligible_for_best"] is False
    assert candidate.dev_eval_candidate["ineligibility_reasons"] == ["confidence_prob"]
    assert candidate.dev_eval_candidate["confidence"]["min_prob_gt_half"] == pytest.approx(0.58)
    assert candidate.dev_eval_candidate["worst_truncation_rate"] == pytest.approx(2 / 32)
    assert candidate.dev_eval_candidate["worst_no_progress_timeout_rate"] == pytest.approx(3 / 32)
    assert candidate.dev_eval_candidate["worst_natural_timeout_rate"] == pytest.approx(1 / 32)
    assert candidate.dev_eval_candidate["worst_stall_rate"] == pytest.approx(3 / 32)
    assert (
        checkpoint_alias_candidates.dev_eval_candidate_diagnostics(
            stack=stack,
            dev_eval_summary=None,
        )
        is None
    )
    assert (
        checkpoint_alias_candidates.should_update_observed_best(
            existing_record={"metric_value": 0.61},
            observed_score=0.62,
        )
        is True
    )
    assert (
        checkpoint_alias_candidates.should_update_observed_best(
            existing_record={"metric_value": 0.63},
            observed_score=0.62,
        )
        is False
    )
    assert (
        checkpoint_alias_candidates.should_update_observed_best(
            existing_record={"metric_value": "stale"},
            observed_score=0.62,
        )
        is True
    )
    assert (
        checkpoint_alias_candidates.should_update_observed_best(
            existing_record=None,
            observed_score=None,
        )
        is False
    )
