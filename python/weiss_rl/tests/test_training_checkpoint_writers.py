from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from weiss_rl.training import (
    checkpoint_alias_candidates,
    checkpoint_alias_publication,
    checkpoint_aliases,
    checkpoint_io,
    checkpoint_lifecycle,
    checkpoint_lifecycle_decisions,
    checkpoint_lifecycle_plans,
    checkpoint_lifecycle_transitions,
    checkpoint_structured_guard,
    checkpoint_tracker,
)
from weiss_rl.training.checkpoint_alias_mutation import (
    CheckpointAliasMutation,
    alias_record_for_mutation,
    apply_checkpoint_alias_mutation,
)
from weiss_rl.training.checkpoints import (
    OBSERVED_BEST_CHECKPOINT_FILENAME,
    append_checkpoint_guard_event,
    best_checkpoint_record,
    checkpoint_path_for_update,
    current_focal_policy_id,
    ensure_current_checkpoint,
    extract_structured_guard_b2_anchor_score,
    initialize_model_from_checkpoint,
    maybe_log_structured_mainmove_guard,
    publish_checkpoint_aliases,
    restore_minimal_train_checkpoint,
    write_minimal_train_checkpoint,
    write_scalars_record,
)


def test_checkpoints_reexports_canonical_checkpoint_alias_boundary() -> None:
    from weiss_rl.training import checkpoints

    assert checkpoints.load_checkpoint_tracker is checkpoint_aliases.load_checkpoint_tracker
    assert checkpoints.write_checkpoint_tracker is checkpoint_aliases.write_checkpoint_tracker
    assert checkpoints.build_checkpoint_record is checkpoint_aliases.build_checkpoint_record
    assert checkpoints.publish_checkpoint_aliases is checkpoint_aliases.publish_checkpoint_aliases
    assert checkpoints.observed_best_checkpoint_path is checkpoint_aliases.observed_best_checkpoint_path
    assert checkpoints.CHECKPOINT_TRACKER_FILENAME == checkpoint_aliases.CHECKPOINT_TRACKER_FILENAME
    assert checkpoint_aliases.publish_checkpoint_aliases.__module__ == "weiss_rl.training.checkpoint_aliases"


def test_checkpoint_aliases_reexport_canonical_candidate_boundary() -> None:
    assert checkpoint_aliases.CheckpointAliasCandidate is checkpoint_alias_candidates.CheckpointAliasCandidate
    assert checkpoint_aliases.checkpoint_alias_candidate is checkpoint_alias_candidates.checkpoint_alias_candidate
    assert (
        checkpoint_aliases.dev_eval_candidate_diagnostics is checkpoint_alias_candidates.dev_eval_candidate_diagnostics
    )
    assert checkpoint_aliases.should_update_observed_best is checkpoint_alias_candidates.should_update_observed_best
    assert checkpoint_alias_candidates.checkpoint_alias_candidate.__module__ == (
        "weiss_rl.training.checkpoint_alias_candidates"
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
        "weiss_rl.training.checkpoint_alias_publication"
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
    assert checkpoint_tracker.load_checkpoint_tracker.__module__ == "weiss_rl.training.checkpoint_tracker"


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


def test_checkpoints_reexports_canonical_checkpoint_lifecycle_boundary() -> None:
    from weiss_rl.training import checkpoints

    assert checkpoints.append_checkpoint_guard_event is checkpoint_lifecycle.append_checkpoint_guard_event
    assert checkpoints.checkpoint_guard_log_path is checkpoint_lifecycle.checkpoint_guard_log_path
    assert (
        checkpoints.extract_structured_guard_b2_anchor_score
        is checkpoint_lifecycle.extract_structured_guard_b2_anchor_score
    )
    assert checkpoints.maybe_log_structured_mainmove_guard is checkpoint_lifecycle.maybe_log_structured_mainmove_guard
    assert checkpoints.maybe_rollback_to_best_checkpoint is checkpoint_lifecycle.maybe_rollback_to_best_checkpoint
    assert checkpoints.maybe_finalize_from_best_checkpoint is checkpoint_lifecycle.maybe_finalize_from_best_checkpoint
    assert checkpoint_lifecycle.maybe_rollback_to_best_checkpoint.__module__ == (
        "weiss_rl.training.checkpoint_lifecycle"
    )


def test_checkpoint_lifecycle_rollback_decision_collects_reason_and_diagnostics() -> None:
    checkpoint_guard = SimpleNamespace(
        min_best_score=0.55,
        rollback_score_margin=0.10,
        rollback_truncation_rate_threshold=0.25,
        rollback_max_prob_lt_half=0.80,
    )

    decision = checkpoint_lifecycle.rollback_to_best_decision(
        checkpoint_guard=checkpoint_guard,
        best_record={"metric_kind": "dev_eval_mean", "metric_value": 0.70, "update_count": 80},
        learner_update_count=120,
        dev_eval_summary={
            "aggregate_score": 0.54,
            "anchors": {
                "B2": {
                    "summary": {"games": 20, "truncations": 1, "no_progress_timeouts": 3, "natural_timeouts": 2},
                    "uncertainty": {"prob_gt_half": 0.40, "prob_lt_half": 0.60, "ci_half_width": 0.15},
                }
            },
        },
    )

    assert decision is not None
    assert decision.current_score == pytest.approx(0.54)
    assert decision.best.score == pytest.approx(0.70)
    assert decision.best.update_count == 80
    assert decision.reasons == ["score_drop"]
    assert decision.confidence["max_prob_lt_half"] == pytest.approx(0.60)
    assert decision.worst_truncation_rate == pytest.approx(0.05)
    assert decision.worst_no_progress_timeout_rate == pytest.approx(0.15)
    assert decision.worst_natural_timeout_rate == pytest.approx(0.10)
    assert decision.worst_stall_rate == pytest.approx(0.15)


def test_checkpoint_lifecycle_reexports_canonical_decision_boundary() -> None:
    assert checkpoint_lifecycle.RollbackToBestDecision is checkpoint_lifecycle_decisions.RollbackToBestDecision
    assert checkpoint_lifecycle.FinalizeToBestDecision is checkpoint_lifecycle_decisions.FinalizeToBestDecision
    assert checkpoint_lifecycle.rollback_lifecycle_decision is checkpoint_lifecycle_plans.rollback_lifecycle_decision
    assert checkpoint_lifecycle.finalize_lifecycle_decision is checkpoint_lifecycle_plans.finalize_lifecycle_decision
    assert checkpoint_lifecycle.rollback_to_best_decision is checkpoint_lifecycle_decisions.rollback_to_best_decision
    assert checkpoint_lifecycle.rollback_to_best_event_payload is (
        checkpoint_lifecycle_decisions.rollback_to_best_event_payload
    )
    assert checkpoint_lifecycle.finalize_to_best_decision is checkpoint_lifecycle_decisions.finalize_to_best_decision
    assert checkpoint_lifecycle.finalize_to_best_event_payload is (
        checkpoint_lifecycle_decisions.finalize_to_best_event_payload
    )
    assert checkpoint_lifecycle_decisions.rollback_to_best_decision.__module__ == (
        "weiss_rl.training.checkpoint_lifecycle_decisions"
    )
    assert checkpoint_lifecycle_plans.rollback_lifecycle_decision.__module__ == (
        "weiss_rl.training.checkpoint_lifecycle_plans"
    )


def test_checkpoint_lifecycle_plan_skips_tracker_lookup_before_rollback_is_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                checkpoint_guard=SimpleNamespace(
                    enabled=True,
                    cooldown_updates=20,
                    min_best_score=0.55,
                    rollback_score_margin=0.1,
                    rollback_truncation_rate_threshold=0.25,
                    rollback_max_prob_lt_half=0.8,
                )
            )
        )
    )

    def fail_load_tracker(_training_paths: object) -> dict[str, object]:
        calls.append("tracker")
        raise AssertionError("ineligible rollback must not load checkpoint tracker")

    monkeypatch.setattr(checkpoint_lifecycle_plans, "load_checkpoint_tracker", fail_load_tracker)

    assert (
        checkpoint_lifecycle_plans.rollback_lifecycle_decision(
            stack=stack,
            training_paths=object(),
            learner_update_count=30,
            dev_eval_summary={"aggregate_score": 0.40},
            last_rollback_update=15,
        )
        is None
    )
    assert (
        checkpoint_lifecycle_plans.rollback_lifecycle_decision(
            stack=stack,
            training_paths=object(),
            learner_update_count=30,
            dev_eval_summary={"anchors": {}},
            last_rollback_update=None,
        )
        is None
    )
    assert calls == []


def test_checkpoint_lifecycle_plan_builds_rollback_and_finalize_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_paths = object()
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                checkpoint_guard=SimpleNamespace(
                    enabled=True,
                    cooldown_updates=20,
                    min_best_score=0.55,
                    rollback_score_margin=0.1,
                    rollback_truncation_rate_threshold=0.25,
                    rollback_max_prob_lt_half=0.8,
                )
            )
        )
    )
    best_record = {"metric_kind": "dev_eval_mean", "metric_value": 0.70, "update_count": 80}
    summary = {
        "aggregate_score": 0.54,
        "anchors": {
            "B2": {
                "summary": {"games": 20, "truncations": 1},
                "uncertainty": {"prob_gt_half": 0.40, "prob_lt_half": 0.60, "ci_half_width": 0.15},
            }
        },
    }

    monkeypatch.setattr(
        checkpoint_lifecycle_plans,
        "load_checkpoint_tracker",
        lambda received_paths: {"best": best_record} if received_paths is training_paths else {},
    )
    monkeypatch.setattr(
        checkpoint_lifecycle_plans,
        "best_checkpoint_record",
        lambda received_paths: best_record if received_paths is training_paths else None,
    )

    rollback = checkpoint_lifecycle_plans.rollback_lifecycle_decision(
        stack=stack,
        training_paths=training_paths,
        learner_update_count=120,
        dev_eval_summary=summary,
        last_rollback_update=80,
    )
    finalize = checkpoint_lifecycle_plans.finalize_lifecycle_decision(
        stack=stack,
        training_paths=training_paths,
        dev_eval_summary=summary,
    )

    assert rollback is not None
    assert rollback.current_score == pytest.approx(0.54)
    assert rollback.best.update_count == 80
    assert rollback.reasons == ["score_drop"]
    assert finalize is not None
    assert finalize.current_score == pytest.approx(0.54)
    assert finalize.best.score == pytest.approx(0.70)


def test_checkpoint_lifecycle_decision_payloads_preserve_decision_diagnostics() -> None:
    rollback_decision = checkpoint_lifecycle_decisions.RollbackToBestDecision(
        current_score=0.42,
        best=checkpoint_lifecycle_decisions.BestDevEvalCheckpoint(score=0.70, update_count=80),
        confidence={"min_prob_gt_half": 0.2, "max_prob_lt_half": 0.8, "max_ci_half_width": 0.15},
        reasons=["score_drop", "confidence"],
        worst_truncation_rate=0.05,
        worst_stall_rate=0.2,
        worst_no_progress_timeout_rate=0.1,
        worst_natural_timeout_rate=0.03,
    )

    rollback_payload = checkpoint_lifecycle_decisions.rollback_to_best_event_payload(
        learner_update_count=120,
        policy_version=7,
        decision=rollback_decision,
        best_checkpoint_path="training/checkpoints/best.pt",
        latest_checkpoint_path="training/checkpoints/latest.pt",
        publish_metrics={"snapshot_publish_latency_ms": 1.25, "snapshot_apply_latency_ms": 2.5},
        latest_metrics={"loss": 3.0},
        demoted_champions=["policy_000120"],
    )

    assert rollback_payload["action"] == "rollback_to_best"
    assert rollback_payload["current_score"] == pytest.approx(0.42)
    assert rollback_payload["best_score"] == pytest.approx(0.70)
    assert rollback_payload["best_update_count"] == 80
    assert rollback_payload["worst_stall_rate"] == pytest.approx(0.2)
    assert rollback_payload["reasons"] == ["score_drop", "confidence"]
    assert rollback_payload["rolled_back_checkpoint_path"] == "training/checkpoints/best.pt"
    assert rollback_payload["latest_loss"] == pytest.approx(3.0)
    assert rollback_payload["demoted_champions"] == ["policy_000120"]

    finalize_decision = checkpoint_lifecycle_decisions.FinalizeToBestDecision(
        current_score=0.60,
        best=checkpoint_lifecycle_decisions.BestDevEvalCheckpoint(score=0.70, update_count=80),
        confidence={"min_prob_gt_half": 0.45, "max_prob_lt_half": 0.55, "max_ci_half_width": 0.10},
    )
    finalize_payload = checkpoint_lifecycle_decisions.finalize_to_best_event_payload(
        learner_update_count=140,
        policy_version=9,
        decision=finalize_decision,
        latest_metrics=None,
        best_checkpoint_path="training/checkpoints/best.pt",
        latest_checkpoint_path="training/checkpoints/latest.pt",
        demoted_champions=[],
    )

    assert finalize_payload["action"] == "finalize_to_best"
    assert finalize_payload["current_score"] == pytest.approx(0.60)
    assert finalize_payload["best_score"] == pytest.approx(0.70)
    assert finalize_payload["latest_loss"] is None
    assert finalize_payload["demoted_champions"] == []


def test_checkpoint_lifecycle_transitions_apply_effects_and_build_relative_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    learner = _Learner()
    learner.update_count = 120
    learner.policy_version = 7
    runtime = object()
    learner_model = object()
    restore_checkpoint = object()
    calls: list[tuple[str, dict[str, object]]] = []

    rollback_decision = checkpoint_lifecycle_decisions.RollbackToBestDecision(
        current_score=0.42,
        best=checkpoint_lifecycle_decisions.BestDevEvalCheckpoint(score=0.70, update_count=80),
        confidence={"min_prob_gt_half": 0.2, "max_prob_lt_half": 0.8, "max_ci_half_width": 0.15},
        reasons=["score_drop"],
        worst_truncation_rate=0.05,
        worst_stall_rate=0.2,
        worst_no_progress_timeout_rate=0.1,
        worst_natural_timeout_rate=0.03,
    )

    def fake_rollback_effects(**kwargs: object) -> SimpleNamespace:
        calls.append(("rollback_effects", dict(kwargs)))
        return SimpleNamespace(
            best_checkpoint_path=paths.best_checkpoint_path,
            demoted_champions=["policy_000120"],
            publish_metrics={"snapshot_publish_latency_ms": 1.25, "snapshot_apply_latency_ms": 2.5},
        )

    monkeypatch.setattr(
        checkpoint_lifecycle_transitions,
        "apply_rollback_to_best_effects",
        fake_rollback_effects,
    )

    rollback_payload = checkpoint_lifecycle_transitions.apply_rollback_decision_to_event_payload(
        training_paths=paths,
        run_dir=tmp_path,
        runtime=runtime,
        learner=learner,
        learner_model=learner_model,
        latest_metrics={"loss": 3.0},
        decision=rollback_decision,
        restore_checkpoint=restore_checkpoint,
    )

    assert rollback_payload["action"] == "rollback_to_best"
    assert rollback_payload["update_count"] == 120
    assert rollback_payload["policy_version"] == 7
    assert rollback_payload["best_checkpoint_path"] == "training/checkpoints/best.pt"
    assert rollback_payload["latest_checkpoint_path"] == "training/checkpoints/latest.pt"
    assert rollback_payload["snapshot_publish_latency_ms"] == pytest.approx(1.25)
    assert rollback_payload["latest_loss"] == pytest.approx(3.0)
    assert rollback_payload["demoted_champions"] == ["policy_000120"]
    assert calls[0] == (
        "rollback_effects",
        {
            "training_paths": paths,
            "runtime": runtime,
            "learner_model": learner_model,
            "learner_update_count": 120,
            "best_update_count": 80,
            "restore_checkpoint": restore_checkpoint,
        },
    )

    finalize_decision = checkpoint_lifecycle_decisions.FinalizeToBestDecision(
        current_score=0.60,
        best=checkpoint_lifecycle_decisions.BestDevEvalCheckpoint(score=0.70, update_count=80),
        confidence={"min_prob_gt_half": 0.45, "max_prob_lt_half": 0.55, "max_ci_half_width": 0.10},
    )

    def fake_finalize_effects(**kwargs: object) -> SimpleNamespace:
        calls.append(("finalize_effects", dict(kwargs)))
        return SimpleNamespace(
            best_checkpoint_path=paths.best_checkpoint_path,
            demoted_champions=[],
            publish_metrics={},
        )

    monkeypatch.setattr(
        checkpoint_lifecycle_transitions,
        "apply_finalize_to_best_effects",
        fake_finalize_effects,
    )

    finalize_payload = checkpoint_lifecycle_transitions.apply_finalize_decision_to_event_payload(
        training_paths=paths,
        run_dir=tmp_path,
        runtime=runtime,
        learner=learner,
        latest_metrics=None,
        decision=finalize_decision,
        restore_checkpoint=restore_checkpoint,
    )

    assert finalize_payload["action"] == "finalize_to_best"
    assert finalize_payload["update_count"] == 120
    assert finalize_payload["policy_version"] == 7
    assert finalize_payload["best_checkpoint_path"] == "training/checkpoints/best.pt"
    assert finalize_payload["latest_checkpoint_path"] == "training/checkpoints/latest.pt"
    assert finalize_payload["latest_loss"] is None
    assert finalize_payload["demoted_champions"] == []
    assert calls[1] == (
        "finalize_effects",
        {
            "training_paths": paths,
            "runtime": runtime,
            "best_update_count": 80,
            "restore_checkpoint": restore_checkpoint,
        },
    )


def test_checkpoint_lifecycle_finalize_decision_requires_current_score_below_best() -> None:
    best_record = {"metric_kind": "dev_eval_mean", "metric_value": 0.70, "update_count": 80}
    losing_summary = {
        "aggregate_score": 0.60,
        "anchors": {
            "B2": {
                "summary": {"games": 20, "truncations": 0},
                "uncertainty": {"prob_gt_half": 0.45, "prob_lt_half": 0.55, "ci_half_width": 0.10},
            }
        },
    }

    decision = checkpoint_lifecycle.finalize_to_best_decision(
        best_record=best_record,
        dev_eval_summary=losing_summary,
    )

    assert decision is not None
    assert decision.current_score == pytest.approx(0.60)
    assert decision.best.update_count == 80
    assert decision.confidence["min_prob_gt_half"] == pytest.approx(0.45)
    assert (
        checkpoint_lifecycle.finalize_to_best_decision(
            best_record=best_record,
            dev_eval_summary={"aggregate_score": 0.70},
        )
        is None
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
    assert checkpoint_io.write_minimal_train_checkpoint.__module__ == "weiss_rl.training.checkpoint_io"


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


class _Model:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor([1.0])}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _Optimizer:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def state_dict(self) -> dict[str, float]:
        return {"lr": 0.01}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _GradScaler:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def state_dict(self) -> dict[str, float]:
        return {"scale": 2.0}

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict


class _Learner:
    update_count = 3
    total_samples_processed = 42

    def __init__(self, *, model=...) -> None:
        self.model = _Model() if model is ... else model
        self.optimizer = _Optimizer()
        self._grad_scaler = _GradScaler()
        self.policy_version = 7
        self.start_time = 0.0
        self.init_schedule_offset_updates = 0
        self.anchor_state: dict[str, torch.Tensor] | None = None
        self.loaded_anchor_state: dict[str, torch.Tensor] | None = None
        self.reset_anchor_calls = 0

    def get_policy_version(self) -> int:
        return self.policy_version

    def _optimizer_for_step(self) -> _Optimizer:
        return self.optimizer

    def policy_anchor_state_dict(self) -> dict[str, torch.Tensor] | None:
        return self.anchor_state

    def load_policy_anchor_state_dict(self, state_dict) -> None:
        self.loaded_anchor_state = state_dict

    def reset_policy_anchor_to_current_model(self) -> None:
        self.reset_anchor_calls += 1


class _TrainingPaths:
    def __init__(self, root: Path) -> None:
        self.checkpoints_dir = root
        self.checkpoint_tracker_path = root / "checkpoint_tracker.json"
        self.logs_dir = root / "logs"
        self.latest_checkpoint_path = root / "latest.pt"
        self.best_checkpoint_path = root / "best.pt"
        self.snapshots_dir = root / "snapshots"


def test_write_scalars_record_appends_stable_json_line(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("weiss_rl.training.checkpoint_io.time.time", lambda: 105.25)
    scalars_path = tmp_path / "scalars.jsonl"

    record = write_scalars_record(
        scalars_path=scalars_path,
        learner=_Learner(),
        metrics={"loss": 1.5},
        start_time=100.0,
    )

    assert record["update_count"] == 3
    assert record["policy_version"] == 7
    assert record["wall_clock_seconds"] == pytest.approx(5.25)
    assert record["wall_clock_ms"] == 5250
    assert json.loads(scalars_path.read_text(encoding="utf-8")) == record


def test_ensure_current_checkpoint_reuses_existing_file_and_writes_missing_file(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoints_dir.mkdir(parents=True)
    learner = _Learner()
    existing_path = checkpoint_path_for_update(paths.checkpoints_dir, update_count=learner.update_count)
    existing_path.write_bytes(b"existing")
    write_calls: list[Path] = []

    assert current_focal_policy_id(learner=learner) == "train_u3_p7"
    assert (
        ensure_current_checkpoint(
            training_paths=paths,
            learner=learner,
            write_checkpoint=lambda path: write_calls.append(path),
        )
        == existing_path
    )
    assert write_calls == []

    learner.update_count = 4

    def _write_checkpoint(path: Path) -> None:
        write_calls.append(path)
        path.write_bytes(b"new")

    new_path = ensure_current_checkpoint(
        training_paths=paths,
        learner=learner,
        write_checkpoint=_write_checkpoint,
    )

    assert new_path == paths.checkpoints_dir / "checkpoint_4.pt"
    assert new_path.read_bytes() == b"new"
    assert write_calls == [new_path]


def test_write_minimal_train_checkpoint_payload_shape_and_save(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    learner = _Learner()
    learner.init_schedule_offset_updates = 90

    payload = write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.25},
    )

    assert checkpoint_path.is_file()
    assert payload["format"] == "minimal_train_checkpoint_v1"
    assert payload["update_count"] == 3
    assert payload["policy_version"] == 7
    assert payload["device"] == "cpu"
    assert payload["config_hash256"] == "abc"
    assert payload["spec_hash256"] == "def"
    assert payload["algorithm"] == "impala_vtrace_gru"
    assert payload["recurrent_core"] == "gru"
    assert payload["total_samples_processed"] == 42
    assert payload["init_schedule_offset_updates"] == 90
    assert payload["policy_anchor_model_state_dict"] is None
    assert payload["public_heuristic_logit_bias_scale"] == pytest.approx(0.25)
    assert payload["optimizer_state_dict"] == {"lr": 0.01}
    assert payload["grad_scaler_state_dict"] == {"scale": 2.0}
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert loaded["update_count"] == payload["update_count"]
    assert loaded["model_state_dict"]["weight"].tolist() == [1.0]


def test_minimal_train_checkpoint_round_trips_policy_anchor_state(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.anchor_state = {"anchor_weight": torch.tensor([2.0])}

    payload = write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
    )
    restored = _Learner()
    restore_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=restored,
        device=torch.device("cpu"),
        expected_config_hash="abc",
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda _model, _payload: None,
    )

    assert payload["policy_anchor_model_state_dict"]["anchor_weight"].tolist() == [2.0]
    assert restored.loaded_anchor_state is not None
    assert restored.loaded_anchor_state["anchor_weight"].tolist() == [2.0]


def test_write_minimal_train_checkpoint_requires_model(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Cannot write a checkpoint without a learner model"):
        write_minimal_train_checkpoint(
            checkpoint_path=tmp_path / "checkpoint.pt",
            learner=_Learner(model=None),
            device=torch.device("cpu"),
            config_hash256="abc",
        )


def test_restore_minimal_train_checkpoint_restores_state_and_can_preserve_counters(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.init_schedule_offset_updates = 90
    write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="abc",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.25},
    )
    restored = _Learner()
    restored.update_count = 99
    restored.policy_version = 88
    restored.total_samples_processed = 77
    guidance_calls: list[tuple[object, float]] = []

    resume = restore_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=restored,
        device=torch.device("cpu"),
        expected_config_hash="abc",
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda model, payload: guidance_calls.append(
            (model, payload["public_heuristic_logit_bias_scale"])
        ),
        restore_counters=False,
    )

    assert resume.update_count == 99
    assert resume.policy_version == 88
    assert resume.total_samples_processed == 77
    assert restored.model.loaded_state is not None
    assert restored.optimizer.loaded_state == {"lr": 0.01}
    assert restored.init_schedule_offset_updates == 90
    assert resume.init_schedule_offset_updates == 90
    assert guidance_calls[0][0] is restored.model
    assert guidance_calls[0][1] == pytest.approx(0.25)


def test_initialize_model_from_checkpoint_loads_weights_without_optimizer_or_counters(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    writer = _Learner()
    writer.init_schedule_offset_updates = 90
    write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=writer,
        device=torch.device("cpu"),
        config_hash256="source-config",
        spec_hash256="def",
        algorithm="impala_vtrace_gru",
        recurrent_core="gru",
        guidance_payload={"public_heuristic_logit_bias_scale": 0.5},
    )
    initialized = _Learner()
    initialized.update_count = 99
    initialized.policy_version = 88
    initialized.total_samples_processed = 77
    guidance_calls: list[tuple[object, float]] = []

    source = initialize_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=initialized,
        device=torch.device("cpu"),
        expected_spec_hash256="def",
        algorithm="impala_vtrace_gru",
        restore_model_guidance=lambda model, payload: guidance_calls.append(
            (model, payload["public_heuristic_logit_bias_scale"])
        ),
    )

    assert source.update_count == 3
    assert source.policy_version == 7
    assert source.total_samples_processed == 42
    assert source.init_schedule_offset_updates == 90
    assert initialized.update_count == 99
    assert initialized.policy_version == 88
    assert initialized.total_samples_processed == 77
    assert initialized.model.loaded_state is not None
    assert initialized.optimizer.loaded_state is None
    assert initialized._grad_scaler.loaded_state is None
    assert initialized.loaded_anchor_state is None
    assert initialized.reset_anchor_calls == 1
    assert guidance_calls[0][0] is initialized.model
    assert guidance_calls[0][1] == pytest.approx(0.5)


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
        dev_eval_summary={
            "aggregate_score": 0.62,
            "anchors": {
                "B2 HeuristicPublic": {
                    "summary": {"games": 32, "truncations": 0, "no_progress_timeouts": 0, "natural_timeouts": 0},
                    "uncertainty": {"prob_gt_half": 0.58, "prob_lt_half": 0.42, "ci_half_width": 0.12},
                }
            },
        },
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
        dev_eval_summary={
            "aggregate_score": 0.59,
            "anchors": {
                "B2 HeuristicPublic": {
                    "summary": {"games": 32, "truncations": 0, "no_progress_timeouts": 0, "natural_timeouts": 0},
                    "uncertainty": {"prob_gt_half": 0.57, "prob_lt_half": 0.43, "ci_half_width": 0.12},
                }
            },
        },
    )

    assert tracker["latest"]["source_checkpoint_path"] == "checkpoint_50.pt"
    assert tracker["observed_best"]["source_checkpoint_path"] == "checkpoint_25.pt"
    assert observed_best_path.read_bytes() == b"checkpoint"


def test_publish_checkpoint_aliases_separates_observed_best_from_guarded_best(tmp_path) -> None:
    paths = _TrainingPaths(tmp_path / "training" / "checkpoints")
    paths.checkpoint_tracker_path.parent.mkdir(parents=True)
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
        dev_eval_summary={
            "aggregate_score": 0.61,
            "anchors": {
                "B2 HeuristicPublic": {
                    "summary": {"games": 32, "truncations": 0, "no_progress_timeouts": 0, "natural_timeouts": 0},
                    "uncertainty": {"prob_gt_half": 0.85, "prob_lt_half": 0.15, "ci_half_width": 0.12},
                }
            },
        },
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
        dev_eval_summary={
            "aggregate_score": 0.64,
            "anchors": {
                "B2 HeuristicPublic": {
                    "summary": {"games": 32, "truncations": 0, "no_progress_timeouts": 0, "natural_timeouts": 0},
                    "uncertainty": {"prob_gt_half": 0.55, "prob_lt_half": 0.45, "ci_half_width": 0.12},
                }
            },
        },
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
        "weiss_rl.training.checkpoint_structured_guard"
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
