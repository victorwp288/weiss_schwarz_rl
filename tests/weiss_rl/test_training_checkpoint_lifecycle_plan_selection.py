from __future__ import annotations

from types import SimpleNamespace

import pytest
import weiss_rl.training.checkpointing.lifecycle_plans as checkpoint_lifecycle_plans


def _checkpoint_guard_stack() -> SimpleNamespace:
    return SimpleNamespace(
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


def _dev_eval_summary() -> dict[str, object]:
    return {
        "aggregate_score": 0.54,
        "anchors": {
            "B2": {
                "summary": {"games": 20, "truncations": 1},
                "uncertainty": {"prob_gt_half": 0.40, "prob_lt_half": 0.60, "ci_half_width": 0.15},
            }
        },
    }


def test_checkpoint_lifecycle_plan_skips_tracker_lookup_before_rollback_is_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_load_tracker(_training_paths: object) -> dict[str, object]:
        calls.append("tracker")
        raise AssertionError("ineligible rollback must not load checkpoint tracker")

    monkeypatch.setattr(checkpoint_lifecycle_plans, "load_checkpoint_tracker", fail_load_tracker)

    assert (
        checkpoint_lifecycle_plans.rollback_lifecycle_decision(
            stack=_checkpoint_guard_stack(),
            training_paths=object(),
            learner_update_count=30,
            dev_eval_summary={"aggregate_score": 0.40},
            last_rollback_update=15,
        )
        is None
    )
    assert (
        checkpoint_lifecycle_plans.rollback_lifecycle_decision(
            stack=_checkpoint_guard_stack(),
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
    best_record = {"metric_kind": "dev_eval_mean", "metric_value": 0.70, "update_count": 80}

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
        stack=_checkpoint_guard_stack(),
        training_paths=training_paths,
        learner_update_count=120,
        dev_eval_summary=_dev_eval_summary(),
        last_rollback_update=80,
    )
    finalize = checkpoint_lifecycle_plans.finalize_lifecycle_decision(
        stack=_checkpoint_guard_stack(),
        training_paths=training_paths,
        dev_eval_summary=_dev_eval_summary(),
    )

    assert rollback is not None
    assert rollback.current_score == pytest.approx(0.54)
    assert rollback.best.update_count == 80
    assert rollback.reasons == ["score_drop"]
    assert finalize is not None
    assert finalize.current_score == pytest.approx(0.54)
    assert finalize.best.score == pytest.approx(0.70)
