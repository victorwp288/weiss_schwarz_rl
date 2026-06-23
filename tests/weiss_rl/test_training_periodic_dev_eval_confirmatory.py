from __future__ import annotations

from types import SimpleNamespace

import pytest
from weiss_rl.training.checkpointing.periodic_dev_eval_confirmatory import (
    PeriodicDevEvalEffectiveSummary,
    checkpoint_tracker_best_record,
    maybe_run_confirmatory_dev_eval,
)

from tests.weiss_rl.training_periodic_dev_eval_guard_test_support import make_periodic_dev_eval_hooks


def test_periodic_dev_eval_confirmatory_helper_skips_without_request() -> None:
    summary = {"anchor_scores": {}, "aggregate_score": 0.25}

    result = maybe_run_confirmatory_dev_eval(
        hooks=make_periodic_dev_eval_hooks(
            load_checkpoint_tracker=lambda _paths: {"best": "not-a-record"},
            confirmatory_dev_eval_request=lambda **kwargs: None,
        ),
        stack=object(),
        learner=SimpleNamespace(update_count=4, get_policy_version=lambda: 9),
        summary_payload=summary,
        contract=object(),
        artifacts=object(),
        training_paths=object(),
        device=object(),
        run_id256="run-id",
        config_hash256="config",
        spec_hash256="spec",
        update_count=4,
    )

    assert result == PeriodicDevEvalEffectiveSummary(summary=summary)
    assert checkpoint_tracker_best_record({"best": "not-a-record"}) is None


def test_periodic_dev_eval_confirmatory_helper_expands_and_runs_override_pairs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    summary = {"anchor_scores": {"B2 HeuristicPublic": 0.25}, "aggregate_score": 0.25}
    effective_summary = {"anchor_scores": {"B2 HeuristicPublic": 0.5}, "aggregate_score": 0.5}
    best_record = {"policy_id": "best"}
    learner = SimpleNamespace(update_count=4, get_policy_version=lambda: 9)

    def confirmatory_request(**kwargs: object) -> dict[str, object]:
        events.append(("request", kwargs))
        return {"target_pairs": 3, "reasons": ["wide_ci"]}

    def run_eval(**kwargs: object) -> dict[str, object]:
        events.append(("run", kwargs))
        return effective_summary

    result = maybe_run_confirmatory_dev_eval(
        hooks=make_periodic_dev_eval_hooks(
            run_periodic_dev_eval=run_eval,
            load_checkpoint_tracker=lambda _paths: {"best": best_record},
            confirmatory_dev_eval_request=confirmatory_request,
            periodic_dev_eval_schedule=lambda _stack: (SimpleNamespace(name="dev_seeds.txt"), {}, [10, 20], "seed-sha"),
            expand_periodic_dev_eval_paired_seeds=lambda *args, **kwargs: (
                events.append(("expand", {"args": args, "kwargs": kwargs})) or ["pair-a", "pair-b", "pair-c"]
            ),
        ),
        stack=object(),
        learner=learner,
        summary_payload=summary,
        contract=object(),
        artifacts=object(),
        training_paths=object(),
        device=object(),
        run_id256="run-id",
        config_hash256="config",
        spec_hash256="spec",
        update_count=4,
    )

    assert result.summary is effective_summary
    assert result.confirmatory_request == {"target_pairs": 3, "reasons": ["wide_ci"]}
    assert result.confirmatory_pair_count == 3
    assert events[0][1]["existing_best_record"] is best_record
    assert events[1][1]["kwargs"] == {
        "requested_pairs": 3,
        "seed_file_sha256": "seed-sha",
        "update_count": 4,
        "policy_version": 9,
        "scope": "periodic_dev_eval_confirmatory",
    }
    assert events[2][1]["artifact_scope"] == "periodic_dev_eval_confirmatory"
    assert events[2][1]["paired_seeds_override"] == ["pair-a", "pair-b", "pair-c"]
    stdout = capsys.readouterr().out
    assert "Confirmatory dev eval: update=4 paired_seeds=3 aggregate=0.5000 reasons=wide_ci" in stdout
