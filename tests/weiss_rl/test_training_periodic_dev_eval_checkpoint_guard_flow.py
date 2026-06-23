from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from weiss_rl.training.checkpointing.periodic_dev_eval import (
    PeriodicDevEvalGuardResult,
    maybe_run_periodic_dev_eval_and_checkpoint_guard,
)

from tests.weiss_rl.training_periodic_dev_eval_guard_test_support import make_periodic_dev_eval_hooks


class FakeTensorBoardLogger:
    def __init__(self, events: list[tuple[str, dict[str, object]]]) -> None:
        self.events = events

    def log_periodic_dev_eval(self, payload: object, *, step: int) -> None:
        self.events.append(("tb_eval", {"payload": payload, "step": step}))

    def log_checkpoint_tracker(self, payload: object, *, step: int) -> None:
        self.events.append(("tb_tracker", {"payload": payload, "step": step}))


def test_periodic_dev_eval_guard_runs_confirmatory_eval_and_stop_after_rollback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    latest_metrics = {"loss": 1.0}
    learner = SimpleNamespace(update_count=8, get_policy_version=lambda: 13)
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(checkpoint_guard=SimpleNamespace(stop_after_rollback=True)),
        )
    )
    artifacts = SimpleNamespace(run_dir=tmp_path / "run")
    training_paths = SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints")
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint_8.pt"
    summary_payload = {"anchor_scores": {"B2 HeuristicPublic": 0.1}, "aggregate_score": 0.25}
    effective_summary = {"anchor_scores": {"B2 HeuristicPublic": 0.2}, "aggregate_score": 0.5}

    def run_periodic_dev_eval(**kwargs: object) -> dict[str, object]:
        events.append(("run_eval", kwargs))
        if kwargs.get("artifact_scope") == "periodic_dev_eval_confirmatory":
            return effective_summary
        return summary_payload

    def slug_policy_id(policy_id: str) -> str:
        events.append(("slug", {"policy_id": policy_id}))
        return "b2"

    def load_checkpoint_tracker(paths: object) -> dict[str, object]:
        events.append(("load_tracker", {"paths": paths}))
        return {"best": {"policy_id": "best"}}

    def confirmatory_dev_eval_request(**kwargs: object) -> dict[str, object]:
        events.append(("confirm_request", kwargs))
        return {"target_pairs": 3, "reasons": ["wide_ci", "new_best"]}

    def periodic_dev_eval_schedule(config_stack: object) -> tuple[object, list[object], list[int], str]:
        events.append(("schedule", {"stack": config_stack}))
        return SimpleNamespace(name="dev_seeds.txt"), [], [10, 20], "seed-sha"

    def expand_periodic_dev_eval_paired_seeds(*args: object, **kwargs: object) -> list[str]:
        events.append(("expand", {"args": args, "kwargs": kwargs}))
        return ["pair-a", "pair-b", "pair-c"]

    def maybe_rollback_to_best_checkpoint(**kwargs: object) -> dict[str, object]:
        events.append(("rollback", kwargs))
        return {
            "update_count": 8,
            "best_update_count": 6,
            "current_score": 0.25,
            "best_score": 0.5,
            "reasons": ["score_regressed"],
        }

    result = maybe_run_periodic_dev_eval_and_checkpoint_guard(
        learner=learner,
        model=object(),
        stack=stack,
        contract=object(),
        artifacts=artifacts,
        training_paths=training_paths,
        runtime=object(),
        device=object(),
        spec_hash256="spec-hash",
        algorithm=object(),
        latest_metrics=latest_metrics,
        last_dev_eval_summary=None,
        last_dev_eval_update_count=None,
        last_checkpoint_guard_rollback_update=4,
        run_id256="run-id",
        config_hash256="config-hash",
        tensorboard_logger=FakeTensorBoardLogger(events),
        hooks=make_periodic_dev_eval_hooks(
            should_run_periodic_dev_eval=lambda *_args, **_kwargs: True,
            run_periodic_dev_eval=run_periodic_dev_eval,
            slug_policy_id=slug_policy_id,
            load_checkpoint_tracker=load_checkpoint_tracker,
            confirmatory_dev_eval_request=confirmatory_dev_eval_request,
            periodic_dev_eval_schedule=periodic_dev_eval_schedule,
            expand_periodic_dev_eval_paired_seeds=expand_periodic_dev_eval_paired_seeds,
            ensure_current_checkpoint=lambda **kwargs: events.append(("ensure_checkpoint", kwargs)) or checkpoint_path,
            publish_checkpoint_aliases=lambda **kwargs: events.append(("aliases", kwargs)) or {"best": {"update": 8}},
            maybe_log_structured_mainmove_guard=lambda **kwargs: events.append(("guard_log", kwargs)),
            maybe_rollback_to_best_checkpoint=maybe_rollback_to_best_checkpoint,
        ),
    )

    assert result == PeriodicDevEvalGuardResult(
        last_dev_eval_summary=effective_summary,
        last_dev_eval_update_count=8,
        last_checkpoint_guard_rollback_update=8,
        stop_requested=True,
    )
    assert latest_metrics["checkpoint_guard_stop_after_rollback"] == 1.0
    assert [event[0] for event in events] == [
        "run_eval",
        "slug",
        "load_tracker",
        "confirm_request",
        "schedule",
        "expand",
        "run_eval",
        "ensure_checkpoint",
        "aliases",
        "guard_log",
        "rollback",
        "tb_eval",
        "tb_tracker",
    ]
    assert events[0][1]["run_id256"] == "run-id"
    assert events[0][1]["config_hash256"] == "config-hash"
    assert events[0][1]["spec_hash256"] == "spec-hash"
    assert events[3][1]["existing_best_record"] == {"policy_id": "best"}
    assert events[3][1]["dev_eval_summary"] is summary_payload
    assert events[5][1]["args"] == ([10, 20],)
    assert events[5][1]["kwargs"] == {
        "requested_pairs": 3,
        "seed_file_sha256": "seed-sha",
        "update_count": 8,
        "policy_version": 13,
        "scope": "periodic_dev_eval_confirmatory",
    }
    assert events[6][1]["artifact_dir_name"] == "dev_eval_confirmatory"
    assert events[6][1]["paired_seeds_override"] == ["pair-a", "pair-b", "pair-c"]
    assert events[6][1]["persist_summary"] is False
    assert events[6][1]["update_stall_monitor"] is False
    assert events[8][1]["checkpoint_path"] == checkpoint_path
    assert events[8][1]["dev_eval_summary"] is effective_summary
    assert events[9][1]["dev_eval_summary"] is effective_summary
    assert events[10][1]["last_rollback_update"] == 4
    assert events[10][1]["dev_eval_summary"] is effective_summary
    assert events[11][1] == {"payload": effective_summary, "step": 8}
    assert events[12][1] == {"payload": {"best": {"update": 8}}, "step": 8}
    stdout = capsys.readouterr().out
    assert "Periodic dev eval: update=8 opponent=b2 aggregate=0.2500 anchors=B2 HeuristicPublic" in stdout
    assert "Confirmatory dev eval: update=8 paired_seeds=3 aggregate=0.5000" in stdout
    assert "Checkpoint guard rollback: update=8 best_update=6 current_score=0.2500 best_score=0.5000" in stdout
    assert "Checkpoint guard early stop after rollback: update=8 best_update=6" in stdout
