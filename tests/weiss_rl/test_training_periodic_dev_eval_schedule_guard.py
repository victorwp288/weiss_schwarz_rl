from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.checkpointing.guards.periodic_dev_eval import (
    PeriodicDevEvalGuardResult,
    maybe_run_periodic_dev_eval_and_checkpoint_guard,
)

from tests.weiss_rl.training_periodic_dev_eval_guard_test_support import make_periodic_dev_eval_hooks


def test_periodic_dev_eval_guard_skips_when_schedule_says_no(tmp_path: Path) -> None:
    previous_summary = {"aggregate_score": 0.1}

    result = maybe_run_periodic_dev_eval_and_checkpoint_guard(
        learner=SimpleNamespace(update_count=7),
        model=object(),
        stack=object(),
        contract=object(),
        artifacts=SimpleNamespace(run_dir=tmp_path / "run"),
        training_paths=SimpleNamespace(checkpoints_dir=tmp_path / "checkpoints"),
        runtime=object(),
        device=object(),
        spec_hash256="spec",
        algorithm=object(),
        latest_metrics={"loss": 1.0},
        last_dev_eval_summary=previous_summary,
        last_dev_eval_update_count=3,
        last_checkpoint_guard_rollback_update=2,
        run_id256="run-id",
        config_hash256="config",
        tensorboard_logger=None,
        hooks=make_periodic_dev_eval_hooks(
            should_run_periodic_dev_eval=lambda *_args, **_kwargs: False,
        ),
    )

    assert result == PeriodicDevEvalGuardResult(
        last_dev_eval_summary=previous_summary,
        last_dev_eval_update_count=3,
        last_checkpoint_guard_rollback_update=2,
        stop_requested=False,
    )
