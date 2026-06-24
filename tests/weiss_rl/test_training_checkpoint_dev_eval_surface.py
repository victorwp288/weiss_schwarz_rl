from __future__ import annotations

import weiss_rl.training.checkpointing.guards.periodic_dev_eval as checkpoint_periodic_dev_eval
import weiss_rl.training.checkpointing.guards.periodic_dev_eval_confirmatory as checkpoint_periodic_dev_eval_confirmatory
import weiss_rl.training.checkpointing.guards.periodic_dev_eval_guard as checkpoint_periodic_dev_eval_guard


def test_checkpoint_periodic_dev_eval_reexports_guard_application_boundary() -> None:
    assert (
        checkpoint_periodic_dev_eval.PeriodicDevEvalEffectiveSummary
        is checkpoint_periodic_dev_eval_confirmatory.PeriodicDevEvalEffectiveSummary
    )
    assert (
        checkpoint_periodic_dev_eval.checkpoint_tracker_best_record
        is checkpoint_periodic_dev_eval_confirmatory.checkpoint_tracker_best_record
    )
    assert (
        checkpoint_periodic_dev_eval.maybe_run_confirmatory_dev_eval
        is checkpoint_periodic_dev_eval_confirmatory.maybe_run_confirmatory_dev_eval
    )
    assert (
        checkpoint_periodic_dev_eval.CheckpointGuardApplicationResult
        is checkpoint_periodic_dev_eval_guard.CheckpointGuardApplicationResult
    )
    assert (
        checkpoint_periodic_dev_eval.apply_periodic_dev_eval_checkpoint_guard
        is checkpoint_periodic_dev_eval_guard.apply_periodic_dev_eval_checkpoint_guard
    )
    assert checkpoint_periodic_dev_eval_guard.apply_periodic_dev_eval_checkpoint_guard.__module__ == (
        "weiss_rl.training.checkpointing.guards.periodic_dev_eval_guard"
    )
    assert checkpoint_periodic_dev_eval_confirmatory.maybe_run_confirmatory_dev_eval.__module__ == (
        "weiss_rl.training.checkpointing.guards.periodic_dev_eval_confirmatory"
    )
