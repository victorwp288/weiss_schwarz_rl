from __future__ import annotations

from collections.abc import Callable

from weiss_rl.training.checkpointing.guards.periodic_dev_eval import TrainingPeriodicDevEvalHooks


def unexpected_hook(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("periodic dev-eval hook was not expected to run")


def make_periodic_dev_eval_hooks(**overrides: Callable[..., object]) -> TrainingPeriodicDevEvalHooks:
    hooks: dict[str, Callable[..., object]] = {
        "should_run_periodic_dev_eval": unexpected_hook,
        "run_periodic_dev_eval": unexpected_hook,
        "slug_policy_id": unexpected_hook,
        "load_checkpoint_tracker": unexpected_hook,
        "confirmatory_dev_eval_request": unexpected_hook,
        "periodic_dev_eval_schedule": unexpected_hook,
        "expand_periodic_dev_eval_paired_seeds": unexpected_hook,
        "ensure_current_checkpoint": unexpected_hook,
        "publish_checkpoint_aliases": unexpected_hook,
        "maybe_log_structured_mainmove_guard": unexpected_hook,
        "maybe_rollback_to_best_checkpoint": unexpected_hook,
    }
    hooks.update(overrides)
    return TrainingPeriodicDevEvalHooks(**hooks)
