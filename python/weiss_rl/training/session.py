"""Training-session lifecycle helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any


def wall_clock_budget_seconds(max_wall_clock_minutes: float | None) -> float | None:
    if max_wall_clock_minutes is None:
        return None
    return float(max_wall_clock_minutes) * 60.0


def wall_clock_budget_reached(
    *,
    start_time: float,
    max_wall_clock_seconds: float | None,
    now: float | None = None,
) -> bool:
    if max_wall_clock_seconds is None:
        return False
    current_time = time.time() if now is None else float(now)
    return (current_time - float(start_time)) >= float(max_wall_clock_seconds)


def wall_clock_budget_metric_updates(
    *,
    max_wall_clock_seconds: float,
    elapsed_seconds: float,
) -> dict[str, float]:
    return {
        "wall_clock_budget_reached": 1.0,
        "wall_clock_budget_seconds": float(max_wall_clock_seconds),
        "wall_clock_budget_elapsed_seconds": float(elapsed_seconds),
    }


def format_wall_clock_budget_reached_message(
    *,
    elapsed_seconds: float,
    max_wall_clock_seconds: float,
) -> str:
    return (
        f"Wall clock budget reached: elapsed={float(elapsed_seconds):.2f}s budget={float(max_wall_clock_seconds):.2f}s"
    )


def format_training_completed_message(metrics: Mapping[str, Any]) -> str:
    return (
        "Completed canonical single-node training run: "
        f"loss={float(metrics.get('loss', 0.0)):.6f} "
        f"policy_loss={float(metrics.get('policy_loss', 0.0)):.6f} "
        f"value_loss={float(metrics.get('value_loss', 0.0)):.6f} "
        f"entropy={float(metrics.get('entropy', 0.0)):.6f}"
    )


def format_resumed_learner_state_message(
    *,
    checkpoint_path: object,
    update_count: int,
    policy_version: int,
) -> str:
    return (
        "Resumed learner state: "
        f"checkpoint={checkpoint_path} "
        f"update={int(update_count)} "
        f"policy_version={int(policy_version)}"
    )


def format_resume_config_hash_mismatch_warning(
    *,
    checkpoint_config_hash: str,
    current_config_hash: str,
) -> str:
    return (
        "Warning: resuming checkpoint under a different config hash "
        f"(checkpoint={checkpoint_config_hash}, current={current_config_hash}). "
        "Use this only for explicit research continuations."
    )


def format_seeded_checkpoint_best_alias_message(seeded_best_record: Mapping[str, Any]) -> str:
    return (
        "Seeded checkpoint best alias from resumed dev-eval best: "
        f"update={int(seeded_best_record['update_count'])} "
        f"metric={float(seeded_best_record['metric_value']):.4f}"
    )


def format_seeded_resume_dev_eval_summary_message(
    *,
    update_count: int,
    aggregate_score: float,
) -> str:
    return f"Seeded resume dev-eval summary: update={int(update_count)} aggregate={float(aggregate_score):.4f}"


def format_structured_profiling_enabled_message(training_config: Any) -> str:
    return (
        "Structured profiling enabled: "
        f"profile_timers={bool(training_config.profile_timers)} "
        f"torch_profiler={bool(training_config.torch_profiler)} "
        f"structured_metrics_mode={training_config.structured_metrics_mode} "
        f"teacher_aux_mode={training_config.teacher_aux_mode} "
        f"fixed_opponent_backend={training_config.fixed_opponent_backend}"
    )
