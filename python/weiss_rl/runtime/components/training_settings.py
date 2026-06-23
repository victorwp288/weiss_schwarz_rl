"""Runtime-facing training settings resolved at QueueRuntime startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeTrainingSettings:
    actor_policy_backend: str
    actor_heuristic_fraction: float
    actor_heuristic_start_updates: int
    actor_heuristic_end_updates: int
    actor_heuristic_final_fraction: float
    train_on_heuristic_actor_rows: bool
    diverse_opponent_actor_count: int
    diverse_model_actor_count: int
    diverse_opponent_batch_fraction: float
    diverse_opponent_batch_wait_ms: int
    heuristic_actor_hidden_state_tracking: bool
    trajectory_retention_enabled: bool
    trajectory_retention_policy_ids: tuple[str, ...]
    trajectory_retention_sources: tuple[str, ...]
    actor_behavior_values_required: bool


def resolve_runtime_training_settings(
    *,
    training_config: Any,
    actor_count: int,
) -> RuntimeTrainingSettings:
    actor_policy_backend = (
        "model" if training_config is None else str(getattr(training_config, "actor_policy_backend", "model")).lower()
    ).strip()
    if actor_policy_backend not in {"model", "heuristic_public"}:
        raise ValueError("training.actor_policy_backend must be one of: model, heuristic_public")

    actor_heuristic_fraction = (
        1.0 if training_config is None else float(getattr(training_config, "actor_heuristic_fraction", 1.0))
    )
    _require_fraction(
        actor_heuristic_fraction,
        field_name="training.actor_heuristic_fraction",
    )
    actor_heuristic_start_updates = (
        0 if training_config is None else int(getattr(training_config, "actor_heuristic_start_updates", 0))
    )
    if actor_heuristic_start_updates < 0:
        raise ValueError("training.actor_heuristic_start_updates must be >= 0")

    actor_heuristic_end_updates = (
        -1 if training_config is None else int(getattr(training_config, "actor_heuristic_end_updates", -1))
    )
    if actor_heuristic_end_updates < -1:
        raise ValueError("training.actor_heuristic_end_updates must be >= -1")

    actor_heuristic_final_fraction = (
        actor_heuristic_fraction
        if training_config is None
        else float(getattr(training_config, "actor_heuristic_final_fraction", actor_heuristic_fraction))
    )
    _require_fraction(
        actor_heuristic_final_fraction,
        field_name="training.actor_heuristic_final_fraction",
    )
    if actor_heuristic_end_updates >= 0 and actor_heuristic_end_updates < actor_heuristic_start_updates:
        raise ValueError("training.actor_heuristic_end_updates must be >= training.actor_heuristic_start_updates")

    requested_diverse_actor_count = (
        0 if training_config is None else int(getattr(training_config, "diverse_opponent_actor_count", 0))
    )
    if requested_diverse_actor_count < 0:
        raise ValueError("training.diverse_opponent_actor_count must be >= 0")

    requested_diverse_model_actor_count = (
        0 if training_config is None else int(getattr(training_config, "diverse_model_actor_count", 0))
    )
    if requested_diverse_model_actor_count < 0:
        raise ValueError("training.diverse_model_actor_count must be >= 0")

    diverse_opponent_actor_count = min(int(actor_count), requested_diverse_actor_count)
    diverse_model_actor_count = min(int(diverse_opponent_actor_count), requested_diverse_model_actor_count)
    diverse_opponent_batch_fraction = (
        0.0 if training_config is None else float(getattr(training_config, "diverse_opponent_batch_fraction", 0.0))
    )
    _require_fraction(
        diverse_opponent_batch_fraction,
        field_name="training.diverse_opponent_batch_fraction",
    )
    diverse_opponent_batch_wait_ms = (
        0 if training_config is None else int(getattr(training_config, "diverse_opponent_batch_wait_ms", 0))
    )
    if diverse_opponent_batch_wait_ms < 0:
        raise ValueError("training.diverse_opponent_batch_wait_ms must be >= 0")

    trajectory_retention_enabled = bool(
        training_config is not None and float(getattr(training_config, "trajectory_retention_coef", 0.0)) > 0.0
    )
    algorithm_name = "" if training_config is None else str(getattr(training_config, "algorithm", "")).strip().lower()
    return RuntimeTrainingSettings(
        actor_policy_backend=actor_policy_backend,
        actor_heuristic_fraction=actor_heuristic_fraction,
        actor_heuristic_start_updates=actor_heuristic_start_updates,
        actor_heuristic_end_updates=actor_heuristic_end_updates,
        actor_heuristic_final_fraction=actor_heuristic_final_fraction,
        train_on_heuristic_actor_rows=(
            True if training_config is None else bool(getattr(training_config, "train_on_heuristic_actor_rows", True))
        ),
        diverse_opponent_actor_count=diverse_opponent_actor_count,
        diverse_model_actor_count=diverse_model_actor_count,
        diverse_opponent_batch_fraction=diverse_opponent_batch_fraction,
        diverse_opponent_batch_wait_ms=diverse_opponent_batch_wait_ms,
        heuristic_actor_hidden_state_tracking=(
            True
            if training_config is None
            else bool(getattr(training_config, "heuristic_actor_hidden_state_tracking", True))
        ),
        trajectory_retention_enabled=trajectory_retention_enabled,
        trajectory_retention_policy_ids=_clean_string_tuple(
            () if training_config is None else getattr(training_config, "trajectory_retention_policy_ids", ())
        ),
        trajectory_retention_sources=_clean_lower_string_tuple(
            () if training_config is None else getattr(training_config, "trajectory_retention_sources", ())
        ),
        actor_behavior_values_required="ppo" in algorithm_name,
    )


def _require_fraction(value: float, *, field_name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0 inclusive")


def _clean_string_tuple(values: Any) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def _clean_lower_string_tuple(values: Any) -> tuple[str, ...]:
    return tuple(str(value).strip().lower() for value in values if str(value).strip())


__all__ = ["RuntimeTrainingSettings", "resolve_runtime_training_settings"]
