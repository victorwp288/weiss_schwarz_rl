"""Child-runtime config helpers for process-backed collectors."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from weiss_rl.config import StackConfig
from weiss_rl.runtime.components.config import QueueRuntimeConfig


def stack_for_child_device_config(
    *,
    stack: StackConfig,
    actor_device_name: str | None,
    learner_device_name: str | None,
) -> StackConfig:
    system_config = stack.config.system
    stack_for_child = stack
    if system_config is not None:
        child_system = system_config
        if actor_device_name is not None:
            child_system = replace(child_system, actor_device=str(actor_device_name))
        if learner_device_name is not None:
            child_system = replace(child_system, learner_device=str(learner_device_name))
        if child_system is not system_config:
            stack_for_child = replace(
                stack,
                config=replace(
                    stack.config,
                    system=child_system,
                ),
            )
            system_config = stack_for_child.config.system
    if (
        system_config is not None
        and str(getattr(system_config, "collection_backend", "auto")).strip().lower() == "process"
    ):
        stack_for_child = replace(
            stack_for_child,
            config=replace(
                stack_for_child.config,
                system=replace(system_config, collection_backend="auto"),
            ),
        )
    return stack_for_child


def child_queue_runtime_config(config: QueueRuntimeConfig) -> QueueRuntimeConfig:
    return QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=1,
        envs_per_actor=int(config.envs_per_actor),
        unroll_length=int(config.unroll_length),
        batch_unrolls_per_update=1,
        queue_capacity_unrolls=1,
        profile=str(config.profile),
        base_seed=int(config.base_seed),
        pass_action_id=int(config.pass_action_id),
        actor_reload_interval_updates=int(config.actor_reload_interval_updates),
        pass_with_nonpass_penalty=float(getattr(config, "pass_with_nonpass_penalty", 0.0)),
        mulligan_select_with_confirm_penalty=float(getattr(config, "mulligan_select_with_confirm_penalty", 0.0)),
        terminal_outcome_backfill_reward=float(getattr(config, "terminal_outcome_backfill_reward", 0.0)),
        terminal_outcome_trace_backfill_reward=float(getattr(config, "terminal_outcome_trace_backfill_reward", 0.0)),
        actor_sampling_temperature=float(getattr(config, "actor_sampling_temperature", 1.0)),
        mulligan_force_confirm_after_select=bool(getattr(config, "mulligan_force_confirm_after_select", False)),
        force_pass_over_main_move_only=bool(getattr(config, "force_pass_over_main_move_only", False)),
        main_move_only_max_consecutive=int(getattr(config, "main_move_only_max_consecutive", 0)),
        force_attack_over_pass_when_attack_legal=bool(
            getattr(config, "force_attack_over_pass_when_attack_legal", False)
        ),
    )


def restore_parent_actor_lane_counts(
    *,
    runtime: Any,
    stack: StackConfig,
    parent_actor_count: int,
) -> None:
    """Preserve parent global actor-lane limits inside a single-actor child runtime."""

    training_config = getattr(getattr(stack, "config", None), "training", None)
    requested_diverse_actor_count = (
        0 if training_config is None else int(getattr(training_config, "diverse_opponent_actor_count", 0))
    )
    requested_diverse_model_actor_count = (
        0 if training_config is None else int(getattr(training_config, "diverse_model_actor_count", 0))
    )
    global_actor_count = max(0, int(parent_actor_count))
    diverse_actor_count = min(global_actor_count, max(0, requested_diverse_actor_count))
    runtime._diverse_opponent_actor_count = diverse_actor_count
    runtime._diverse_model_actor_count = min(diverse_actor_count, max(0, requested_diverse_model_actor_count))


__all__ = [
    "child_queue_runtime_config",
    "restore_parent_actor_lane_counts",
    "stack_for_child_device_config",
]
