"""Queue runtime configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass

from weiss_rl.config import StackConfig
from weiss_rl.runtime.components.topology import QueueRuntimeMode, resolve_actor_topology


@dataclass(frozen=True, slots=True)
class QueueRuntimeConfig:
    mode: QueueRuntimeMode
    actor_count: int
    envs_per_actor: int
    unroll_length: int
    batch_unrolls_per_update: int
    queue_capacity_unrolls: int
    profile: str
    base_seed: int
    pass_action_id: int
    actor_reload_interval_updates: int
    pass_with_nonpass_penalty: float = 0.0
    mulligan_select_with_confirm_penalty: float = 0.0
    terminal_outcome_backfill_reward: float = 0.0
    terminal_outcome_trace_backfill_reward: float = 0.0
    actor_sampling_temperature: float = 1.0
    mulligan_force_confirm_after_select: bool = False
    force_pass_over_main_move_only: bool = False
    main_move_only_max_consecutive: int = 0
    force_attack_over_pass_when_attack_legal: bool = False
    fixed_model_opponent_action_selection: str = "sample"

    @property
    def total_envs(self) -> int:
        return int(self.actor_count * self.envs_per_actor)


def build_runtime_config(
    *,
    stack: StackConfig,
    num_envs: int,
    unroll_length: int,
    profile: str,
    seed: int,
    pass_action_id: int,
    runtime_mode: QueueRuntimeMode,
    minimal_batch: bool = False,
) -> QueueRuntimeConfig:
    system = stack.config.system
    training = stack.config.training
    if system is None or training is None:
        raise RuntimeError("stack config is missing system or training blocks")
    rewards = getattr(stack.config, "rewards", None)
    reward_shaping = None if rewards is None else getattr(rewards, "shaping", None)

    configured_actor_count = int(system.actor_process_count)
    configured_envs_per_actor = int(system.envs_per_actor)
    actor_count, envs_per_actor = resolve_actor_topology(
        num_envs=int(num_envs),
        runtime_mode=runtime_mode,
        configured_actor_count=configured_actor_count,
        configured_envs_per_actor=configured_envs_per_actor,
    )

    batch_unrolls_per_update = int(training.batch_unrolls_per_update)
    queue_capacity_unrolls = max(int(system.actor_queue_capacity_unrolls), batch_unrolls_per_update)
    if minimal_batch:
        batch_unrolls_per_update = int(actor_count)
        queue_capacity_unrolls = int(actor_count)

    return QueueRuntimeConfig(
        mode=runtime_mode,
        actor_count=actor_count,
        envs_per_actor=envs_per_actor,
        unroll_length=int(unroll_length),
        batch_unrolls_per_update=batch_unrolls_per_update,
        queue_capacity_unrolls=queue_capacity_unrolls,
        profile=profile,
        base_seed=int(seed),
        pass_action_id=int(pass_action_id),
        actor_reload_interval_updates=max(1, int(training.actor_reload_interval_updates)),
        pass_with_nonpass_penalty=float(getattr(reward_shaping, "pass_with_nonpass_penalty", 0.0)),
        mulligan_select_with_confirm_penalty=float(
            getattr(reward_shaping, "mulligan_select_with_confirm_penalty", 0.0)
        ),
        terminal_outcome_backfill_reward=float(getattr(reward_shaping, "terminal_outcome_backfill_reward", 0.0)),
        terminal_outcome_trace_backfill_reward=float(
            getattr(reward_shaping, "terminal_outcome_trace_backfill_reward", 0.0)
        ),
        actor_sampling_temperature=float(getattr(training, "actor_sampling_temperature", 1.0)),
        mulligan_force_confirm_after_select=bool(getattr(training, "mulligan_force_confirm_after_select", False)),
        force_pass_over_main_move_only=bool(getattr(training, "force_pass_over_main_move_only", False)),
        main_move_only_max_consecutive=int(getattr(training, "main_move_only_max_consecutive", 0)),
        force_attack_over_pass_when_attack_legal=bool(
            getattr(training, "force_attack_over_pass_when_attack_legal", False)
        ),
        fixed_model_opponent_action_selection=str(getattr(training, "fixed_model_opponent_action_selection", "sample")),
    )
