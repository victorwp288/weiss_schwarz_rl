from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from weiss_rl.runtime.components.config import QueueRuntimeConfig, build_runtime_config


def _stack() -> Any:
    return SimpleNamespace(
        config=SimpleNamespace(
            system=SimpleNamespace(
                actor_process_count=12,
                envs_per_actor=8,
                actor_queue_capacity_unrolls=256,
            ),
            training=SimpleNamespace(
                batch_unrolls_per_update=128,
                actor_reload_interval_updates=0,
                actor_sampling_temperature=0.25,
                fixed_model_opponent_action_selection="argmax",
                mulligan_force_confirm_after_select=True,
                force_pass_over_main_move_only=True,
                main_move_only_max_consecutive=1,
                force_attack_over_pass_when_attack_legal=True,
            ),
            rewards=SimpleNamespace(
                shaping=SimpleNamespace(
                    pass_with_nonpass_penalty=0.02,
                    mulligan_select_with_confirm_penalty=0.03,
                    terminal_outcome_backfill_reward=0.04,
                    terminal_outcome_trace_backfill_reward=0.05,
                ),
            ),
        )
    )


def test_queue_runtime_config_total_envs_multiplies_actor_layout() -> None:
    config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=3,
        envs_per_actor=5,
        unroll_length=4,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=8,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )

    assert config.total_envs == 15


def test_build_runtime_config_preserves_minimal_batch_and_reload_clamp_behavior() -> None:
    config = build_runtime_config(
        stack=_stack(),
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )

    assert config.actor_count == 12
    assert config.envs_per_actor == 8
    assert config.batch_unrolls_per_update == 12
    assert config.queue_capacity_unrolls == 12
    assert config.actor_reload_interval_updates == 1
    assert config.pass_with_nonpass_penalty == pytest.approx(0.02)
    assert config.mulligan_select_with_confirm_penalty == pytest.approx(0.03)
    assert config.terminal_outcome_backfill_reward == pytest.approx(0.04)
    assert config.terminal_outcome_trace_backfill_reward == pytest.approx(0.05)
    assert config.actor_sampling_temperature == pytest.approx(0.25)
    assert config.fixed_model_opponent_action_selection == "argmax"
    assert config.mulligan_force_confirm_after_select is True
    assert config.force_pass_over_main_move_only is True
    assert config.main_move_only_max_consecutive == 1
    assert config.force_attack_over_pass_when_attack_legal is True


def test_build_runtime_config_requires_system_and_training_blocks() -> None:
    with pytest.raises(RuntimeError, match="stack config is missing system or training blocks"):
        build_runtime_config(
            stack=cast(Any, SimpleNamespace(config=SimpleNamespace(system=None, training=SimpleNamespace()))),
            num_envs=1,
            unroll_length=4,
            profile="fast",
            seed=7,
            pass_action_id=51,
            runtime_mode="train_ordered",
        )
