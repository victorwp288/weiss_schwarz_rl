from __future__ import annotations

import queue
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.runtime import (
    QueueRuntime,
    build_runtime_config,
)
from weiss_rl.runtime.components.ipc_shared.collector_commands import handle_collector_commands

from .runtime_test_support import (
    _make_runtime_unroll,
)


def test_handle_collector_commands_tracks_update_and_refreshes_pool() -> None:
    refresh_calls: list[int] = []

    runtime = cast(
        Any,
        SimpleNamespace(
            _current_learner_update=0,
            _effective_learner_update=0,
            refresh_opponent_pool=lambda: refresh_calls.append(1),
            _teacher_policy=None,
            _opponent_heuristic_policies={},
            _opponent_models={},
            _opponent_model_locks={},
            _forced_fixed_opponent_policy_ids=(),
            _reset_actor_state_for_fixed_opponents=lambda actor: None,
        ),
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.loaded = 0
            self.evaluated = 0

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.loaded += len(state_dict)

        def eval(self) -> _FakeModel:
            self.evaluated += 1
            return self

    actor = cast(Any, SimpleNamespace(model=_FakeModel(), snapshot_version=0, fixed_opponent_policy_id_by_env=None))

    class _Queue:
        def __init__(self, commands: list[dict[str, Any]]) -> None:
            self._commands = list(commands)

        def get_nowait(self) -> dict[str, Any]:
            if not self._commands:
                raise queue.Empty
            return self._commands.pop(0)

    queue_obj = _Queue(
        [
            {"kind": "set_update", "update": 7, "refresh_opponent_pool": True},
            {"kind": "reload", "model_state_dict": {"w": torch.tensor([1.0])}, "update": 8, "effective_update": 5},
            {"kind": "refresh_opponent_pool", "update": 9},
        ]
    )

    should_stop = handle_collector_commands(
        runtime=runtime,
        actor=actor,
        control_queue=queue_obj,
        default_fixed_slots=None,
        default_forced_policy_ids=(),
        default_teacher_active=False,
        default_has_noleague_baseline=False,
    )

    assert should_stop is False
    assert actor.snapshot_version == 8
    assert runtime._current_learner_update == 9
    assert runtime._effective_learner_update == 5
    assert actor.model.loaded == 1
    assert actor.model.evaluated == 1
    assert len(refresh_calls) == 2


def test_build_runtime_config_minimal_batch_uses_one_unroll_per_actor() -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            system=SimpleNamespace(
                actor_process_count=12,
                envs_per_actor=8,
                actor_queue_capacity_unrolls=256,
            ),
            training=SimpleNamespace(
                batch_unrolls_per_update=128,
                actor_reload_interval_updates=1000,
            ),
        )
    )

    small = build_runtime_config(
        stack=cast(Any, stack),
        num_envs=1,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )
    assert small.actor_count == 1
    assert small.envs_per_actor == 1
    assert small.batch_unrolls_per_update == 1
    assert small.queue_capacity_unrolls == 1

    full = build_runtime_config(
        stack=cast(Any, stack),
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
        minimal_batch=True,
    )
    assert full.actor_count == 12
    assert full.envs_per_actor == 8
    assert full.batch_unrolls_per_update == 12
    assert full.queue_capacity_unrolls == 12

    default = build_runtime_config(
        stack=cast(Any, stack),
        num_envs=96,
        unroll_length=4,
        profile="fast",
        seed=7,
        pass_action_id=51,
        runtime_mode="train_ordered",
    )
    assert default.batch_unrolls_per_update == 128
    assert default.queue_capacity_unrolls == 256


def test_runtime_metrics_report_window_and_cumulative_env_step_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._runtime_start = 100.0
    runtime_any._runtime_last_metrics_time = 108.0
    runtime_any._runtime_cumulative_env_steps = 128
    runtime_any._last_published_snapshot_version = 5
    runtime_any._current_learner_update = 5
    runtime_any._effective_learner_update = 3
    runtime_any._league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=5,
            heuristic_public_final_mix_fraction=0.25,
        )
    )
    runtime_any._actor_heuristic_fraction = 1.0
    runtime_any._actor_heuristic_end_updates = 5
    runtime_any._actor_heuristic_final_fraction = 0.25
    runtime_any._pfsp_pool_size = 3
    runtime_any._pfsp_quarantined_opponents = 1
    runtime_any._pfsp_champion_pool_size = 1
    runtime_any._pfsp_recent_pool_size = 1
    runtime_any._pfsp_hard_negative_pool_size = 1
    runtime_any._pfsp_last_sampled_envs = 2
    runtime_any._pfsp_last_mirror_envs = 6
    runtime_any._pfsp_last_heuristic_public_envs = 2
    runtime_any._pfsp_last_noleague_baseline_envs = 1
    runtime_any._pfsp_last_champion_envs = 1
    runtime_any._pfsp_last_recent_envs = 0
    runtime_any._pfsp_last_hard_negative_envs = 1
    runtime_any._pfsp_epoch = 3

    monkeypatch.setattr("weiss_rl.runtime.queue_runtime.time.time", lambda: 110.0)
    metrics = QueueRuntime._runtime_metrics(
        runtime,
        [
            _make_runtime_unroll(
                actor_id=0,
                unroll_seq=0,
                behavior_policy_version=4,
                counters={
                    "engine_fault_done_rows": 2,
                    "no_progress_timeout_rows": 1,
                    "pass_actions": 3,
                    "main_move_actions": 4,
                    "max_consecutive_main_moves": 2,
                },
            ),
            replace(
                _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=5),
                obs=np.zeros((2, 3, 1), dtype=np.float32),
            ),
        ],
        occupancy_samples=[0.25, 0.75],
    )

    assert metrics["batch_env_steps"] == pytest.approx(7.0)
    assert metrics["actor_env_steps_per_sec"] == pytest.approx(3.5)
    assert metrics["actor_env_steps_per_sec_cumulative"] == pytest.approx(13.5)
    assert metrics["policy_version_lag_p50"] == pytest.approx(0.5)
    assert metrics["learner_actor_update_lag_p50"] == pytest.approx(0.5)
    assert metrics["learner_actor_update_lag_p90"] == pytest.approx(0.9)
    assert metrics["league_effective_update"] == pytest.approx(3.0)
    assert metrics["league_update_lag"] == pytest.approx(2.0)
    assert metrics["actor_heuristic_fraction_active"] == pytest.approx(0.55)
    assert metrics["heuristic_public_mix_fraction_active"] == pytest.approx(0.55)
    assert metrics["pfsp_quarantined_opponents"] == pytest.approx(1.0)
    assert metrics["pfsp_champion_pool_size"] == pytest.approx(1.0)
    assert metrics["pfsp_heuristic_public_envs"] == pytest.approx(2.0)
    assert metrics["pfsp_noleague_baseline_envs"] == pytest.approx(1.0)
    assert metrics["pfsp_hard_negative_envs"] == pytest.approx(1.0)
    assert metrics["pfsp_epoch"] == pytest.approx(3.0)
    assert metrics["queue_occupancy_p50"] == pytest.approx(0.5)
    assert metrics["collector_engine_fault_done_rows"] == pytest.approx(2.0)
    assert metrics["collector_no_progress_timeout_rows"] == pytest.approx(1.0)
    assert metrics["collector_pass_actions"] == pytest.approx(3.0)
    assert metrics["collector_main_move_actions"] == pytest.approx(4.0)
    assert metrics["collector_max_consecutive_main_moves"] == pytest.approx(2.0)
    assert runtime_any._runtime_last_metrics_time == pytest.approx(110.0)
    assert runtime_any._runtime_cumulative_env_steps == 135
