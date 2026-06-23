from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.runtime import (
    QueueRuntime,
    QueueRuntimeConfig,
)
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


def test_structured_warmstart_source_mix_balances_sources_and_restores_actor_slots() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=8,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=7,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._collector_result_queue = None
    runtime_any._teacher_policy = object()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._opponent_models = {NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._device = torch.device("cpu")
    runtime_any._forced_fixed_opponent_policy_ids = ()

    def fake_assign_episode_roles(actor, done, *, initial: bool = False) -> None:
        fixed = actor.fixed_opponent_policy_id_by_env
        assigned = np.full(actor.focal_seat_by_env.shape, MIRROR_OPPONENT_POLICY_ID, dtype=object)
        if fixed is not None:
            fixed_array = np.asarray(fixed, dtype=object)
            non_empty = np.asarray([bool(str(value).strip()) for value in fixed_array.tolist()], dtype=np.bool_)
            assigned[non_empty] = fixed_array[non_empty]
        actor.opponent_policy_id_by_env[:] = assigned

    runtime_any._assign_episode_roles = fake_assign_episode_roles

    class _FakeEnv:
        def reset_done(self, done):
            return {"reset_done": np.asarray(done, dtype=np.bool_).copy()}

    class _FakeModel:
        def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
            return torch.zeros((batch_size, 3), device=device)

    actor = cast(
        Any,
        SimpleNamespace(
            env=_FakeEnv(),
            model=_FakeModel(),
            rng=np.random.default_rng(7),
            seat_hidden=torch.ones((8, 3)),
            opponent_hidden=torch.ones((8, 3)),
            focal_seat_by_env=np.zeros((8,), dtype=np.int64),
            opponent_policy_id_by_env=np.full((8,), MIRROR_OPPONENT_POLICY_ID, dtype=object),
            fixed_opponent_policy_id_by_env=None,
            current_batch=None,
        ),
    )
    runtime_any._actors = [actor]

    with QueueRuntime.structured_warmstart_source_mix(runtime) as metrics:
        assert metrics["structured_warmstart_source_count"] == 3.0
        assert metrics["structured_warmstart_self_play_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b1_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b2_envs_per_actor"] == 2.0
        assert actor.fixed_opponent_policy_id_by_env is not None
        assert actor.opponent_policy_id_by_env.tolist().count(NOLEAGUE_BASELINE_POLICY_ID) == 3
        assert actor.opponent_policy_id_by_env.tolist().count(HEURISTIC_PUBLIC_POLICY_ID) == 2
        assert actor.opponent_policy_id_by_env.tolist().count(MIRROR_OPPONENT_POLICY_ID) == 3
        assert HEURISTIC_PUBLIC_POLICY_ID in runtime_any._opponent_heuristic_policies
        assert runtime_any._forced_fixed_opponent_policy_ids == (
            NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )

    assert actor.fixed_opponent_policy_id_by_env is None
    assert actor.opponent_policy_id_by_env.tolist() == [MIRROR_OPPONENT_POLICY_ID] * 8
    assert HEURISTIC_PUBLIC_POLICY_ID not in runtime_any._opponent_heuristic_policies
    assert runtime_any._forced_fixed_opponent_policy_ids == ()


def test_structured_warmstart_source_mix_process_collectors_pushes_fixed_sources() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=2,
        envs_per_actor=8,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=11,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._collector_result_queue = object()
    runtime_any._teacher_policy = object()
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._opponent_models = {NOLEAGUE_BASELINE_POLICY_ID: torch.nn.Linear(1, 1)}
    runtime_any._forced_fixed_opponent_policy_ids = ()
    runtime_any._actors = []

    class _FakeControlQueue:
        def __init__(self) -> None:
            self.commands: list[dict[str, Any]] = []

        def put(self, payload: dict[str, Any]) -> None:
            self.commands.append(payload)

    control_queues = [_FakeControlQueue(), _FakeControlQueue()]
    runtime_any._collector_control_queues = control_queues

    with QueueRuntime.structured_warmstart_source_mix(runtime) as metrics:
        assert metrics["structured_warmstart_source_count"] == 3.0
        assert metrics["structured_warmstart_self_play_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b1_envs_per_actor"] == 3.0
        assert metrics["structured_warmstart_b2_envs_per_actor"] == 2.0
        assert runtime_any._forced_fixed_opponent_policy_ids == (
            NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )

    for control_queue in control_queues:
        assert len(control_queue.commands) == 2
        apply_payload, restore_payload = control_queue.commands
        assert apply_payload["kind"] == "set_fixed_opponents"
        assert apply_payload["restore_defaults"] is False
        assert apply_payload["activate_teacher_heuristic"] is True
        assert tuple(apply_payload["forced_policy_ids"]) == (
            NOLEAGUE_BASELINE_POLICY_ID,
            HEURISTIC_PUBLIC_POLICY_ID,
        )
        assert apply_payload["fixed_opponent_policy_id_by_env"].count(NOLEAGUE_BASELINE_POLICY_ID) == 3
        assert apply_payload["fixed_opponent_policy_id_by_env"].count(HEURISTIC_PUBLIC_POLICY_ID) == 2
        assert apply_payload["fixed_opponent_policy_id_by_env"].count("") == 3
        assert isinstance(apply_payload["noleague_baseline_state_dict"], dict)
        assert restore_payload == {"kind": "set_fixed_opponents", "restore_defaults": True}


def test_structured_warmstart_source_mix_restores_actor_slots_after_exception() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = QueueRuntimeConfig(
        mode="train_ordered",
        actor_count=1,
        envs_per_actor=4,
        unroll_length=1,
        batch_unrolls_per_update=2,
        queue_capacity_unrolls=2,
        profile="fast",
        base_seed=17,
        pass_action_id=51,
        actor_reload_interval_updates=1,
    )
    runtime_any._collector_result_queue = None
    teacher_policy = object()
    runtime_any._teacher_policy = teacher_policy
    runtime_any._opponent_heuristic_policies = {}
    runtime_any._opponent_models = {NOLEAGUE_BASELINE_POLICY_ID: object()}
    runtime_any._device = torch.device("cpu")
    runtime_any._forced_fixed_opponent_policy_ids = ("preexisting",)

    reset_calls: list[np.ndarray | None] = []

    def fake_reset_actor_state_for_fixed_opponents(actor) -> None:
        fixed = actor.fixed_opponent_policy_id_by_env
        reset_calls.append(None if fixed is None else np.asarray(fixed, dtype=object).copy())

    runtime_any._reset_actor_state_for_fixed_opponents = fake_reset_actor_state_for_fixed_opponents
    saved_slots = np.asarray(["existing", ""], dtype=object)
    actor = cast(
        Any,
        SimpleNamespace(
            fixed_opponent_policy_id_by_env=saved_slots.copy(),
        ),
    )
    runtime_any._actors = [actor]

    with pytest.raises(RuntimeError, match="boom"):
        with QueueRuntime.structured_warmstart_source_mix(runtime):
            assert HEURISTIC_PUBLIC_POLICY_ID in runtime_any._opponent_heuristic_policies
            assert actor.fixed_opponent_policy_id_by_env is not None
            raise RuntimeError("boom")

    assert np.array_equal(actor.fixed_opponent_policy_id_by_env, saved_slots)
    assert runtime_any._forced_fixed_opponent_policy_ids == ("preexisting",)
    assert HEURISTIC_PUBLIC_POLICY_ID not in runtime_any._opponent_heuristic_policies
    assert len(reset_calls) == 2
    assert reset_calls[0] is not None
    assert reset_calls[1] is not None
    assert np.array_equal(reset_calls[1], saved_slots)
