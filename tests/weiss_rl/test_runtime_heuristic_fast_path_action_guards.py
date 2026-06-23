from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import torch
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.runtime import QueueRuntime


def test_collect_all_heuristic_ids_fast_applies_action_surface_guards() -> None:
    class StepOut:
        def __init__(self) -> None:
            self.obs = np.zeros((1, 4), dtype=np.float32)
            self.actor = np.zeros((1,), dtype=np.int8)
            self.decision_kind = np.zeros((1,), dtype=np.int32)
            self.decision_id = np.zeros((1,), dtype=np.uint32)
            self.legal_ids = np.array([51, 10], dtype=np.uint32)
            self.legal_offsets = np.array([0, 2], dtype=np.uint32)
            self.legal_action_meta = np.array([[2, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint16)
            self.rewards = np.zeros((1,), dtype=np.float32)
            self.terminated = np.zeros((1,), dtype=np.bool_)
            self.truncated = np.zeros((1,), dtype=np.bool_)
            self.engine_status = np.zeros((1,), dtype=np.int32)
            self.main_move_action = np.zeros((1,), dtype=np.bool_)
            self.main_pass_action = np.zeros((1,), dtype=np.bool_)

    captured_actions: list[int] = []

    class Pool:
        def step_into_i16_legal_ids(self, actions: np.ndarray, step_out: StepOut) -> None:
            captured_actions.append(int(actions[0]))
            step_out.obs[:] = 1.0

        def reset_done_into_i16_legal_ids(self, done: np.ndarray, step_out: StepOut) -> None:
            del done, step_out

        def episode_seed_batch(self) -> np.ndarray:
            return np.array([123], dtype=np.uint64)

    step_out = StepOut()
    env = SimpleNamespace(
        pool=Pool(),
        _step_out=step_out,
        _record_python_timing=lambda *args, **kwargs: None,
        _handle_engine_status=lambda *args, **kwargs: None,
    )
    initial_batch = DecisionBoundaryBatch(
        obs=step_out.obs,
        reward=step_out.rewards,
        terminated=step_out.terminated,
        truncated=step_out.truncated,
        to_play=step_out.actor,
        actor=step_out.actor,
        decision_id=np.zeros((1,), dtype=np.uint32),
        engine_status=step_out.engine_status,
        decision_count=np.zeros((1,), dtype=np.uint32),
        tick_count=np.zeros((1,), dtype=np.uint32),
        episode_seed=np.array([123], dtype=np.uint64),
        episode_key=np.array([456], dtype=np.uint64),
        decision_kind=step_out.decision_kind,
        ids_offsets=(step_out.legal_ids, step_out.legal_offsets),
        legal_action_meta=step_out.legal_action_meta,
    )
    actor = SimpleNamespace(
        actor_id=0,
        next_unroll_seq=0,
        snapshot_version=0,
        current_batch=initial_batch,
        env=env,
        focal_seat_by_env=np.array([0], dtype=np.int8),
        seat_hidden=torch.zeros((1, 1, 2), dtype=torch.float32),
        opponent_hidden=torch.zeros((1, 1, 2), dtype=torch.float32),
    )

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = SimpleNamespace(
        unroll_length=1,
        envs_per_actor=1,
        pass_action_id=51,
        force_attack_over_pass_when_attack_legal=True,
        force_pass_over_main_move_only=False,
        mulligan_force_confirm_after_select=False,
    )
    runtime_any.observation_dim = 4
    runtime_any.action_dim = 64
    runtime_any._actor_behavior_values_required = False
    runtime_any._teacher_policy = object()
    runtime_any._teacher_guidance_active_for_collection = lambda: False
    runtime_any._policy_train_mask_for_actor = lambda *, actor, focal_rows, **_kwargs: np.asarray(
        focal_rows, dtype=np.bool_
    )
    runtime_any._ensure_legal_action_meta = lambda legal_ids, meta: meta
    runtime_any._should_track_heuristic_actor_hidden_state = lambda: False
    runtime_any._action_family_index = {"attack": 1, "pass": 2}
    runtime_any._last_action_arg0_obs_index = -1
    runtime_any._heuristic_public_actions_from_ids = lambda **kwargs: np.array(
        [int(kwargs["legal_ids"][0])],
        dtype=np.int64,
    )
    runtime_any._maybe_debug_validate_sampled_packed_actions = lambda **kwargs: None
    runtime_any._sync_actor_batch_from_step_out = lambda *, actor, step_out, pool: SimpleNamespace(
        obs=np.array(step_out.obs, copy=True),
        actor=np.array(step_out.actor, copy=True),
    )

    unroll = QueueRuntime._collect_actor_unroll_all_heuristic_ids_fast(runtime, actor)

    assert captured_actions == [10]
    assert unroll.actions.tolist() == [[10]]
    assert unroll.legal_actions.ids is not None
    npt.assert_array_equal(unroll.legal_actions.ids, np.array([10], dtype=np.uint32))
    assert unroll.counters is not None
    assert unroll.counters["attack_available_force_attack_actions"] == 1
    assert unroll.counters["focal_row_count"] == 1
