from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import torch
from weiss_rl.runtime import QueueRuntime


def test_collect_all_heuristic_ids_fast_snapshots_reused_step_out_before_step() -> None:
    class ReusingStepOut:
        def __init__(self) -> None:
            self.step_index = 0
            self.obs = np.zeros((2, 4), dtype=np.float32)
            self.actor = np.zeros((2,), dtype=np.int8)
            self.decision_kind = np.zeros((2,), dtype=np.int32)
            self.decision_id = np.zeros((2,), dtype=np.uint32)
            self.legal_ids = np.zeros((4,), dtype=np.uint32)
            self.legal_offsets = np.array([0, 2, 4], dtype=np.uint32)
            self.rewards = np.zeros((2,), dtype=np.float32)
            self.terminated = np.zeros((2,), dtype=np.bool_)
            self.truncated = np.zeros((2,), dtype=np.bool_)
            self.engine_status = np.zeros((2,), dtype=np.int32)
            self.main_move_action = np.zeros((2,), dtype=np.bool_)
            self.main_pass_action = np.zeros((2,), dtype=np.bool_)
            self.fill(0)

        def fill(self, step_index: int) -> None:
            self.step_index = step_index
            self.obs[:] = np.float32(10 + step_index * 7) + np.arange(2, dtype=np.float32)[:, None]
            self.actor[:] = ((np.arange(2, dtype=np.int32) + step_index) % 2).astype(np.int8)
            self.decision_kind[:] = np.int32(20 + step_index)
            self.legal_ids[:] = np.array(
                [
                    (step_index * 4) % 64,
                    (step_index * 4 + 1) % 64,
                    (step_index * 4 + 2) % 64,
                    (step_index * 4 + 3) % 64,
                ],
                dtype=np.uint32,
            )
            self.rewards[:] = np.float32(step_index)
            self.terminated[:] = False
            self.truncated[:] = False
            self.engine_status[:] = 0

    class ReusingPool:
        def __init__(self, step_out: ReusingStepOut) -> None:
            self.step_out = step_out

        def step_into_i16_legal_ids(self, actions: np.ndarray, step_out: ReusingStepOut) -> None:
            assert step_out is self.step_out
            assert actions.shape == (2,)
            step_out.fill(step_out.step_index + 1)

        def reset_done_into_i16_legal_ids(self, done: np.ndarray, step_out: ReusingStepOut) -> None:
            assert not np.any(done)
            assert step_out is self.step_out

        def episode_seed_batch(self) -> np.ndarray:
            return np.array(
                [30_000 + self.step_out.step_index * 10, 30_001 + self.step_out.step_index * 10], dtype=np.uint64
            )

    step_out = ReusingStepOut()
    pool = ReusingPool(step_out)
    env = SimpleNamespace(
        pool=pool,
        _step_out=step_out,
        _record_python_timing=lambda *args, **kwargs: None,
        _handle_engine_status=lambda *args, **kwargs: None,
    )
    initial_batch = SimpleNamespace(
        obs=step_out.obs,
        actor=step_out.actor,
        decision_kind=step_out.decision_kind,
        ids_offsets=(step_out.legal_ids, step_out.legal_offsets),
        legal_action_meta=None,
    )
    actor = SimpleNamespace(
        actor_id=3,
        next_unroll_seq=0,
        snapshot_version=0,
        current_batch=initial_batch,
        env=env,
        focal_seat_by_env=np.array([0, 1], dtype=np.int8),
        seat_hidden=torch.zeros((2, 1, 2), dtype=torch.float32),
        opponent_hidden=torch.zeros((2, 1, 2), dtype=torch.float32),
    )

    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any.config = SimpleNamespace(unroll_length=2, envs_per_actor=2, pass_action_id=63)
    runtime_any.observation_dim = 4
    runtime_any.action_dim = 64
    runtime_any._actor_behavior_values_required = False
    runtime_any._teacher_policy = object()
    runtime_any._teacher_guidance_active_for_collection = lambda: False
    runtime_any._policy_train_mask_for_actor = lambda *, actor, focal_rows, **_kwargs: np.asarray(
        focal_rows, dtype=np.bool_
    )
    runtime_any._ensure_legal_action_meta = lambda legal_ids, meta: None
    runtime_any._should_track_heuristic_actor_hidden_state = lambda: False
    runtime_any._heuristic_public_actions_from_ids = lambda **kwargs: np.array(
        [
            int(kwargs["legal_ids"][int(kwargs["legal_offsets"][row])])
            for row in np.asarray(kwargs["row_indices"], dtype=np.int64)
        ],
        dtype=np.int64,
    )
    runtime_any._maybe_debug_validate_sampled_packed_actions = lambda **kwargs: None
    runtime_any._sync_actor_batch_from_step_out = lambda *, actor, step_out, pool: SimpleNamespace(
        obs=np.array(step_out.obs, copy=True),
        actor=np.array(step_out.actor, copy=True),
    )

    unroll = QueueRuntime._collect_actor_unroll_all_heuristic_ids_fast(runtime, actor)

    npt.assert_array_equal(
        unroll.obs[0],
        np.repeat(np.array([[10.0], [11.0]], dtype=np.float32), 4, axis=1),
    )
    npt.assert_array_equal(
        unroll.obs[1],
        np.repeat(np.array([[17.0], [18.0]], dtype=np.float32), 4, axis=1),
    )
    npt.assert_array_equal(unroll.to_play_seat[0], np.array([0, 1], dtype=np.int8))
    npt.assert_array_equal(unroll.to_play_seat[1], np.array([1, 0], dtype=np.int8))
    npt.assert_array_equal(unroll.episode_seed[0], np.array([30_010, 30_011], dtype=np.uint64))
    npt.assert_array_equal(unroll.episode_seed[1], np.array([30_020, 30_021], dtype=np.uint64))
    assert unroll.legal_actions.ids is not None
    assert unroll.legal_actions.offsets is not None
    npt.assert_array_equal(unroll.legal_actions.ids, np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint32))
    npt.assert_array_equal(unroll.legal_actions.offsets, np.array([0, 2, 4, 6, 8], dtype=np.uint32))
