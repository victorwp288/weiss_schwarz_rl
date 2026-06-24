from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from weiss_rl.runtime.components.central.central_actor_action import (
    execute_mask_central_actor_action,
    execute_packed_central_actor_action,
)
from weiss_rl.runtime.components.collection.central_actor_action_context import (
    CentralActorActionInputs,
    MaskCentralActorActionCallbacks,
    PackedCentralActorActionCallbacks,
    PackedCentralActorActionMode,
)
from weiss_rl.runtime.components.collection.collector_state import allocate_collector_unroll_state
from weiss_rl.runtime.components.teacher_labels import teacher_label_arrays


class _StepEnv:
    def __init__(self) -> None:
        self.actions: np.ndarray | None = None

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self.actions = np.array(actions, copy=True)
        return SimpleNamespace(
            reward=np.asarray([1.0, 1.0], dtype=np.float32),
            main_move_action=np.asarray([False, True], dtype=np.bool_),
        )


def _state() -> Any:
    return allocate_collector_unroll_state(
        time_steps=1,
        batch_size=2,
        observation_dim=2,
        obs_dtype=np.float32,
        seat_hidden=torch.zeros((2, 2), dtype=torch.float32),
        trajectory_retention_enabled=False,
    )


def _actor(env: _StepEnv) -> SimpleNamespace:
    return SimpleNamespace(env=env, rng=np.random.default_rng(17))


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        pass_action_id=0,
        actor_sampling_temperature=1.0,
        pass_with_nonpass_penalty=0.25,
        mulligan_select_with_confirm_penalty=0.0,
    )


def test_execute_packed_central_actor_action_uses_context_for_legal_capture_step_and_rewards() -> None:
    env = _StepEnv()
    state = _state()
    labels = teacher_label_arrays(2)
    labels[4][:] = np.asarray([8, 9], dtype=np.int32)
    labels[5][:] = np.asarray([True, False], dtype=np.bool_)
    teacher_calls: list[dict[str, Any]] = []

    result = execute_packed_central_actor_action(
        inputs=CentralActorActionInputs(
            actor=_actor(env),
            batch=SimpleNamespace(
                ids_offsets=(
                    np.asarray([0, 1, 0, 2], dtype=np.uint32),
                    np.asarray([0, 2, 4], dtype=np.uint32),
                ),
                legal_action_meta=np.asarray([[0], [5], [0], [6]], dtype=np.uint16),
                decision_kind=np.asarray([1, 1], dtype=np.int32),
            ),
            state=state,
            obs_step=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            focal_rows=np.asarray([True, False], dtype=np.bool_),
            logits_step=None,
            config=_config(),
            action_family_index={"main_move": 99},
        ),
        mode=PackedCentralActorActionMode(
            actor_index=0,
            structured_central_packed=True,
            structured_action_steps=[np.asarray([0, 2], dtype=np.int64)],
            structured_logp_steps=[np.asarray([-0.1, -0.2], dtype=np.float32)],
        ),
        callbacks=PackedCentralActorActionCallbacks(
            ensure_legal_action_meta=lambda legal_ids, legal_action_meta: legal_action_meta,
            teacher_labels_from_ids=lambda **kwargs: teacher_calls.append(kwargs) or labels,
        ),
    )

    assert env.actions is not None
    assert env.actions.dtype == np.uint32
    assert env.actions.tolist() == [0, 2]
    assert result.actions.tolist() == [0, 2]
    np.testing.assert_allclose(result.behavior_logp, np.asarray([-0.1, -0.2], dtype=np.float32))
    np.testing.assert_allclose(result.rewards, np.asarray([0.75, 1.0], dtype=np.float32))
    assert result.teacher_labels[4].tolist() == [8, 9]
    assert len(teacher_calls) == 1
    assert teacher_calls[0]["focal_rows"].tolist() == [True, False]
    assert state.packed_ids[0].tolist() == [0, 1, 0, 2]
    assert state.packed_offsets[-1].tolist() == [2, 4]
    assert state.counters["packed_candidate_count"] == 4
    assert state.counters["pass_with_nonpass_penalty_count"] == 1
    assert state.counters["total_actions"] == 2


def test_execute_mask_central_actor_action_uses_context_for_mask_sampling_and_rewards() -> None:
    env = _StepEnv()
    state = _state()
    labels = teacher_label_arrays(2)
    mask_calls: list[dict[str, Any]] = []

    result = execute_mask_central_actor_action(
        inputs=CentralActorActionInputs(
            actor=_actor(env),
            batch=SimpleNamespace(
                mask=np.asarray([[False, True, False], [False, False, True]], dtype=np.bool_),
                decision_kind=np.asarray([1, 2], dtype=np.int32),
            ),
            state=state,
            obs_step=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            focal_rows=np.asarray([True, True], dtype=np.bool_),
            logits_step=np.asarray([[0.0, 10.0, -1.0], [0.0, -1.0, 10.0]], dtype=np.float32),
            config=_config(),
            action_family_index={"main_move": 99},
        ),
        callbacks=MaskCentralActorActionCallbacks(
            teacher_labels_from_mask=lambda **kwargs: mask_calls.append(kwargs) or labels,
        ),
    )

    assert env.actions is not None
    assert env.actions.dtype == np.uint32
    assert env.actions.tolist() == [1, 2]
    assert result.actions.tolist() == [1, 2]
    np.testing.assert_allclose(result.rewards, np.asarray([1.0, 1.0], dtype=np.float32))
    assert result.teacher_labels is labels
    assert len(mask_calls) == 1
    assert mask_calls[0]["legal_mask"].tolist() == [[False, True, False], [False, False, True]]
    assert len(state.mask_steps) == 1
    assert state.mask_steps[0].tolist() == [[False, True, False], [False, False, True]]
    assert state.counters["total_actions"] == 2
