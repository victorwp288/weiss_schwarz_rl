from __future__ import annotations

import numpy as np
from weiss_rl.runtime.components.central.central_actor_step import execute_central_actor_step
from weiss_rl.runtime.components.teacher_labels import teacher_label_arrays

from .central_actor_step_test_support import (
    StepEnv,
    central_step_actor,
    central_step_inputs,
    central_step_next_batch,
    central_step_packed_batch,
    central_step_runtime_context,
    central_step_state,
)


def test_execute_central_actor_step_stores_structured_packed_step_outputs() -> None:
    state = central_step_state()
    env = StepEnv(central_step_next_batch())
    actor = central_step_actor(env)
    batch = central_step_packed_batch()
    teacher_labels = teacher_label_arrays(2)
    teacher_labels[4][:] = np.asarray([12, 13], dtype=np.int32)
    teacher_labels[5][:] = np.asarray([True, False], dtype=np.bool_)
    policy_masks: list[np.ndarray] = []
    retention_masks: list[np.ndarray] = []

    returned = execute_central_actor_step(
        actor=actor,
        batch=batch,
        state=state,
        inputs=central_step_inputs(
            step_index=1,
            obs_storage_step=np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16),
            actor_step=np.asarray([0, 0], dtype=np.int64),
            value_step=np.asarray([0.25, 0.75], dtype=np.float32),
            structured_action_steps=[np.asarray([0, 2], dtype=np.int64)],
            structured_logp_steps=[np.asarray([-0.1, -0.2], dtype=np.float32)],
        ),
        runtime=central_step_runtime_context(
            action_family_index={"main_move": 99},
            policy_train_mask_for_actor=lambda *, actor, focal_rows: (
                policy_masks.append(focal_rows.copy()) or focal_rows
            ),
            trajectory_retention_mask_for_actor=lambda *, actor, focal_rows: (
                retention_masks.append(np.asarray([False, True], dtype=np.bool_))
                or np.asarray([False, True], dtype=np.bool_)
            ),
            teacher_labels_from_ids=lambda **_: teacher_labels,
        ),
    )

    assert returned is env.next_batch
    assert env.actions is not None
    assert env.actions.dtype == np.uint32
    assert env.actions.tolist() == [0, 2]
    assert policy_masks[0].tolist() == [True, False]
    assert retention_masks[0].tolist() == [False, True]
    assert state.policy_train_mask[1].tolist() == [True, False]
    np.testing.assert_array_equal(state.obs[1], np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float16))
    assert state.actions[1].dtype == np.uint16
    assert state.actions[1].tolist() == [0, 2]
    np.testing.assert_allclose(state.rewards[1], np.asarray([0.75, 1.0], dtype=np.float32))
    np.testing.assert_allclose(state.behavior_logp[1], np.asarray([-0.1, -0.2], dtype=np.float32))
    np.testing.assert_allclose(state.values[1], np.asarray([0.25, 0.75], dtype=np.float32))
    assert state.episode_seed[1].tolist() == [101, 202]
    assert state.teacher_action[1].tolist() == [12, 13]
    assert state.teacher_valid[1].tolist() == [True, False]
    assert state.trajectory_retention_valid is not None
    assert state.trajectory_retention_valid[1].tolist() == [False, True]
    assert state.packed_ids[0].tolist() == [0, 1, 0, 2]
    assert state.packed_offsets[-1].tolist() == [2, 4]
    assert state.counters["pass_with_nonpass_penalty_count"] == 1
