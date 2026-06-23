from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from weiss_rl.runtime.components.collection.central_actor_step_context import (
    CentralActorStepCallbacks,
    CentralActorStepInputs,
    CentralActorStepPolicyInputs,
    CentralActorStepRuntimeContext,
)
from weiss_rl.runtime.components.collector_state import allocate_collector_unroll_state
from weiss_rl.runtime.components.teacher_labels import teacher_label_arrays


class StepEnv:
    def __init__(self, next_batch: SimpleNamespace) -> None:
        self.next_batch = next_batch
        self.actions: np.ndarray | None = None

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self.actions = np.array(actions, copy=True)
        return self.next_batch


class ContextModel:
    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
        opponent_policy_ids: object | None = None,
    ) -> torch.Tensor:
        fill = 7.0 if opponent_policy_ids is not None else 3.0
        return torch.full((int(batch_size), 2), fill, dtype=torch.float32, device=device)


def central_step_state(*, trajectory_retention_enabled: bool = True) -> Any:
    return allocate_collector_unroll_state(
        time_steps=2,
        batch_size=2,
        observation_dim=2,
        obs_dtype=np.float16,
        seat_hidden=torch.zeros((2, 2), dtype=torch.float32),
        trajectory_retention_enabled=trajectory_retention_enabled,
    )


def central_step_config() -> SimpleNamespace:
    return SimpleNamespace(
        pass_action_id=0,
        actor_sampling_temperature=1.0,
        pass_with_nonpass_penalty=0.25,
        mulligan_select_with_confirm_penalty=0.0,
    )


def central_step_packed_batch() -> SimpleNamespace:
    return SimpleNamespace(
        ids_offsets=(
            np.asarray([0, 1, 0, 2], dtype=np.uint32),
            np.asarray([0, 2, 4], dtype=np.uint32),
        ),
        legal_action_meta=np.asarray([[0], [5], [0], [6]], dtype=np.uint16),
        decision_kind=np.asarray([1, 1], dtype=np.int32),
        mask=None,
    )


def central_step_next_batch(*, terminated: list[bool] | None = None) -> SimpleNamespace:
    terminated_array = np.asarray([False, False] if terminated is None else terminated, dtype=np.bool_)
    return SimpleNamespace(
        reward=np.asarray([1.0, 1.0], dtype=np.float32),
        terminated=terminated_array,
        truncated=np.asarray([False, False], dtype=np.bool_),
        episode_seed=np.asarray([101, 202], dtype=np.uint64),
        main_move_action=np.asarray([False, True], dtype=np.bool_),
        engine_status=np.asarray([0, 0], dtype=np.int64),
        decision_count=np.asarray([3, 4], dtype=np.int32),
        tick_count=np.asarray([30, 40], dtype=np.int32),
        no_progress_count=np.asarray([0, 0], dtype=np.int32),
    )


def central_step_actor(env: StepEnv) -> SimpleNamespace:
    return SimpleNamespace(
        actor_id=0,
        layout_name="i16_legal_ids",
        env=env,
        rng=np.random.default_rng(11),
        focal_seat_by_env=np.asarray([0, 1], dtype=np.int64),
        model=ContextModel(),
        seat_hidden=torch.zeros((2, 2), dtype=torch.float32),
        opponent_hidden=torch.zeros((2, 2), dtype=torch.float32),
        opponent_policy_id_by_env=np.asarray(["policy_a", "policy_b"], dtype=object),
    )


def central_step_inputs(
    *,
    step_index: int,
    actor_index: int = 0,
    obs_storage_step: np.ndarray | None = None,
    actor_step: np.ndarray | None = None,
    logits_step: np.ndarray | None = None,
    value_step: np.ndarray | None = None,
    structured_central_packed: bool = True,
    structured_action_steps: list[np.ndarray] | None = None,
    structured_logp_steps: list[np.ndarray] | None = None,
) -> CentralActorStepInputs:
    return CentralActorStepInputs(
        step_index=step_index,
        obs_storage_step=(
            np.ones((2, 2), dtype=np.float16) if obs_storage_step is None else np.asarray(obs_storage_step)
        ),
        actor_step=(
            np.asarray([0, 0], dtype=np.int64) if actor_step is None else np.asarray(actor_step, dtype=np.int64)
        ),
        policy=CentralActorStepPolicyInputs(
            actor_index=actor_index,
            logits_step=logits_step,
            value_step=(
                np.asarray([0.0, 0.0], dtype=np.float32)
                if value_step is None
                else np.asarray(value_step, dtype=np.float32)
            ),
            structured_central_packed=structured_central_packed,
            structured_action_steps=structured_action_steps,
            structured_logp_steps=structured_logp_steps,
        ),
    )


def central_step_runtime_context(
    *,
    action_family_index: dict[str, int] | None = None,
    policy_train_mask_for_actor: Any | None = None,
    trajectory_retention_mask_for_actor: Any | None = None,
    ensure_legal_action_meta: Any | None = None,
    teacher_labels_from_ids: Any | None = None,
    teacher_labels_from_mask: Any | None = None,
    update_outcomes: Any | None = None,
    assign_episode_roles: Any | None = None,
    reset_done_rows: Any | None = None,
) -> CentralActorStepRuntimeContext:
    return CentralActorStepRuntimeContext(
        config=central_step_config(),
        action_family_index={} if action_family_index is None else action_family_index,
        device=torch.device("cpu"),
        timeout_limits={"max_decisions": None, "max_ticks": None, "max_no_progress_decisions": None},
        callbacks=CentralActorStepCallbacks(
            policy_train_mask_for_actor=(
                (lambda *, actor, focal_rows: focal_rows)
                if policy_train_mask_for_actor is None
                else policy_train_mask_for_actor
            ),
            trajectory_retention_mask_for_actor=(
                (lambda *, actor, focal_rows: None)
                if trajectory_retention_mask_for_actor is None
                else trajectory_retention_mask_for_actor
            ),
            ensure_legal_action_meta=(
                (lambda legal_ids, legal_action_meta: legal_action_meta)
                if ensure_legal_action_meta is None
                else ensure_legal_action_meta
            ),
            teacher_labels_from_ids=(
                (lambda **_: teacher_label_arrays(2)) if teacher_labels_from_ids is None else teacher_labels_from_ids
            ),
            teacher_labels_from_mask=(
                (lambda **_: teacher_label_arrays(2)) if teacher_labels_from_mask is None else teacher_labels_from_mask
            ),
            update_outcomes=(lambda **_: None) if update_outcomes is None else update_outcomes,
            assign_episode_roles=(lambda *_, **__: None) if assign_episode_roles is None else assign_episode_roles,
            reset_done_rows=(lambda *_: None) if reset_done_rows is None else reset_done_rows,
        ),
    )
