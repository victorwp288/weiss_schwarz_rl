"""Action execution helpers for one central actor step."""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import numpy as np

from weiss_rl.runtime.components.collection.central_actor_action_context import (
    CentralActorActionInputs,
    MaskCentralActorActionCallbacks,
    PackedCentralActorActionCallbacks,
    PackedCentralActorActionMode,
)
from weiss_rl.runtime.components.collector_action_execution import (
    fused_step_packed_from_logits_with_logp,
    record_mask_action_summary,
    record_packed_action_summary,
    sample_and_step_mask_from_logits,
    sample_and_step_packed_from_logits,
    step_env_with_actions,
)
from weiss_rl.runtime.components.collector_step_legal import (
    capture_mask_step_legal,
    capture_packed_step_legal,
)
from weiss_rl.runtime.components.reward_shaping import apply_collector_reward_shaping
from weiss_rl.runtime.components.teacher_labels import TeacherLabelArrays


class CentralActorActionResult(NamedTuple):
    next_batch: Any
    actions: np.ndarray
    behavior_logp: np.ndarray
    rewards: np.ndarray
    teacher_labels: TeacherLabelArrays


def execute_packed_central_actor_action(
    *,
    inputs: CentralActorActionInputs,
    mode: PackedCentralActorActionMode,
    callbacks: PackedCentralActorActionCallbacks,
) -> CentralActorActionResult:
    state = inputs.state
    packed_legal = capture_packed_step_legal(
        batch=inputs.batch,
        focal_rows=inputs.focal_rows,
        obs_step=inputs.obs_step,
        counters=state.counters,
        ensure_legal_action_meta=callbacks.ensure_legal_action_meta,
        teacher_labels_from_ids=callbacks.teacher_labels_from_ids,
        packed_ids=state.packed_ids,
        packed_meta=state.packed_meta,
        packed_offsets=state.packed_offsets,
    )
    if mode.structured_central_packed:
        if mode.structured_action_steps is None or mode.structured_logp_steps is None:
            raise ValueError("structured central packed execution requires action and logp steps")
        action_step = np.asarray(mode.structured_action_steps[int(mode.actor_index)], dtype=np.int64)
        logp_step = np.asarray(mode.structured_logp_steps[int(mode.actor_index)], dtype=np.float32)
        next_batch = step_env_with_actions(
            env=inputs.actor.env,
            actions=action_step,
            counters=state.counters,
        )
    elif hasattr(inputs.actor.env, "step_sample_from_logits_with_logp"):
        executed = fused_step_packed_from_logits_with_logp(
            env=inputs.actor.env,
            logits=cast(np.ndarray, inputs.logits_step),
            rng=inputs.actor.rng,
            counters=state.counters,
            temperature=float(getattr(inputs.config, "actor_sampling_temperature", 1.0)),
        )
        next_batch = executed.next_batch
        action_step = executed.actions
        logp_step = executed.logp
    else:
        executed = sample_and_step_packed_from_logits(
            env=inputs.actor.env,
            logits=cast(np.ndarray, inputs.logits_step),
            legal_ids=packed_legal.legal_ids,
            legal_offsets=packed_legal.legal_offsets,
            rng=inputs.actor.rng,
            counters=state.counters,
            pass_action_id=inputs.config.pass_action_id,
            temperature=float(getattr(inputs.config, "actor_sampling_temperature", 1.0)),
        )
        next_batch = executed.next_batch
        action_step = executed.actions
        logp_step = executed.logp
    record_packed_action_summary(
        counters=state.counters,
        state=state.action_sequence_state,
        actions=action_step,
        legal_ids=packed_legal.reward_legal_ids,
        legal_offsets=packed_legal.reward_legal_offsets,
        pass_action_id=inputs.config.pass_action_id,
        next_batch=next_batch,
    )
    reward_step = apply_collector_reward_shaping(
        np.asarray(next_batch.reward, dtype=np.float32),
        np.asarray(action_step, dtype=np.int64),
        counters=state.counters,
        pass_action_id=inputs.config.pass_action_id,
        pass_with_nonpass_penalty=float(getattr(inputs.config, "pass_with_nonpass_penalty", 0.0)),
        mulligan_select_with_confirm_penalty=float(getattr(inputs.config, "mulligan_select_with_confirm_penalty", 0.0)),
        action_family_index=inputs.action_family_index,
        legal_ids=packed_legal.reward_legal_ids,
        legal_offsets=packed_legal.reward_legal_offsets,
        legal_action_meta=packed_legal.legal_action_meta,
    )
    return CentralActorActionResult(
        next_batch=next_batch,
        actions=action_step,
        behavior_logp=logp_step,
        rewards=reward_step,
        teacher_labels=packed_legal.teacher_labels,
    )


def execute_mask_central_actor_action(
    *,
    inputs: CentralActorActionInputs,
    callbacks: MaskCentralActorActionCallbacks,
) -> CentralActorActionResult:
    state = inputs.state
    mask_legal = capture_mask_step_legal(
        batch=inputs.batch,
        focal_rows=inputs.focal_rows,
        obs_step=inputs.obs_step,
        counters=state.counters,
        teacher_labels_from_mask=callbacks.teacher_labels_from_mask,
        mask_steps=state.mask_steps,
    )
    executed = sample_and_step_mask_from_logits(
        env=inputs.actor.env,
        logits=cast(np.ndarray, inputs.logits_step),
        legal_mask=mask_legal.legal_mask,
        rng=inputs.actor.rng,
        counters=state.counters,
        pass_action_id=inputs.config.pass_action_id,
        temperature=float(getattr(inputs.config, "actor_sampling_temperature", 1.0)),
    )
    next_batch = executed.next_batch
    action_step = executed.actions
    logp_step = executed.logp
    record_mask_action_summary(
        counters=state.counters,
        state=state.action_sequence_state,
        actions=action_step,
        legal_mask=mask_legal.reward_legal_mask,
        pass_action_id=inputs.config.pass_action_id,
        next_batch=next_batch,
    )
    reward_step = apply_collector_reward_shaping(
        np.asarray(next_batch.reward, dtype=np.float32),
        np.asarray(action_step, dtype=np.int64),
        counters=state.counters,
        pass_action_id=inputs.config.pass_action_id,
        pass_with_nonpass_penalty=float(getattr(inputs.config, "pass_with_nonpass_penalty", 0.0)),
        mulligan_select_with_confirm_penalty=float(getattr(inputs.config, "mulligan_select_with_confirm_penalty", 0.0)),
        action_family_index=inputs.action_family_index,
        legal_mask=mask_legal.reward_legal_mask,
    )
    return CentralActorActionResult(
        next_batch=next_batch,
        actions=action_step,
        behavior_logp=logp_step,
        rewards=reward_step,
        teacher_labels=mask_legal.teacher_labels,
    )
