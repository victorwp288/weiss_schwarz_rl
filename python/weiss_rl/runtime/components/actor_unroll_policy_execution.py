"""Packed and dense-mask policy execution for generic actor unrolls."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from weiss_rl.runtime.components.collector_action_execution import (
    fused_step_mask_from_logits,
    fused_step_packed_from_logits_with_logp,
    record_mask_action_summary,
    record_packed_action_summary,
    step_env_with_actions,
)
from weiss_rl.runtime.components.collector_step_legal import (
    capture_mask_step_legal,
    capture_packed_step_legal,
)
from weiss_rl.runtime.components.teacher_labels import TeacherLabelArrays


@dataclass(frozen=True, slots=True)
class ActorPolicyExecutionInputs:
    actor: Any
    batch: Any
    obs_step: np.ndarray
    actor_step: np.ndarray
    focal_rows: np.ndarray
    value_step: np.ndarray
    action_step: np.ndarray
    logp_step: np.ndarray
    logits_step: np.ndarray
    config: Any
    counters: dict[str, int]
    action_sequence_state: Any
    use_simulator_fused_logits_step: bool


@dataclass(frozen=True, slots=True)
class PackedActorPolicyCallbacks:
    fill_policy_outputs_ids: Callable[..., None]
    maybe_debug_validate_env_step_packed_actions: Callable[..., None]
    ensure_legal_action_meta: Callable[..., np.ndarray | None]
    teacher_labels_from_ids: Callable[..., TeacherLabelArrays]


@dataclass(frozen=True, slots=True)
class PackedActorPolicyStorage:
    packed_ids: list[np.ndarray]
    packed_meta: list[np.ndarray]
    packed_offsets: list[np.ndarray]


@dataclass(frozen=True, slots=True)
class MaskActorPolicyCallbacks:
    fill_policy_outputs_mask: Callable[..., None]
    teacher_labels_from_mask: Callable[..., TeacherLabelArrays]


@dataclass(frozen=True, slots=True)
class MaskActorPolicyStorage:
    mask_steps: list[np.ndarray]


class ActorPolicyExecutionResult(NamedTuple):
    next_batch: Any
    action_step: np.ndarray
    logp_step: np.ndarray
    teacher_labels: TeacherLabelArrays
    reward_legal_ids: np.ndarray | None
    reward_legal_offsets: np.ndarray | None
    reward_legal_meta: np.ndarray | None
    reward_legal_mask: np.ndarray | None


def execute_generic_packed_actor_policy(
    *,
    inputs: ActorPolicyExecutionInputs,
    callbacks: PackedActorPolicyCallbacks,
    storage: PackedActorPolicyStorage,
) -> ActorPolicyExecutionResult:
    packed_legal = capture_packed_step_legal(
        batch=inputs.batch,
        focal_rows=inputs.focal_rows,
        obs_step=inputs.obs_step,
        counters=inputs.counters,
        ensure_legal_action_meta=callbacks.ensure_legal_action_meta,
        teacher_labels_from_ids=callbacks.teacher_labels_from_ids,
        packed_ids=storage.packed_ids,
        packed_meta=storage.packed_meta,
        packed_offsets=storage.packed_offsets,
    )
    if inputs.use_simulator_fused_logits_step and hasattr(inputs.actor.env, "step_sample_from_logits_with_logp"):
        policy_started = time.perf_counter()
        callbacks.fill_policy_outputs_ids(
            actor=inputs.actor,
            obs_step=inputs.obs_step,
            actor_step=inputs.actor_step,
            focal_rows=inputs.focal_rows,
            legal_ids=packed_legal.legal_ids,
            legal_offsets=packed_legal.legal_offsets,
            legal_action_meta=packed_legal.legal_action_meta,
            logits_out=inputs.logits_step,
            values_out=inputs.value_step,
            actions_out=None,
            logp_out=None,
            rng=inputs.actor.rng,
            sample_actions=False,
        )
        inputs.counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
        executed = fused_step_packed_from_logits_with_logp(
            env=inputs.actor.env,
            logits=inputs.logits_step,
            rng=inputs.actor.rng,
            counters=inputs.counters,
            temperature=float(getattr(inputs.config, "actor_sampling_temperature", 1.0)),
        )
        next_batch = executed.next_batch
        action_step = executed.actions
        logp_step = executed.logp
    else:
        policy_started = time.perf_counter()
        callbacks.fill_policy_outputs_ids(
            actor=inputs.actor,
            obs_step=inputs.obs_step,
            actor_step=inputs.actor_step,
            focal_rows=inputs.focal_rows,
            legal_ids=packed_legal.legal_ids,
            legal_offsets=packed_legal.legal_offsets,
            legal_action_meta=packed_legal.legal_action_meta,
            logits_out=None,
            values_out=inputs.value_step,
            actions_out=inputs.action_step,
            logp_out=inputs.logp_step,
            rng=inputs.actor.rng,
        )
        inputs.counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)

        def _validate_env_step_packed_actions() -> None:
            callbacks.maybe_debug_validate_env_step_packed_actions(
                actor=inputs.actor,
                source_label="collect:packed",
                actions=inputs.action_step,
                legal_ids=packed_legal.legal_ids,
                legal_offsets=packed_legal.legal_offsets,
            )

        next_batch = step_env_with_actions(
            env=inputs.actor.env,
            actions=inputs.action_step,
            counters=inputs.counters,
            before_step=_validate_env_step_packed_actions,
        )
        action_step = inputs.action_step
        logp_step = inputs.logp_step

    record_packed_action_summary(
        counters=inputs.counters,
        state=inputs.action_sequence_state,
        actions=action_step,
        legal_ids=packed_legal.reward_legal_ids,
        legal_offsets=packed_legal.reward_legal_offsets,
        pass_action_id=inputs.config.pass_action_id,
        next_batch=next_batch,
    )
    return ActorPolicyExecutionResult(
        next_batch=next_batch,
        action_step=action_step,
        logp_step=logp_step,
        teacher_labels=packed_legal.teacher_labels,
        reward_legal_ids=packed_legal.reward_legal_ids,
        reward_legal_offsets=packed_legal.reward_legal_offsets,
        reward_legal_meta=packed_legal.reward_legal_meta,
        reward_legal_mask=None,
    )


def execute_generic_mask_actor_policy(
    *,
    inputs: ActorPolicyExecutionInputs,
    callbacks: MaskActorPolicyCallbacks,
    storage: MaskActorPolicyStorage,
) -> ActorPolicyExecutionResult:
    mask_legal = capture_mask_step_legal(
        batch=inputs.batch,
        focal_rows=inputs.focal_rows,
        obs_step=inputs.obs_step,
        counters=inputs.counters,
        teacher_labels_from_mask=callbacks.teacher_labels_from_mask,
        mask_steps=storage.mask_steps,
    )
    if inputs.use_simulator_fused_logits_step:
        current_legal_mask = mask_legal.reward_legal_mask
        policy_started = time.perf_counter()
        callbacks.fill_policy_outputs_mask(
            actor=inputs.actor,
            obs_step=inputs.obs_step,
            actor_step=inputs.actor_step,
            focal_rows=inputs.focal_rows,
            legal_mask=current_legal_mask,
            logits_out=inputs.logits_step,
            values_out=inputs.value_step,
            actions_out=None,
            logp_out=None,
            rng=inputs.actor.rng,
            sample_actions=False,
        )
        inputs.counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
        executed = fused_step_mask_from_logits(
            env=inputs.actor.env,
            logits=inputs.logits_step,
            legal_mask=current_legal_mask,
            rng=inputs.actor.rng,
            counters=inputs.counters,
            pass_action_id=inputs.config.pass_action_id,
            temperature=float(getattr(inputs.config, "actor_sampling_temperature", 1.0)),
        )
        next_batch = executed.next_batch
        action_step = executed.actions
        logp_step = executed.logp
    else:
        current_legal_mask = mask_legal.reward_legal_mask
        policy_started = time.perf_counter()
        callbacks.fill_policy_outputs_mask(
            actor=inputs.actor,
            obs_step=inputs.obs_step,
            actor_step=inputs.actor_step,
            focal_rows=inputs.focal_rows,
            legal_mask=mask_legal.legal_mask,
            logits_out=None,
            values_out=inputs.value_step,
            actions_out=inputs.action_step,
            logp_out=inputs.logp_step,
            rng=inputs.actor.rng,
        )
        inputs.counters["actor_policy_forward_ms"] += int((time.perf_counter() - policy_started) * 1000.0)
        next_batch = step_env_with_actions(
            env=inputs.actor.env,
            actions=inputs.action_step,
            counters=inputs.counters,
        )
        action_step = inputs.action_step
        logp_step = inputs.logp_step

    record_mask_action_summary(
        counters=inputs.counters,
        state=inputs.action_sequence_state,
        actions=action_step,
        legal_mask=current_legal_mask,
        pass_action_id=inputs.config.pass_action_id,
        next_batch=next_batch,
    )
    return ActorPolicyExecutionResult(
        next_batch=next_batch,
        action_step=action_step,
        logp_step=logp_step,
        teacher_labels=mask_legal.teacher_labels,
        reward_legal_ids=None,
        reward_legal_offsets=None,
        reward_legal_meta=None,
        reward_legal_mask=current_legal_mask,
    )


__all__ = [
    "ActorPolicyExecutionInputs",
    "ActorPolicyExecutionResult",
    "MaskActorPolicyCallbacks",
    "MaskActorPolicyStorage",
    "PackedActorPolicyCallbacks",
    "PackedActorPolicyStorage",
    "execute_generic_mask_actor_policy",
    "execute_generic_packed_actor_policy",
]
