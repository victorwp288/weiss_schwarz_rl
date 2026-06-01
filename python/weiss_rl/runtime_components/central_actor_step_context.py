"""Typed context objects for central actor-step execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CentralActorStepPolicyInputs:
    actor_index: int
    logits_step: np.ndarray | None
    value_step: np.ndarray
    structured_central_packed: bool
    structured_action_steps: Sequence[np.ndarray] | None
    structured_logp_steps: Sequence[np.ndarray] | None


@dataclass(frozen=True, slots=True)
class CentralActorStepCallbacks:
    policy_train_mask_for_actor: Callable[..., np.ndarray]
    trajectory_retention_mask_for_actor: Callable[..., np.ndarray | None]
    ensure_legal_action_meta: Callable[..., np.ndarray | None]
    teacher_labels_from_ids: Callable[..., Any]
    teacher_labels_from_mask: Callable[..., Any]
    update_outcomes: Callable[..., None]
    assign_episode_roles: Callable[..., None]
    reset_done_rows: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class CentralActorStepRuntimeContext:
    config: Any
    action_family_index: Mapping[str, int] | None
    device: Any
    timeout_limits: Any
    callbacks: CentralActorStepCallbacks


@dataclass(frozen=True, slots=True)
class CentralActorStepInputs:
    step_index: int
    obs_storage_step: np.ndarray
    actor_step: np.ndarray
    policy: CentralActorStepPolicyInputs


__all__ = [
    "CentralActorStepCallbacks",
    "CentralActorStepInputs",
    "CentralActorStepPolicyInputs",
    "CentralActorStepRuntimeContext",
]
