"""Runtime data containers shared by queue-runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.shared import SharedPendingUnroll


@dataclass(frozen=True, slots=True)
class RuntimeUnroll:
    actor_id: int
    unroll_seq: int
    behavior_policy_version: int
    unroll_hash: str
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    values: np.ndarray
    legal_actions: LegalActionBatch
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray
    bootstrap_value: np.ndarray
    initial_hidden_state: np.ndarray
    final_hidden_state: np.ndarray
    episode_seed: np.ndarray
    policy_train_mask: np.ndarray
    opponent_context_index: np.ndarray | None = None
    teacher_family: np.ndarray | None = None
    teacher_slot: np.ndarray | None = None
    teacher_move_source: np.ndarray | None = None
    teacher_attack_type: np.ndarray | None = None
    teacher_action: np.ndarray | None = None
    teacher_valid: np.ndarray | None = None
    trajectory_retention_valid: np.ndarray | None = None
    behavior_logits: np.ndarray | None = None
    counters: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class RuntimeBatch:
    learner_batch: dict[str, Any]
    runtime_metrics: dict[str, float]


PendingUnroll = RuntimeUnroll | SharedPendingUnroll
