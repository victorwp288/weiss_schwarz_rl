from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weiss_rl.diagnostics.action_diagnostics import ActionSequenceState, make_action_sequence_state
from weiss_rl.runtime.components.counters import collector_counter_template


@dataclass(slots=True)
class CollectorUnrollState:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    to_play_seat: np.ndarray
    behavior_logp: np.ndarray
    values: np.ndarray
    episode_seed: np.ndarray
    policy_train_mask: np.ndarray
    opponent_context_index: np.ndarray
    teacher_family: np.ndarray
    teacher_slot: np.ndarray
    teacher_move_source: np.ndarray
    teacher_attack_type: np.ndarray
    teacher_action: np.ndarray
    teacher_valid: np.ndarray
    trajectory_retention_valid: np.ndarray | None
    packed_ids: list[np.ndarray]
    packed_meta: list[np.ndarray]
    packed_offsets: list[np.ndarray]
    mask_steps: list[np.ndarray]
    initial_hidden_state: np.ndarray
    counters: dict[str, int]
    action_sequence_state: ActionSequenceState


def allocate_collector_unroll_state(
    *,
    time_steps: int,
    batch_size: int,
    observation_dim: int,
    obs_dtype: Any,
    seat_hidden: Any,
    trajectory_retention_enabled: bool,
) -> CollectorUnrollState:
    shape = (int(time_steps), int(batch_size))
    obs_shape = (*shape, int(observation_dim))
    return CollectorUnrollState(
        obs=np.zeros(obs_shape, dtype=obs_dtype),
        actions=np.zeros(shape, dtype=np.uint16),
        rewards=np.zeros(shape, dtype=np.float32),
        terminated=np.zeros(shape, dtype=np.bool_),
        truncated=np.zeros(shape, dtype=np.bool_),
        to_play_seat=np.zeros(shape, dtype=np.int8),
        behavior_logp=np.zeros(shape, dtype=np.float32),
        values=np.zeros(shape, dtype=np.float32),
        episode_seed=np.zeros(shape, dtype=np.uint64),
        policy_train_mask=np.zeros(shape, dtype=np.bool_),
        opponent_context_index=np.zeros(shape, dtype=np.int16),
        teacher_family=np.full(shape, -1, dtype=np.int32),
        teacher_slot=np.full(shape, -1, dtype=np.int32),
        teacher_move_source=np.full(shape, -1, dtype=np.int32),
        teacher_attack_type=np.full(shape, -1, dtype=np.int32),
        teacher_action=np.full(shape, -1, dtype=np.int32),
        teacher_valid=np.zeros(shape, dtype=np.bool_),
        trajectory_retention_valid=(np.zeros(shape, dtype=np.bool_) if bool(trajectory_retention_enabled) else None),
        packed_ids=[],
        packed_meta=[],
        packed_offsets=[np.array([0], dtype=np.uint32)],
        mask_steps=[],
        initial_hidden_state=np.asarray(seat_hidden.detach().cpu().numpy()).copy(),
        counters=collector_counter_template(),
        action_sequence_state=make_action_sequence_state(batch_size),
    )
