from __future__ import annotations

import numpy as np
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.ipc_shared.shared_transport import create_shared_collector_slot_config
from weiss_rl.runtime.components.types import RuntimeUnroll


def make_shared_slot_config(*, actor_id: int, slot_id: int = 0) -> dict[str, object]:
    return create_shared_collector_slot_config(
        actor_id=actor_id,
        slot_id=slot_id,
        profile="fast",
        unroll_length=2,
        envs_per_actor=2,
        observation_dim=3,
        action_dim=5,
        hidden_size=4,
        layout_name="i16_legal_ids",
    )


def make_packed_legal_actions() -> LegalActionBatch:
    return LegalActionBatch.from_packed(
        np.array([0, 1, 2, 3, 4, 1], dtype=np.uint32),
        np.array([0, 2, 3, 5, 6], dtype=np.uint32),
        meta=np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [2, 0, 0, 0],
                [3, 0, 1, 0],
                [3, 1, 1, 0],
                [8, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
        action_space=5,
    )


def make_packed_shared_unroll(
    *,
    actor_id: int,
    unroll_seq: int,
    behavior_policy_version: int,
    unroll_hash: str,
    opponent_context_index: np.ndarray,
    trajectory_retention_valid: np.ndarray,
) -> RuntimeUnroll:
    return RuntimeUnroll(
        actor_id=actor_id,
        unroll_seq=unroll_seq,
        behavior_policy_version=behavior_policy_version,
        unroll_hash=unroll_hash,
        obs=np.arange(12, dtype=np.int16).reshape(2, 2, 3),
        actions=np.array([[1, 2], [3, 4]], dtype=np.uint16),
        rewards=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        terminated=np.array([[False, True], [False, False]], dtype=np.bool_),
        truncated=np.array([[False, False], [True, False]], dtype=np.bool_),
        to_play_seat=np.array([[0, 1], [1, 0]], dtype=np.int8),
        behavior_logp=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        values=np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
        legal_actions=make_packed_legal_actions(),
        bootstrap_obs=np.arange(6, dtype=np.float32).reshape(2, 3),
        bootstrap_actor=np.array([0, 1], dtype=np.int64),
        bootstrap_value=np.array([0.25, -0.5], dtype=np.float32),
        initial_hidden_state=np.arange(16, dtype=np.float32).reshape(2, 2, 4),
        final_hidden_state=np.arange(16, 32, dtype=np.float32).reshape(2, 2, 4),
        episode_seed=np.array([[5, 6], [7, 8]], dtype=np.uint64),
        policy_train_mask=np.array([[True, False], [True, True]], dtype=np.bool_),
        opponent_context_index=opponent_context_index,
        teacher_family=np.array([[1, 2], [3, -1]], dtype=np.int32),
        teacher_slot=np.array([[0, -1], [2, -1]], dtype=np.int32),
        teacher_move_source=np.array([[-1, 1], [0, -1]], dtype=np.int32),
        teacher_attack_type=np.array([[-1, 1], [0, -1]], dtype=np.int32),
        teacher_action=np.array([[4, 9], [12, -1]], dtype=np.int32),
        teacher_valid=np.array([[True, True], [True, False]], dtype=np.bool_),
        trajectory_retention_valid=trajectory_retention_valid,
        behavior_logits=None,
    )
