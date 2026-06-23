from __future__ import annotations

from typing import cast

import numpy as np
from weiss_rl.runtime.components.ipc_shared.shared_transport import (
    open_shared_collector_slot,
    read_unroll_from_shared_slot,
    shared_unroll_metadata,
    write_unroll_to_shared_slot,
)

from tests.weiss_rl.runtime_shared_memory_test_support import make_packed_shared_unroll, make_shared_slot_config


def test_shared_collector_slot_round_trip_preserves_packed_unroll_payload() -> None:
    slot_config = make_shared_slot_config(actor_id=0)
    slot = open_shared_collector_slot(slot_config, create=True)
    try:
        unroll = make_packed_shared_unroll(
            actor_id=0,
            unroll_seq=7,
            behavior_policy_version=11,
            unroll_hash="roundtrip",
            opponent_context_index=np.array([[1, 2], [3, 4]], dtype=np.int16),
            trajectory_retention_valid=np.array([[False, True], [True, False]], dtype=np.bool_),
        )

        write_unroll_to_shared_slot(slot, unroll)
        metadata = shared_unroll_metadata(unroll)
        restored = read_unroll_from_shared_slot(slot, metadata)

        assert metadata["has_trajectory_retention_label"] is True
        assert metadata["has_opponent_context_index"] is True
        assert restored.actor_id == unroll.actor_id
        assert restored.unroll_seq == unroll.unroll_seq
        assert restored.behavior_policy_version == unroll.behavior_policy_version
        assert np.array_equal(restored.obs, unroll.obs)
        assert np.array_equal(restored.actions, unroll.actions)
        assert np.array_equal(restored.bootstrap_obs, unroll.bootstrap_obs)
        assert np.array_equal(restored.bootstrap_value, unroll.bootstrap_value)
        assert np.array_equal(restored.final_hidden_state, unroll.final_hidden_state)
        assert np.array_equal(
            cast(np.ndarray, restored.opponent_context_index),
            cast(np.ndarray, unroll.opponent_context_index),
        )
        assert np.array_equal(cast(np.ndarray, restored.teacher_family), cast(np.ndarray, unroll.teacher_family))
        assert np.array_equal(cast(np.ndarray, restored.teacher_slot), cast(np.ndarray, unroll.teacher_slot))
        assert np.array_equal(
            cast(np.ndarray, restored.teacher_move_source),
            cast(np.ndarray, unroll.teacher_move_source),
        )
        assert np.array_equal(
            cast(np.ndarray, restored.teacher_attack_type),
            cast(np.ndarray, unroll.teacher_attack_type),
        )
        assert np.array_equal(cast(np.ndarray, restored.teacher_action), cast(np.ndarray, unroll.teacher_action))
        assert np.array_equal(cast(np.ndarray, restored.teacher_valid), cast(np.ndarray, unroll.teacher_valid))
        assert np.array_equal(
            cast(np.ndarray, restored.trajectory_retention_valid),
            cast(np.ndarray, unroll.trajectory_retention_valid),
        )
        assert restored.legal_actions.ids is not None
        assert restored.legal_actions.offsets is not None
        assert restored.legal_actions.action_space == 5
        assert restored.legal_actions.ids.tolist() == cast(np.ndarray, unroll.legal_actions.ids).tolist()
        assert restored.legal_actions.offsets.tolist() == cast(np.ndarray, unroll.legal_actions.offsets).tolist()
        assert restored.legal_actions.meta is not None
        assert restored.legal_actions.meta.tolist() == cast(np.ndarray, unroll.legal_actions.meta).tolist()
    finally:
        slot.close(unlink=True)
