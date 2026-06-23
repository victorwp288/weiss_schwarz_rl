from __future__ import annotations

import numpy as np
import numpy.testing as npt
from weiss_rl.runtime.components.policy_inference.heuristic_actor_outputs import (
    legal_action_ids_from_mask_rows,
    write_heuristic_actor_outputs_ids,
    write_heuristic_actor_outputs_mask,
)


def test_legal_action_ids_from_mask_rows_preserves_row_order_and_uint32_dtype() -> None:
    legal_mask = np.array(
        [
            [False, True, False, True],
            [True, False, False, False],
            [False, False, True, False],
        ],
        dtype=np.bool_,
    )

    legal_ids = legal_action_ids_from_mask_rows(
        legal_mask=legal_mask,
        row_indices=np.array([2, 0], dtype=np.int64),
    )

    assert [ids.dtype for ids in legal_ids] == [np.dtype(np.uint32), np.dtype(np.uint32)]
    npt.assert_array_equal(legal_ids[0], np.array([2], dtype=np.uint32))
    npt.assert_array_equal(legal_ids[1], np.array([1, 3], dtype=np.uint32))


def test_write_heuristic_actor_outputs_mask_scatter_actions_logp_and_logits() -> None:
    logits = np.full((3, 5), 7.0, dtype=np.float32)
    actions = np.full((3,), -1, dtype=np.int64)
    logp = np.full((3,), 9.0, dtype=np.float32)
    legal_mask = np.array(
        [
            [False, True, False, True, False],
            [True, False, False, False, True],
            [True, True, False, False, False],
        ],
        dtype=np.bool_,
    )

    write_heuristic_actor_outputs_mask(
        logits_out=logits,
        row_indices=np.array([0, 2], dtype=np.int64),
        chosen_actions=np.array([3, 0], dtype=np.int64),
        legal_mask=legal_mask,
        actions_out=actions,
        logp_out=logp,
        action_dim=5,
    )

    npt.assert_array_equal(actions, np.array([3, -1, 0], dtype=np.int64))
    npt.assert_array_equal(logp, np.array([0.0, 9.0, 0.0], dtype=np.float32))
    npt.assert_array_equal(
        logits,
        np.array(
            [
                [-1.0e9, -100.0, -1.0e9, 0.0, -1.0e9],
                [7.0, 7.0, 7.0, 7.0, 7.0],
                [0.0, -100.0, -1.0e9, -1.0e9, -1.0e9],
            ],
            dtype=np.float32,
        ),
    )


def test_write_heuristic_actor_outputs_ids_scatter_actions_logp_and_packed_logits() -> None:
    logits = np.full((3, 5), 7.0, dtype=np.float32)
    actions = np.full((3,), -1, dtype=np.int64)
    logp = np.full((3,), 9.0, dtype=np.float32)

    write_heuristic_actor_outputs_ids(
        logits_out=logits,
        row_indices=np.array([1, 2], dtype=np.int64),
        chosen_actions=np.array([4, 0], dtype=np.int64),
        legal_ids=np.array([1, 3, 2, 4, 0], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4, 5], dtype=np.uint32),
        actions_out=actions,
        logp_out=logp,
    )

    npt.assert_array_equal(actions, np.array([-1, 4, 0], dtype=np.int64))
    npt.assert_array_equal(logp, np.array([9.0, 0.0, 0.0], dtype=np.float32))
    npt.assert_array_equal(
        logits,
        np.array(
            [
                [7.0, 7.0, 7.0, 7.0, 7.0],
                [-1.0e9, -1.0e9, -100.0, -1.0e9, 0.0],
                [0.0, -1.0e9, -1.0e9, -1.0e9, -1.0e9],
            ],
            dtype=np.float32,
        ),
    )


def test_write_heuristic_actor_outputs_noops_optional_buffers() -> None:
    row_indices = np.array([0], dtype=np.int64)
    chosen_actions = np.array([0], dtype=np.int64)

    write_heuristic_actor_outputs_mask(
        logits_out=None,
        row_indices=row_indices,
        chosen_actions=chosen_actions,
        legal_mask=np.array([[True]], dtype=np.bool_),
        actions_out=None,
        logp_out=None,
        action_dim=1,
    )
    write_heuristic_actor_outputs_ids(
        logits_out=None,
        row_indices=row_indices,
        chosen_actions=chosen_actions,
        legal_ids=np.array([0], dtype=np.uint32),
        legal_offsets=np.array([0, 1], dtype=np.uint32),
        actions_out=None,
        logp_out=None,
    )
