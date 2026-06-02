from __future__ import annotations

import numpy as np
import numpy.testing as npt

from weiss_rl.runtime.components.deterministic_logits import (
    write_deterministic_logits,
    write_deterministic_logits_from_packed,
)


def test_write_deterministic_logits_preserves_unselected_rows_and_marks_chosen_action() -> None:
    logits = np.full((3, 5), 7.0, dtype=np.float32)

    write_deterministic_logits(
        logits_out=logits,
        row_indices=np.array([0, 2], dtype=np.int64),
        chosen_actions=np.array([3, 1], dtype=np.int64),
        legal_action_ids=[
            np.array([1, 3], dtype=np.uint32),
            np.array([0, 1, 4], dtype=np.uint32),
        ],
        action_dim=5,
    )

    npt.assert_array_equal(
        logits,
        np.array(
            [
                [-1.0e9, -100.0, -1.0e9, 0.0, -1.0e9],
                [7.0, 7.0, 7.0, 7.0, 7.0],
                [-100.0, 0.0, -1.0e9, -1.0e9, -100.0],
            ],
            dtype=np.float32,
        ),
    )


def test_write_deterministic_logits_noops_without_output_buffer() -> None:
    write_deterministic_logits(
        logits_out=None,
        row_indices=np.array([0], dtype=np.int64),
        chosen_actions=np.array([0], dtype=np.int64),
        legal_action_ids=[np.array([0], dtype=np.uint32)],
        action_dim=1,
    )


def test_write_deterministic_logits_from_packed_uses_row_offsets() -> None:
    logits = np.full((3, 5), 7.0, dtype=np.float32)

    write_deterministic_logits_from_packed(
        logits_out=logits,
        row_indices=np.array([1, 2], dtype=np.int64),
        chosen_actions=np.array([4, 0], dtype=np.int64),
        legal_ids=np.array([1, 3, 2, 4, 0], dtype=np.uint32),
        legal_offsets=np.array([0, 2, 4, 5], dtype=np.uint32),
    )

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


def test_write_deterministic_logits_from_packed_noops_without_output_buffer() -> None:
    write_deterministic_logits_from_packed(
        logits_out=None,
        row_indices=np.array([0], dtype=np.int64),
        chosen_actions=np.array([0], dtype=np.int64),
        legal_ids=np.array([0], dtype=np.uint32),
        legal_offsets=np.array([0, 1], dtype=np.uint32),
    )
