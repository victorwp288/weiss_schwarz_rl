from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.runtime.components.opponents.central_heuristic_opponents import (
    build_central_packed_heuristic_batch,
    legal_action_ids_from_mask_rows,
    split_central_heuristic_entries,
)

from .central_heuristic_opponents_test_support import central_heuristic_entry


def test_split_central_heuristic_entries_preserves_surface_order() -> None:
    packed_a = central_heuristic_entry(
        batch=SimpleNamespace(
            ids_offsets=(np.asarray([1], dtype=np.uint32), np.asarray([0, 1], dtype=np.uint32)),
            legal_action_meta=None,
            mask=None,
        ),
        row_indices=[0],
    )
    mask_entry = central_heuristic_entry(
        batch=SimpleNamespace(ids_offsets=None, legal_action_meta=None, mask=np.ones((1, 4))), row_indices=[0]
    )
    packed_b = central_heuristic_entry(
        batch=SimpleNamespace(
            ids_offsets=(np.asarray([2], dtype=np.uint32), np.asarray([0, 1], dtype=np.uint32)),
            legal_action_meta=None,
            mask=None,
        ),
        row_indices=[0],
    )

    groups = split_central_heuristic_entries([packed_a, mask_entry, packed_b])

    assert groups.packed == [packed_a, packed_b]
    assert groups.mask == [mask_entry]


def test_build_central_packed_heuristic_batch_concatenates_rows_and_rebases_offsets() -> None:
    batch_a = SimpleNamespace(
        ids_offsets=(
            np.asarray([10, 11, 12, 13], dtype=np.uint32),
            np.asarray([0, 2, 3, 4], dtype=np.uint32),
        ),
        legal_action_meta=np.asarray(
            [
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [3, 0, 0, 0],
                [4, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
        mask=None,
    )
    batch_b = SimpleNamespace(
        ids_offsets=(
            np.asarray([20, 21, 22], dtype=np.uint32),
            np.asarray([0, 1, 3], dtype=np.uint32),
        ),
        legal_action_meta=np.asarray(
            [
                [5, 0, 0, 0],
                [6, 0, 0, 0],
                [7, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
        mask=None,
    )
    obs_a = np.asarray([[1.9, 0.0], [2.9, 0.0], [3.9, 0.0]], dtype=np.float32)
    obs_b = np.asarray([[4.9, 0.0], [5.9, 0.0]], dtype=np.float32)

    packed_batch = build_central_packed_heuristic_batch(
        [
            central_heuristic_entry(batch=batch_a, row_indices=[0, 2], obs_step=obs_a),
            central_heuristic_entry(batch=batch_b, row_indices=[1], obs_step=obs_b),
        ],
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert np.array_equal(packed_batch.obs_rows, np.asarray([[1, 0], [3, 0], [5, 0]], dtype=np.int32))
    assert np.array_equal(packed_batch.legal_ids, np.asarray([10, 11, 13, 21, 22], dtype=np.uint32))
    assert np.array_equal(packed_batch.legal_offsets, np.asarray([0, 2, 3, 5], dtype=np.uint32))
    assert packed_batch.legal_action_meta is not None
    assert np.array_equal(
        packed_batch.legal_action_meta,
        np.asarray(
            [
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [4, 0, 0, 0],
                [6, 0, 0, 0],
                [7, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
    )
    assert packed_batch.entry_counts == [2, 1]


def test_build_central_packed_heuristic_batch_uses_meta_ensurer() -> None:
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([10, 11], dtype=np.uint32),
            np.asarray([0, 2], dtype=np.uint32),
        ),
        legal_action_meta=None,
        mask=None,
    )

    packed_batch = build_central_packed_heuristic_batch(
        [central_heuristic_entry(batch=batch, row_indices=[0])],
        ensure_legal_action_meta=lambda ids, _meta: np.expand_dims(ids.astype(np.uint16), axis=1),
    )

    assert packed_batch.legal_action_meta is not None
    assert np.array_equal(packed_batch.legal_action_meta, np.asarray([[10], [11]], dtype=np.uint16))


def test_legal_action_ids_from_mask_rows_preserves_row_order() -> None:
    legal_mask = np.asarray(
        [
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ],
        dtype=np.bool_,
    )

    legal_action_ids = legal_action_ids_from_mask_rows(legal_mask, np.asarray([2, 0], dtype=np.int64))

    assert [ids.tolist() for ids in legal_action_ids] == [[2, 3], [0, 2]]
