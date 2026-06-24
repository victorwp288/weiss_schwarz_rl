from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.batching.legal_batching import (
    concatenate_batch_legal_actions,
    concatenate_legal_actions,
    require_ids_offsets,
    slice_packed_rows_with_meta,
    structured_legal_batch_from_packed,
)

from .runtime_test_support import _make_runtime_unroll


def test_runtime_batching_concatenates_packed_legal_actions_in_time_major_order() -> None:
    unroll_a = SimpleNamespace(
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([1, 3, 2, 4], dtype=np.uint32),
            np.asarray([0, 2, 4], dtype=np.uint32),
            action_space=8,
        ),
    )
    unroll_b = SimpleNamespace(
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([5, 7, 6], dtype=np.uint32),
            np.asarray([0, 2, 3], dtype=np.uint32),
            action_space=8,
        ),
    )

    combined = concatenate_legal_actions([unroll_a, unroll_b], action_space=8)

    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.ids.tolist() == [1, 3, 5, 7, 2, 4, 6]
    assert combined.offsets.tolist() == [0, 2, 4, 6, 7]


def test_runtime_batching_concatenate_legal_actions_keeps_packed_ids_fast_path() -> None:
    packed = LegalActionBatch.from_packed(
        np.array([0, 2, 1, 2], dtype=np.uint32),
        np.array([0, 2, 4], dtype=np.uint32),
        action_space=64,
    )
    unroll_a = replace(_make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0), legal_actions=packed)
    unroll_b = replace(_make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0), legal_actions=packed)

    combined = concatenate_legal_actions([unroll_a, unroll_b], action_space=64)

    assert combined.mask is None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.action_space == 64
    assert combined.offsets.tolist() == [0, 2, 4]
    assert combined.ids.tolist() == [0, 2, 0, 2]


def test_runtime_batching_concatenate_legal_actions_reorders_packed_rows_to_match_time_major_layout() -> None:
    packed_a = LegalActionBatch.from_packed(
        np.array([10, 11, 20, 21], dtype=np.uint32),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
    )
    packed_b = LegalActionBatch.from_packed(
        np.array([30, 31, 40, 41], dtype=np.uint32),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
    )
    unroll_a = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        legal_actions=packed_a,
    )
    unroll_b = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        legal_actions=packed_b,
    )

    combined = concatenate_legal_actions([unroll_a, unroll_b], action_space=64)

    assert combined.mask is None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.action_space == 64
    assert combined.offsets.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert combined.ids.tolist() == [10, 11, 30, 31, 20, 21, 40, 41]


def test_runtime_batching_mixed_legal_payloads_fall_back_to_dense_mask() -> None:
    packed_unroll = replace(
        _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([1, 3, 2], dtype=np.uint32),
            np.asarray([0, 2, 3], dtype=np.uint32),
            action_space=5,
        ),
    )
    mask_unroll = replace(
        _make_runtime_unroll(actor_id=1, unroll_seq=0, behavior_policy_version=0),
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_mask(
            np.asarray([[[True, False, False, False, True]], [[False, True, False, True, False]]], dtype=np.bool_),
            action_space=5,
        ),
    )

    combined = concatenate_legal_actions([packed_unroll, mask_unroll], action_space=5)

    assert combined.ids is None
    assert combined.offsets is None
    assert combined.mask is not None
    assert combined.mask.tolist() == [
        [[False, True, False, True, False], [True, False, False, False, True]],
        [[False, False, True, False, False], [False, True, False, True, False]],
    ]


def test_runtime_batching_slices_packed_rows_with_meta() -> None:
    ids, offsets, meta = slice_packed_rows_with_meta(
        np.asarray([1, 2, 3, 4, 5], dtype=np.uint32),
        np.asarray([0, 2, 3, 5], dtype=np.uint32),
        np.asarray([2, 0], dtype=np.int64),
        legal_action_meta=np.asarray([[10], [11], [12], [13], [14]], dtype=np.uint16),
    )

    assert ids.tolist() == [4, 5, 1, 2]
    assert offsets.tolist() == [0, 2, 4]
    assert meta is not None
    assert meta.tolist() == [[13], [14], [10], [11]]


def test_runtime_batching_structured_legal_batch_from_packed_preserves_meta() -> None:
    batch = structured_legal_batch_from_packed(
        np.asarray([1, 2, 3], dtype=np.uint32),
        np.asarray([0, 1, 3], dtype=np.uint32),
        np.asarray([1], dtype=np.int64),
        np.asarray([[10], [11], [12]], dtype=np.uint16),
    )

    assert batch.ids is not None
    assert batch.offsets is not None
    assert batch.meta is not None
    assert batch.ids.tolist() == [2, 3]
    assert batch.offsets.tolist() == [0, 2]
    assert batch.meta.tolist() == [[11], [12]]


def test_legal_action_batch_uses_metadata_to_expand_packed_payloads() -> None:
    packed = LegalActionBatch.from_packed(
        np.array([1, 3], dtype=np.uint32),
        np.array([0, 1, 2], dtype=np.uint32),
        action_space=5,
    )

    mask = packed.to_mask(expected_shape=(1, 2))

    assert packed.action_space == 5
    assert np.array_equal(
        mask,
        np.array([[[False, True, False, False, False], [False, False, False, True, False]]], dtype=np.bool_),
    )


def test_runtime_batching_concatenates_decision_batches_or_rejects_missing_packed_legality() -> None:
    first = SimpleNamespace(
        mask=None,
        ids_offsets=(np.asarray([1, 2], dtype=np.uint32), np.asarray([0, 2], dtype=np.uint32)),
        legal_action_meta=np.asarray([[10], [11]], dtype=np.uint16),
    )
    second = SimpleNamespace(
        mask=None,
        ids_offsets=(np.asarray([3], dtype=np.uint32), np.asarray([0, 1], dtype=np.uint32)),
        legal_action_meta=np.asarray([[12]], dtype=np.uint16),
    )

    combined = concatenate_batch_legal_actions([first, second], action_space=8)

    assert combined is not None
    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.meta is not None
    assert combined.ids.tolist() == [1, 2, 3]
    assert combined.offsets.tolist() == [0, 2, 3]
    assert combined.meta.tolist() == [[10], [11], [12]]

    with pytest.raises(RuntimeError, match="requires ids_offsets"):
        require_ids_offsets(SimpleNamespace(ids_offsets=None))
