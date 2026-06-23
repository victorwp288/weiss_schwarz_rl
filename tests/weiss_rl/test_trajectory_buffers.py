from __future__ import annotations

import numpy as np
import pytest
from weiss_rl.trajectory.buffers import allocate_unroll_batch, finalize_unroll, write_step_ids_offsets


def _write_ids_step(batch, *, legal_ids: np.ndarray, legal_offsets: np.ndarray) -> None:
    write_step_ids_offsets(
        batch,
        obs=np.zeros((batch.N, batch.obs_len), dtype=batch.obs.dtype),
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        to_play_seat=np.zeros((batch.N,), dtype=np.int8),
        decision_id=np.zeros((batch.N,), dtype=np.int32),
        action=np.zeros((batch.N,), dtype=np.uint32),
        reward=np.zeros((batch.N,), dtype=np.float32),
        terminated=np.zeros((batch.N,), dtype=np.bool_),
        truncated=np.zeros((batch.N,), dtype=np.bool_),
        engine_status=np.zeros((batch.N,), dtype=np.int32),
        episode_seed=np.zeros((batch.N,), dtype=np.uint64),
        episode_key=np.zeros((batch.N,), dtype=np.uint64),
        behavior_logp=np.zeros((batch.N,), dtype=np.float32),
    )


def test_write_step_ids_offsets_rejects_unsorted_ids_within_a_row() -> None:
    batch = allocate_unroll_batch(
        T=1,
        N=2,
        obs_len=4,
        action_space=16,
        obs_dtype="i16",
        legal_repr="ids_offsets",
        max_packed_legal=4,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        _write_ids_step(
            batch,
            legal_ids=np.array([3, 1, 4, 5], dtype=np.int32),
            legal_offsets=np.array([0, 2, 4], dtype=np.int32),
        )


def test_write_step_ids_offsets_rejects_out_of_range_action_ids() -> None:
    batch = allocate_unroll_batch(
        T=1,
        N=2,
        obs_len=4,
        action_space=8,
        obs_dtype="i16",
        legal_repr="ids_offsets",
        max_packed_legal=3,
    )

    with pytest.raises(ValueError, match=r"legal_ids must be < action_space \(8\)"):
        _write_ids_step(
            batch,
            legal_ids=np.array([0, 8, 7], dtype=np.int32),
            legal_offsets=np.array([0, 2, 3], dtype=np.int32),
        )


def test_large_action_spaces_store_packed_legal_ids_without_uint16_wrap() -> None:
    batch = allocate_unroll_batch(
        T=1,
        N=2,
        obs_len=4,
        action_space=70_000,
        obs_dtype="i16",
        legal_repr="ids_offsets",
        max_packed_legal=3,
    )

    _write_ids_step(
        batch,
        legal_ids=np.array([0, 69_999, 42], dtype=np.int64),
        legal_offsets=np.array([0, 2, 3], dtype=np.int32),
    )
    finalized = finalize_unroll(batch)

    assert finalized.legal_ids is not None
    assert finalized.legal_offsets is not None
    assert finalized.legal_ids.dtype == np.uint32
    assert np.array_equal(finalized.legal_ids, np.array([0, 69_999, 42], dtype=np.uint32))
    assert np.array_equal(finalized.legal_offsets, np.array([[0, 2, 3]], dtype=np.uint32))


def test_write_step_ids_offsets_stores_aligned_action_meta() -> None:
    batch = allocate_unroll_batch(
        T=1,
        N=2,
        obs_len=4,
        action_space=16,
        obs_dtype="i16",
        legal_repr="ids_offsets",
        max_packed_legal=3,
        legal_action_meta_width=4,
    )

    write_step_ids_offsets(
        batch,
        obs=np.zeros((batch.N, batch.obs_len), dtype=batch.obs.dtype),
        legal_ids=np.array([1, 4, 7], dtype=np.uint32),
        legal_action_meta=np.array(
            [
                [2, 0, 0, 0],
                [6, 1, 2, 65535],
                [8, 0, 1, 0],
            ],
            dtype=np.uint16,
        ),
        legal_offsets=np.array([0, 2, 3], dtype=np.int32),
        to_play_seat=np.zeros((batch.N,), dtype=np.int8),
        decision_id=np.zeros((batch.N,), dtype=np.int32),
        action=np.zeros((batch.N,), dtype=np.uint32),
        reward=np.zeros((batch.N,), dtype=np.float32),
        terminated=np.zeros((batch.N,), dtype=np.bool_),
        truncated=np.zeros((batch.N,), dtype=np.bool_),
        engine_status=np.zeros((batch.N,), dtype=np.int32),
        episode_seed=np.zeros((batch.N,), dtype=np.uint64),
        episode_key=np.zeros((batch.N,), dtype=np.uint64),
        behavior_logp=np.zeros((batch.N,), dtype=np.float32),
    )

    finalized = finalize_unroll(batch)

    assert finalized.legal_action_meta is not None
    assert np.array_equal(
        finalized.legal_action_meta[:3],
        np.array(
            [
                [2, 0, 0, 0],
                [6, 1, 2, 65535],
                [8, 0, 1, 0],
            ],
            dtype=np.uint16,
        ),
    )
