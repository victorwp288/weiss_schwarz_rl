from __future__ import annotations

import numpy as np
from weiss_rl.runtime.components.collector_unroll_storage import (
    legal_actions_from_collector_steps,
    normalize_collector_bootstrap_arrays,
)


def test_normalize_collector_bootstrap_arrays_preserves_runtime_unroll_dtypes() -> None:
    bootstrap = normalize_collector_bootstrap_arrays(
        bootstrap_obs=np.ones((2, 3), dtype=np.float64),
        bootstrap_actor=np.array([0, 1], dtype=np.int32),
        bootstrap_value=np.array([0.1, 0.2], dtype=np.float64),
    )

    assert bootstrap.obs.dtype == np.float32
    assert bootstrap.actor.dtype == np.int64
    assert bootstrap.value.dtype == np.float32
    assert bootstrap.obs.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert bootstrap.actor.tolist() == [0, 1]


def test_legal_actions_from_collector_steps_preserves_packed_and_mask_layouts() -> None:
    packed = legal_actions_from_collector_steps(
        layout_name="i16_legal_ids",
        action_dim=8,
        packed_ids=[np.array([1, 3], dtype=np.uint32), np.array([2], dtype=np.uint32)],
        packed_offsets=[
            np.array([0], dtype=np.uint32),
            np.array([1, 2], dtype=np.uint32),
            np.array([3], dtype=np.uint32),
        ],
        packed_meta=[
            np.array([[10, 0], [30, 0]], dtype=np.uint16),
            np.array([[20, 0]], dtype=np.uint16),
        ],
        mask_steps=[],
    )

    assert packed.ids is not None
    assert packed.offsets is not None
    assert packed.meta is not None
    assert packed.ids.tolist() == [1, 3, 2]
    assert packed.offsets.tolist() == [0, 1, 2, 3]
    assert packed.meta[:, 0].tolist() == [10, 30, 20]

    mask = legal_actions_from_collector_steps(
        layout_name="dense_mask",
        action_dim=3,
        packed_ids=[],
        packed_offsets=[np.array([0], dtype=np.uint32)],
        packed_meta=[],
        mask_steps=[
            np.array([[True, False, False]], dtype=np.bool_),
            np.array([[False, True, True]], dtype=np.bool_),
        ],
    )

    assert mask.mask is not None
    assert mask.mask.shape == (2, 1, 3)
    assert mask.mask.tolist() == [[[True, False, False]], [[False, True, True]]]
