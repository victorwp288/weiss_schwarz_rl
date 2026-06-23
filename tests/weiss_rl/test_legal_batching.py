from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.runtime.components.legal_batching import concatenate_legal_actions


def test_concatenate_legal_actions_fills_missing_ordered_packed_meta_with_sentinel() -> None:
    with_meta = SimpleNamespace(
        obs=np.zeros((1, 2, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([1, 2], dtype=np.uint32),
            np.asarray([0, 1, 2], dtype=np.uint32),
            meta=np.asarray([[10, 11], [20, 21]], dtype=np.uint16),
            action_space=8,
        ),
    )
    without_meta = SimpleNamespace(
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([3], dtype=np.uint32),
            np.asarray([0, 1], dtype=np.uint32),
            action_space=8,
        ),
    )

    combined = concatenate_legal_actions([with_meta, without_meta], action_space=8)

    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.meta is not None
    assert combined.ids.tolist() == [1, 2, 3]
    assert combined.offsets.tolist() == [0, 1, 2, 3]
    assert combined.meta.tolist() == [[10, 11], [20, 21], [65535, 65535]]


def test_concatenate_legal_actions_trims_legacy_packed_payload_to_unroll_rows() -> None:
    first = SimpleNamespace(
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([1, 2, 3, 4], dtype=np.uint32),
            np.asarray([0, 2, 4], dtype=np.uint32),
            meta=np.asarray([[10], [20], [30], [40]], dtype=np.uint16),
            action_space=8,
        ),
    )
    second = SimpleNamespace(
        obs=np.zeros((1, 1, 1), dtype=np.float32),
        legal_actions=LegalActionBatch.from_packed(
            np.asarray([5], dtype=np.uint32),
            np.asarray([0, 1], dtype=np.uint32),
            meta=np.asarray([[50]], dtype=np.uint16),
            action_space=8,
        ),
    )

    combined = concatenate_legal_actions([first, second], action_space=8)

    assert combined.ids is not None
    assert combined.offsets is not None
    assert combined.meta is not None
    assert combined.ids.tolist() == [1, 2, 5]
    assert combined.offsets.tolist() == [0, 2, 3]
    assert combined.meta.tolist() == [[10], [20], [50]]
