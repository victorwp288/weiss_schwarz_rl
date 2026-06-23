from __future__ import annotations

import math

import numpy as np
import pytest
from weiss_rl.core.masking import masked_logp_from_legal_ids

TOY_PASS_ACTION_ID = 4


def test_masked_logp_from_legal_ids_rejects_illegal_action() -> None:
    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    legal_ids = np.array([0, 2, 4], dtype=np.uint32)
    legal_offsets = np.array([0, 3], dtype=np.int32)
    actions = np.array([3], dtype=np.int64)

    with pytest.raises(ValueError, match="illegal action 3 for row 0"):
        masked_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions)


def test_masked_logp_from_legal_ids_rejects_non_finite_legal_logits() -> None:
    logits = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]], dtype=np.float32)
    legal_ids = np.array([1, 3], dtype=np.uint32)
    legal_offsets = np.array([0, 2], dtype=np.int32)
    actions = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="legal logits must be finite for row 0"):
        masked_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions)


def test_masked_logp_from_legal_ids_rejects_malformed_offsets() -> None:
    logits = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    legal_ids = np.array([0, 2], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 1], dtype=np.int32)
    actions = np.array([0, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="legal_offsets must be nondecreasing"):
        masked_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions)


def test_masked_logp_from_legal_ids_supports_pass_fallback() -> None:
    logits = np.array(
        [
            [0.5, -1.0, 2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [3.0, 0.0, -2.0, 4.0, 1.0],
        ],
        dtype=np.float32,
    )
    legal_ids = np.array([0, 2, 3], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 2, 3], dtype=np.int32)
    actions = np.array([2, TOY_PASS_ACTION_ID, 3], dtype=np.int64)

    logp = masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=TOY_PASS_ACTION_ID,
    )

    expected = np.array(
        [
            2.0 - math.log(math.exp(0.5) + math.exp(2.0)),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(logp, expected, atol=1e-6)


def test_masked_logp_from_legal_ids_requires_pass_action_on_empty_slice() -> None:
    logits = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    legal_ids = np.array([], dtype=np.uint32)
    legal_offsets = np.array([0, 0], dtype=np.int32)
    actions = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="expected pass action"):
        masked_logp_from_legal_ids(
            logits,
            legal_ids,
            legal_offsets,
            actions,
            pass_action_id=TOY_PASS_ACTION_ID,
        )
