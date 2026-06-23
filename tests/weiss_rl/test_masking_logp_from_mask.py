from __future__ import annotations

import math

import numpy as np
import pytest
from weiss_rl.core.masking import masked_logp_from_mask

TOY_PASS_ACTION_ID = 4


def test_masked_logp_from_mask_supports_pass_fallback_and_single_legal_action() -> None:
    logits = np.array(
        [
            [1.0, 0.0, -1.0, 2.0, 3.0],
            [0.0, -2.0, 0.5, 1.5, -3.0],
            [-1.0, 4.0, 0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    actions = np.array([0, TOY_PASS_ACTION_ID, 1], dtype=np.int64)

    logp = masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=TOY_PASS_ACTION_ID)

    expected = np.array(
        [
            1.0 - math.log(math.exp(1.0) + math.exp(-1.0)),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(logp, expected, atol=1e-6)


def test_masked_logp_from_mask_rejects_illegal_action() -> None:
    logits = np.array([[2.0, 1.0, 0.0, -3.0]], dtype=np.float32)
    legal_mask = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    actions = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="illegal action 1 for row 0"):
        masked_logp_from_mask(logits, legal_mask, actions)


def test_masked_logp_from_mask_requires_pass_action_id_for_empty_rows() -> None:
    logits = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    legal_mask = np.zeros((1, 5), dtype=np.uint8)
    actions = np.array([TOY_PASS_ACTION_ID], dtype=np.int64)

    with pytest.raises(ValueError, match="pass_action_id is required"):
        masked_logp_from_mask(logits, legal_mask, actions)
