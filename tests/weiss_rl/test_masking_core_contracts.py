from __future__ import annotations

import math

import numpy as np
import pytest
from weiss_rl.core.masking import (
    assert_strictly_increasing_legal_ids,
    masked_log_softmax,
)


def test_legal_ids_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError):
        assert_strictly_increasing_legal_ids(np.array([1, 1, 3], dtype=np.uint32))


def test_legal_ids_accepts_sorted_unique() -> None:
    assert_strictly_increasing_legal_ids(np.array([2, 5, 9], dtype=np.uint32))


def test_masked_log_softmax_keeps_empty_rows_at_negative_infinity() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]], dtype=np.float32)
    legal_mask = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8)

    log_probs = masked_log_softmax(logits, legal_mask)

    assert np.isfinite(log_probs[0, 0])
    assert np.isfinite(log_probs[0, 2])
    assert np.isneginf(log_probs[0, 1])
    assert np.all(np.isneginf(log_probs[1]))


def test_masked_log_softmax_avoids_nan_for_fully_masked_rows() -> None:
    logits = np.array(
        [
            [1.0, 2.0, 0.0, -1.0],
            [3.0, -5.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.array(
        [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    log_probs = masked_log_softmax(logits, legal_mask)

    expected_row0 = np.array(
        [1.0 - math.log(math.exp(1.0) + math.exp(0.0)), -np.inf, -math.log(math.exp(1.0) + math.exp(0.0)), -np.inf],
        dtype=np.float32,
    )
    assert np.allclose(log_probs[0], expected_row0, atol=1e-6, equal_nan=False)
    assert np.all(np.isneginf(log_probs[1]))
    assert not np.any(np.isnan(log_probs[1]))
    assert np.allclose(log_probs[2], np.array([-np.inf, 0.0, -np.inf, -np.inf], dtype=np.float32), atol=1e-6)
