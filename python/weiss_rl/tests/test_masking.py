from __future__ import annotations

import numpy as np
import pytest

from weiss_rl.masking import assert_strictly_increasing_legal_ids


def test_legal_ids_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError):
        assert_strictly_increasing_legal_ids(np.array([1, 1, 3], dtype=np.uint32))


def test_legal_ids_accepts_sorted_unique() -> None:
    assert_strictly_increasing_legal_ids(np.array([2, 5, 9], dtype=np.uint32))
