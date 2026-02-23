from __future__ import annotations

import numpy as np


def test_pinned_rng_vector_smoke() -> None:
    rng = np.random.default_rng(20260212)
    vec = rng.integers(low=0, high=1000, size=5, dtype=np.int64)
    assert vec.tolist() == [537, 532, 627, 95, 875]
