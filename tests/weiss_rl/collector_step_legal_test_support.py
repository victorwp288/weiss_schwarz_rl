from __future__ import annotations

import numpy as np


def teacher_label_arrays(
    num_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.arange(num_rows, dtype=np.int32),
        np.full((num_rows,), 1, dtype=np.int32),
        np.full((num_rows,), 2, dtype=np.int32),
        np.full((num_rows,), 3, dtype=np.int32),
        np.full((num_rows,), 4, dtype=np.int32),
        np.ones((num_rows,), dtype=np.bool_),
    )
