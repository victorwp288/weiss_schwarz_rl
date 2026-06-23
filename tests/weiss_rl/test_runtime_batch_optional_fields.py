from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
from weiss_rl.runtime.components.batching import concat_optional_time_major_field


def test_concat_optional_time_major_field_fills_unlabeled_rows_with_sentinels() -> None:
    labeled = SimpleNamespace(
        obs=np.zeros((2, 2, 1), dtype=np.float32),
        teacher_family=np.asarray([[7, 8], [9, 10]], dtype=np.int32),
        teacher_valid=np.asarray([[True, False], [False, True]], dtype=np.bool_),
    )
    unlabeled = SimpleNamespace(
        obs=np.zeros((2, 3, 1), dtype=np.float32),
        teacher_family=None,
        teacher_valid=None,
    )

    teacher_family = concat_optional_time_major_field(
        [cast(Any, labeled), cast(Any, unlabeled)],
        "teacher_family",
        missing_fill_value=-1,
    )
    teacher_valid = concat_optional_time_major_field(
        [cast(Any, labeled), cast(Any, unlabeled)],
        "teacher_valid",
        missing_fill_value=False,
    )

    assert teacher_family is not None
    assert teacher_valid is not None
    assert teacher_family.shape == (2, 5)
    assert teacher_valid.shape == (2, 5)
    npt.assert_array_equal(teacher_family[:, :2], labeled.teacher_family)
    npt.assert_array_equal(teacher_family[:, 2:], np.full((2, 3), -1, dtype=np.int32))
    npt.assert_array_equal(teacher_valid[:, :2], labeled.teacher_valid)
    npt.assert_array_equal(teacher_valid[:, 2:], np.zeros((2, 3), dtype=np.bool_))
