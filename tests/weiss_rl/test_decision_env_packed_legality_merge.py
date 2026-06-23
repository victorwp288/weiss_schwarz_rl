from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from weiss_rl.envs.decision_env import _merge_packed_legality_rows

from tests.weiss_rl.decision_env_test_support import ReadOnlyFakeIdsOut


def test_merge_packed_legality_rows_mutates_read_only_fake_buffers_in_place() -> None:
    current = ReadOnlyFakeIdsOut(2)
    replacement = ReadOnlyFakeIdsOut(2)
    current.legal_ids[:6] = np.array([1, 5, 7, 8, 99, 99], dtype=np.uint32)
    current.legal_offsets[:] = np.array([0, 2, 4], dtype=np.int32)
    replacement.legal_ids[:4] = np.array([9, 10, 11, 12], dtype=np.uint32)
    replacement.legal_offsets[:] = np.array([0, 2, 4], dtype=np.int32)

    ids_before = current.legal_ids
    offsets_before = current.legal_offsets
    _merge_packed_legality_rows(
        dst=current,
        current=current,
        replacement=replacement,
        rows=np.array([False, True]),
    )

    assert current.legal_ids is ids_before
    assert current.legal_offsets is offsets_before
    npt.assert_array_equal(current.legal_offsets, np.array([0, 2, 4], dtype=np.int32))
    npt.assert_array_equal(current.legal_ids[:4], np.array([1, 5, 11, 12], dtype=np.uint32))
    npt.assert_array_equal(current.legal_ids[4:6], np.array([0, 0], dtype=np.uint32))


def test_merge_packed_legality_rows_mutates_real_sim_buffers_in_place() -> None:
    weiss_sim = pytest.importorskip("weiss_sim")
    current = weiss_sim.BatchOutMinimalI16LegalIds(2)
    replacement = weiss_sim.BatchOutMinimalI16LegalIds(2)
    current.legal_ids[:6] = np.array([1, 5, 7, 8, 99, 99], dtype=current.legal_ids.dtype)
    current.legal_offsets[:] = np.array([0, 2, 4], dtype=current.legal_offsets.dtype)
    replacement.legal_ids[:4] = np.array([9, 10, 11, 12], dtype=replacement.legal_ids.dtype)
    replacement.legal_offsets[:] = np.array([0, 2, 4], dtype=replacement.legal_offsets.dtype)

    ids_before = current.legal_ids
    offsets_before = current.legal_offsets
    _merge_packed_legality_rows(
        dst=current,
        current=current,
        replacement=replacement,
        rows=np.array([False, True]),
    )

    assert current.legal_ids is ids_before
    assert current.legal_offsets is offsets_before
    npt.assert_array_equal(current.legal_offsets, np.array([0, 2, 4], dtype=current.legal_offsets.dtype))
    npt.assert_array_equal(current.legal_ids[:4], np.array([1, 5, 11, 12], dtype=current.legal_ids.dtype))
    npt.assert_array_equal(current.legal_ids[4:6], np.array([0, 0], dtype=current.legal_ids.dtype))
