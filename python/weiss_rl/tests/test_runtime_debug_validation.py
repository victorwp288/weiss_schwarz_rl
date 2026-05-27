from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from weiss_rl.runtime_components.debug_validation import (
    validate_env_step_packed_actions,
    validate_sampled_packed_actions,
)


@dataclass(frozen=True)
class _FakeBatch:
    ids_offsets: tuple[np.ndarray, np.ndarray] | None


def test_validate_sampled_packed_actions_accepts_legal_and_empty_pass_rows() -> None:
    validate_sampled_packed_actions(
        source_label="unit",
        row_indices=np.array([0, 2], dtype=np.int64),
        action_subset=np.array([4, 0], dtype=np.int64),
        legal_ids=np.array([3, 4, 9, 5], dtype=np.uint32),
        legal_offsets=np.array([0, 3, 4, 4], dtype=np.uint32),
        pass_action_id=0,
    )


def test_validate_sampled_packed_actions_rejects_illegal_action() -> None:
    with pytest.raises(ValueError, match="debug invalid sampled packed action source=unit env_row=1"):
        validate_sampled_packed_actions(
            source_label="unit",
            row_indices=np.array([1], dtype=np.int64),
            action_subset=np.array([7], dtype=np.int64),
            legal_ids=np.array([3, 4, 9, 5], dtype=np.uint32),
            legal_offsets=np.array([0, 3, 4], dtype=np.uint32),
            pass_action_id=0,
        )


def test_validate_env_step_packed_actions_accepts_matching_env_legality() -> None:
    ids = np.array([1, 2, 8, 4], dtype=np.uint32)
    offsets = np.array([0, 3, 4], dtype=np.uint32)
    validate_env_step_packed_actions(
        source_label="unit",
        actions=np.array([2, 4], dtype=np.int64),
        legal_ids=ids,
        legal_offsets=offsets,
        env_batch=_FakeBatch(ids_offsets=(ids.copy(), offsets.copy())),
    )


def test_validate_env_step_packed_actions_rejects_mismatched_env_legality() -> None:
    with pytest.raises(ValueError, match="debug env-step legality mismatch source=unit env_row=0"):
        validate_env_step_packed_actions(
            source_label="unit",
            actions=np.array([2], dtype=np.int64),
            legal_ids=np.array([1, 2, 8], dtype=np.uint32),
            legal_offsets=np.array([0, 3], dtype=np.uint32),
            env_batch=_FakeBatch(
                ids_offsets=(
                    np.array([1, 8], dtype=np.uint32),
                    np.array([0, 2], dtype=np.uint32),
                )
            ),
        )
