from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.model_action_surface import (
    ModelActionSurfaceSettings,
    model_action_surface_batch_and_ids,
)


def test_model_action_surface_returns_original_batch_when_guards_are_disabled() -> None:
    batch = _batch_with_ids(
        legal_ids=np.array([5, 10], dtype=np.uint32),
        legal_action_meta=np.array([[0, 65535, 65535, 65535], [1, 0, 1, 65535]], dtype=np.uint16),
    )
    settings = ModelActionSurfaceSettings(pass_action_id=5)

    filtered_batch, filtered_ids = model_action_surface_batch_and_ids(
        model=_model_with_action_catalog(),
        batch=batch,
        legal_ids=np.array([5, 10], dtype=np.uint32),
        settings=settings,
    )

    assert filtered_batch is batch
    assert filtered_ids.tolist() == [5, 10]


def test_model_action_surface_forces_pass_for_main_move_only_rows() -> None:
    batch = _batch_with_ids(
        legal_ids=np.array([5, 10, 11], dtype=np.uint32),
        legal_action_meta=np.array(
            [
                [0, 65535, 65535, 65535],
                [1, 0, 1, 65535],
                [1, 1, 2, 65535],
            ],
            dtype=np.uint16,
        ),
    )
    settings = ModelActionSurfaceSettings(pass_action_id=5, force_pass_over_main_move_only=True)

    filtered_batch, filtered_ids = model_action_surface_batch_and_ids(
        model=_model_with_action_catalog(),
        batch=batch,
        legal_ids=np.array([5, 10, 11], dtype=np.uint32),
        settings=settings,
    )

    assert filtered_batch is not batch
    assert filtered_ids.tolist() == [5]
    assert filtered_batch.ids_offsets is not None
    assert filtered_batch.ids_offsets[0].tolist() == [5]


def test_model_action_surface_allows_main_move_only_rows_below_consecutive_limit() -> None:
    batch = _batch_with_ids(
        legal_ids=np.array([5, 10, 11], dtype=np.uint32),
        legal_action_meta=np.array(
            [
                [0, 65535, 65535, 65535],
                [1, 0, 1, 65535],
                [1, 1, 2, 65535],
            ],
            dtype=np.uint16,
        ),
    )
    settings = ModelActionSurfaceSettings(
        pass_action_id=5,
        force_pass_over_main_move_only=True,
        main_move_only_max_consecutive=2,
    )
    action_sequence_state = SimpleNamespace(consecutive_main_moves_by_env=np.array([1], dtype=np.int32))

    filtered_batch, filtered_ids = model_action_surface_batch_and_ids(
        model=_model_with_action_catalog(),
        batch=batch,
        legal_ids=np.array([5, 10, 11], dtype=np.uint32),
        settings=settings,
        action_sequence_state=action_sequence_state,
    )

    assert filtered_batch is batch
    assert filtered_ids.tolist() == [5, 10, 11]


def _batch_with_ids(*, legal_ids: np.ndarray, legal_action_meta: np.ndarray) -> DecisionBoundaryBatch:
    return DecisionBoundaryBatch(
        obs=np.zeros((1, 4), dtype=np.float32),
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([False]),
        to_play=np.array([0], dtype=np.int32),
        actor=np.array([0], dtype=np.int32),
        decision_id=np.array([0], dtype=np.int64),
        engine_status=np.array([0], dtype=np.uint8),
        decision_count=np.array([0], dtype=np.uint32),
        tick_count=np.array([0], dtype=np.uint32),
        episode_seed=np.array([1], dtype=np.uint64),
        episode_key=np.array([1], dtype=np.uint64),
        ids_offsets=(legal_ids, np.array([0, int(legal_ids.shape[0])], dtype=np.int32)),
        legal_action_meta=legal_action_meta,
    )


def _model_with_action_catalog() -> Any:
    return SimpleNamespace(
        action_catalog=SimpleNamespace(
            families=[
                SimpleNamespace(name="pass"),
                SimpleNamespace(name="main_move"),
                SimpleNamespace(name="attack"),
                SimpleNamespace(name="mulligan_confirm"),
                SimpleNamespace(name="mulligan_select"),
            ],
            attack_type_names=[],
        )
    )
