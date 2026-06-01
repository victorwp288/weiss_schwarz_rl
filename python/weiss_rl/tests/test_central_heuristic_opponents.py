from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from weiss_rl.runtime_components.central_heuristic_opponent_apply import apply_central_heuristic_opponent_outputs
from weiss_rl.runtime_components.central_heuristic_opponents import (
    build_central_packed_heuristic_batch,
    legal_action_ids_from_mask_rows,
    split_central_heuristic_entries,
)
from weiss_rl.runtime_components.central_opponent_groups import CentralOpponentEntry
from weiss_rl.runtime_components.deterministic_logits import (
    write_deterministic_logits,
    write_deterministic_logits_from_packed,
)


def _entry(
    *,
    batch: SimpleNamespace,
    row_indices: list[int],
    obs_step: np.ndarray | None = None,
) -> CentralOpponentEntry:
    rows = np.asarray(row_indices, dtype=np.int64)
    obs = obs_step if obs_step is not None else np.arange(12, dtype=np.float32).reshape(4, 3)
    return CentralOpponentEntry(
        actor=SimpleNamespace(name="actor"),
        batch=batch,
        row_indices=rows,
        obs_step=obs,
        actor_step=np.zeros((obs.shape[0],), dtype=np.int64),
        logits_out=None,
        values_out=np.zeros((obs.shape[0],), dtype=np.float32),
    )


def test_split_central_heuristic_entries_preserves_surface_order() -> None:
    packed_a = _entry(
        batch=SimpleNamespace(
            ids_offsets=(np.asarray([1], dtype=np.uint32), np.asarray([0, 1], dtype=np.uint32)),
            legal_action_meta=None,
            mask=None,
        ),
        row_indices=[0],
    )
    mask_entry = _entry(
        batch=SimpleNamespace(ids_offsets=None, legal_action_meta=None, mask=np.ones((1, 4))), row_indices=[0]
    )
    packed_b = _entry(
        batch=SimpleNamespace(
            ids_offsets=(np.asarray([2], dtype=np.uint32), np.asarray([0, 1], dtype=np.uint32)),
            legal_action_meta=None,
            mask=None,
        ),
        row_indices=[0],
    )

    groups = split_central_heuristic_entries([packed_a, mask_entry, packed_b])

    assert groups.packed == [packed_a, packed_b]
    assert groups.mask == [mask_entry]


def test_build_central_packed_heuristic_batch_concatenates_rows_and_rebases_offsets() -> None:
    batch_a = SimpleNamespace(
        ids_offsets=(
            np.asarray([10, 11, 12, 13], dtype=np.uint32),
            np.asarray([0, 2, 3, 4], dtype=np.uint32),
        ),
        legal_action_meta=np.asarray(
            [
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [3, 0, 0, 0],
                [4, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
        mask=None,
    )
    batch_b = SimpleNamespace(
        ids_offsets=(
            np.asarray([20, 21, 22], dtype=np.uint32),
            np.asarray([0, 1, 3], dtype=np.uint32),
        ),
        legal_action_meta=np.asarray(
            [
                [5, 0, 0, 0],
                [6, 0, 0, 0],
                [7, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
        mask=None,
    )
    obs_a = np.asarray([[1.9, 0.0], [2.9, 0.0], [3.9, 0.0]], dtype=np.float32)
    obs_b = np.asarray([[4.9, 0.0], [5.9, 0.0]], dtype=np.float32)

    packed_batch = build_central_packed_heuristic_batch(
        [
            _entry(batch=batch_a, row_indices=[0, 2], obs_step=obs_a),
            _entry(batch=batch_b, row_indices=[1], obs_step=obs_b),
        ],
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert np.array_equal(packed_batch.obs_rows, np.asarray([[1, 0], [3, 0], [5, 0]], dtype=np.int32))
    assert np.array_equal(packed_batch.legal_ids, np.asarray([10, 11, 13, 21, 22], dtype=np.uint32))
    assert np.array_equal(packed_batch.legal_offsets, np.asarray([0, 2, 3, 5], dtype=np.uint32))
    assert packed_batch.legal_action_meta is not None
    assert np.array_equal(
        packed_batch.legal_action_meta,
        np.asarray(
            [
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [4, 0, 0, 0],
                [6, 0, 0, 0],
                [7, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
    )
    assert packed_batch.entry_counts == [2, 1]


def test_build_central_packed_heuristic_batch_uses_meta_ensurer() -> None:
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([10, 11], dtype=np.uint32),
            np.asarray([0, 2], dtype=np.uint32),
        ),
        legal_action_meta=None,
        mask=None,
    )

    packed_batch = build_central_packed_heuristic_batch(
        [_entry(batch=batch, row_indices=[0])],
        ensure_legal_action_meta=lambda ids, _meta: np.expand_dims(ids.astype(np.uint16), axis=1),
    )

    assert packed_batch.legal_action_meta is not None
    assert np.array_equal(packed_batch.legal_action_meta, np.asarray([[10], [11]], dtype=np.uint16))


def test_legal_action_ids_from_mask_rows_preserves_row_order() -> None:
    legal_mask = np.asarray(
        [
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ],
        dtype=np.bool_,
    )

    legal_action_ids = legal_action_ids_from_mask_rows(legal_mask, np.asarray([2, 0], dtype=np.int64))

    assert [ids.tolist() for ids in legal_action_ids] == [[2, 3], [0, 2]]


def test_apply_central_heuristic_opponent_outputs_batches_packed_entries_and_applies_mask_entries() -> None:
    class _Policy:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def choose_actions_from_meta_batch(
            self,
            obs_rows: np.ndarray,
            legal_ids: np.ndarray,
            legal_offsets: np.ndarray,
            legal_action_meta: np.ndarray | None,
        ) -> np.ndarray:
            self.calls.append(
                {
                    "obs_rows": obs_rows.copy(),
                    "legal_ids": legal_ids.copy(),
                    "legal_offsets": legal_offsets.copy(),
                    "legal_action_meta": None if legal_action_meta is None else legal_action_meta.copy(),
                }
            )
            return np.asarray([1, 2], dtype=np.int64)

    actor = SimpleNamespace(name="actor")
    packed_values = np.full((3,), 5.0, dtype=np.float32)
    packed_logits = np.full((3, 4), -7.0, dtype=np.float32)
    mask_values = np.full((3,), 6.0, dtype=np.float32)
    mask_logits = np.full((3, 4), -8.0, dtype=np.float32)
    packed_entry = CentralOpponentEntry(
        actor=actor,
        batch=SimpleNamespace(
            ids_offsets=(
                np.asarray([0, 1, 2, 3, 0, 2], dtype=np.uint32),
                np.asarray([0, 2, 4, 6], dtype=np.uint32),
            ),
            legal_action_meta=np.asarray([[0], [1], [2], [3], [0], [2]], dtype=np.uint16),
            mask=None,
        ),
        row_indices=np.asarray([0, 2], dtype=np.int64),
        obs_step=np.asarray([[10, 0], [20, 0], [30, 0]], dtype=np.float32),
        actor_step=np.asarray([1, 0, 1], dtype=np.int64),
        logits_out=packed_logits,
        values_out=packed_values,
    )
    mask_entry = CentralOpponentEntry(
        actor=actor,
        batch=SimpleNamespace(
            ids_offsets=None,
            mask=np.asarray(
                [
                    [True, False, True, False],
                    [False, True, False, True],
                    [False, False, True, True],
                ],
                dtype=np.bool_,
            ),
        ),
        row_indices=np.asarray([1], dtype=np.int64),
        obs_step=np.asarray([[1, 0], [2, 0], [3, 0]], dtype=np.float32),
        actor_step=np.asarray([0, 1, 0], dtype=np.int64),
        logits_out=mask_logits,
        values_out=mask_values,
    )
    advanced: list[list[int]] = []
    policy = _Policy()

    apply_central_heuristic_opponent_outputs(
        policy_id="B2 HeuristicPublic",
        entries=[packed_entry, mask_entry],
        heuristic_policy=policy,
        fixed_opponent_backend="python_batched",
        track_heuristic_hidden_state=True,
        central_advance_actor_rows=lambda **kwargs: advanced.append(kwargs["row_indices_by_actor"][0].tolist()),
        heuristic_public_actions_from_ids=lambda **_: np.asarray([], dtype=np.int64),
        heuristic_public_actions_from_mask=lambda **_: np.asarray([3], dtype=np.int64),
        ensure_legal_action_meta=lambda _ids, meta: meta,
        maybe_debug_validate_sampled_packed_actions=lambda **_: None,
        write_deterministic_logits_from_packed=write_deterministic_logits_from_packed,
        write_deterministic_logits=lambda **kwargs: write_deterministic_logits(action_dim=4, **kwargs),
    )

    assert advanced == [[0, 2]]
    assert len(policy.calls) == 1
    assert np.array_equal(policy.calls[0]["obs_rows"], np.asarray([[10, 0], [30, 0]], dtype=np.int32))
    assert packed_values.tolist() == [0.0, 5.0, 0.0]
    assert mask_values.tolist() == [6.0, 0.0, 6.0]
    assert packed_logits[0, 1] == 0.0
    assert packed_logits[2, 2] == 0.0
    assert mask_logits[1, 3] == 0.0
    assert mask_logits[1, 1] == -100.0


def test_apply_central_heuristic_opponent_outputs_simulator_native_uses_ids_callback_and_debug_validation() -> None:
    actor = SimpleNamespace(name="actor")
    logits = np.full((2, 4), -5.0, dtype=np.float32)
    values = np.full((2,), 9.0, dtype=np.float32)
    entry = CentralOpponentEntry(
        actor=actor,
        batch=SimpleNamespace(
            ids_offsets=(np.asarray([0, 2, 1, 3], dtype=np.uint32), np.asarray([0, 2, 4], dtype=np.uint32)),
            legal_action_meta=None,
            mask=None,
        ),
        row_indices=np.asarray([1], dtype=np.int64),
        obs_step=np.asarray([[1, 0], [2, 0]], dtype=np.float32),
        actor_step=np.asarray([0, 1], dtype=np.int64),
        logits_out=logits,
        values_out=values,
    )
    ids_calls: list[dict[str, Any]] = []
    debug_calls: list[dict[str, Any]] = []

    apply_central_heuristic_opponent_outputs(
        policy_id="B2 HeuristicPublic",
        entries=[entry],
        heuristic_policy=object(),
        fixed_opponent_backend="simulator_native",
        track_heuristic_hidden_state=False,
        central_advance_actor_rows=lambda **_: None,
        heuristic_public_actions_from_ids=lambda **kwargs: ids_calls.append(kwargs) or np.asarray([3], dtype=np.int64),
        heuristic_public_actions_from_mask=lambda **_: np.asarray([], dtype=np.int64),
        ensure_legal_action_meta=lambda ids, _meta: np.expand_dims(ids.astype(np.uint16), axis=1),
        maybe_debug_validate_sampled_packed_actions=lambda **kwargs: debug_calls.append(kwargs),
        write_deterministic_logits_from_packed=write_deterministic_logits_from_packed,
        write_deterministic_logits=lambda **kwargs: write_deterministic_logits(action_dim=4, **kwargs),
    )

    assert len(ids_calls) == 1
    assert ids_calls[0]["row_indices"].tolist() == [1]
    assert ids_calls[0]["legal_action_meta"].tolist() == [[0], [2], [1], [3]]
    assert debug_calls[0]["source_label"] == "central:opponent:B2 HeuristicPublic:heuristic"
    assert debug_calls[0]["action_subset"].tolist() == [3]
    assert values.tolist() == [9.0, 0.0]
    assert logits[1, 3] == 0.0
    assert logits[1, 1] == -100.0
