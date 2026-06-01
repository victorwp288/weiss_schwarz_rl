from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weiss_rl.runtime_components.central_opponent_groups import CentralOpponentEntry
from weiss_rl.runtime_components.central_snapshot_opponents import (
    CentralSnapshotModelOutputs,
    apply_central_snapshot_opponent_policy,
    apply_central_snapshot_outputs,
    build_central_snapshot_forward_batch,
    run_central_snapshot_model,
)


def _entry(
    *,
    actor: SimpleNamespace,
    batch: SimpleNamespace,
    row_indices: list[int],
    obs_step: np.ndarray,
    actor_step: np.ndarray,
    logits_out: np.ndarray | None,
    values_out: np.ndarray,
) -> CentralOpponentEntry:
    return CentralOpponentEntry(
        actor=actor,
        batch=batch,
        row_indices=np.asarray(row_indices, dtype=np.int64),
        obs_step=obs_step,
        actor_step=actor_step,
        logits_out=logits_out,
        values_out=values_out,
    )


def test_build_central_snapshot_forward_batch_preserves_entry_and_row_order() -> None:
    actor_a = SimpleNamespace(opponent_hidden=torch.arange(12, dtype=torch.float32).reshape(4, 3))
    actor_b = SimpleNamespace(opponent_hidden=torch.arange(12, 24, dtype=torch.float32).reshape(4, 3))
    entries = [
        _entry(
            actor=actor_a,
            batch=SimpleNamespace(),
            row_indices=[2, 0],
            obs_step=np.asarray([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32),
            actor_step=np.asarray([0, 1, 1, 0], dtype=np.int64),
            logits_out=None,
            values_out=np.zeros((4,), dtype=np.float32),
        ),
        _entry(
            actor=actor_b,
            batch=SimpleNamespace(),
            row_indices=[1],
            obs_step=np.asarray([[5, 0], [6, 0], [7, 0], [8, 0]], dtype=np.float32),
            actor_step=np.asarray([1, 0, 0, 1], dtype=np.int64),
            logits_out=None,
            values_out=np.zeros((4,), dtype=np.float32),
        ),
    ]

    forward_batch = build_central_snapshot_forward_batch(entries)

    assert np.array_equal(forward_batch.obs, np.asarray([[3, 0], [1, 0], [6, 0]], dtype=np.float32))
    assert np.array_equal(forward_batch.actor, np.asarray([1, 0, 0], dtype=np.int64))
    assert forward_batch.hidden.tolist() == [
        [6.0, 7.0, 8.0],
        [0.0, 1.0, 2.0],
        [15.0, 16.0, 17.0],
    ]


def test_run_central_snapshot_model_batches_entries_under_lock() -> None:
    actor = SimpleNamespace(opponent_hidden=torch.zeros((2, 3), dtype=torch.float32))
    entry = _entry(
        actor=actor,
        batch=SimpleNamespace(),
        row_indices=[1, 0],
        obs_step=np.asarray([[1, 0], [2, 0]], dtype=np.float32),
        actor_step=np.asarray([0, 1], dtype=np.int64),
        logits_out=None,
        values_out=np.zeros((2,), dtype=np.float32),
    )

    class _Model:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, torch.Tensor]] = []

        def forward_seat_aware(self, obs_tensor, actor_tensor, hidden_tensor):
            self.calls.append(
                (
                    obs_tensor.detach().cpu().numpy().copy(),
                    actor_tensor.detach().cpu().numpy().copy(),
                    hidden_tensor.detach().cpu().clone(),
                )
            )
            logits = torch.arange(10, dtype=torch.float32).reshape(2, 5)
            values = torch.tensor([4.0, 5.0], dtype=torch.float32)
            return logits, values, hidden_tensor + 7.0

    model = _Model()

    outputs = run_central_snapshot_model(
        model=model,
        entries=[entry],
        lock=threading.Lock(),
        device=torch.device("cpu"),
        amp_enabled=False,
    )

    assert len(model.calls) == 1
    assert np.array_equal(model.calls[0][0], np.asarray([[2, 0], [1, 0]], dtype=np.float32))
    assert np.array_equal(model.calls[0][1], np.asarray([1, 0], dtype=np.int64))
    assert np.array_equal(outputs.logits, np.arange(10, dtype=np.float32).reshape(2, 5))
    assert outputs.values.tolist() == [4.0, 5.0]
    assert outputs.next_hidden.tolist() == [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]]


def test_apply_central_snapshot_opponent_policy_requires_registered_model() -> None:
    with pytest.raises(RuntimeError, match="missing opponent snapshot model for policy_id 'snapshot_a'"):
        apply_central_snapshot_opponent_policy(
            policy_id="snapshot_a",
            entries=[],
            opponent_models={},
            opponent_model_locks={},
            device=torch.device("cpu"),
            amp_enabled=False,
            action_selection="sample",
            pass_action_id=0,
            action_dim=4,
            ensure_legal_action_meta=lambda _ids, meta: meta,
        )


def test_apply_central_snapshot_opponent_policy_runs_model_and_applies_configured_outputs() -> None:
    actor = SimpleNamespace(opponent_hidden=torch.zeros((2, 2), dtype=torch.float32))
    logits_out = np.full((2, 4), -9.0, dtype=np.float32)
    values_out = np.zeros((2,), dtype=np.float32)
    entry = _entry(
        actor=actor,
        batch=SimpleNamespace(
            ids_offsets=None,
            mask=np.asarray(
                [
                    [True, True, False, False],
                    [False, True, False, True],
                ],
                dtype=np.bool_,
            ),
        ),
        row_indices=[1],
        obs_step=np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        actor_step=np.asarray([0, 1], dtype=np.int64),
        logits_out=logits_out,
        values_out=values_out,
    )

    class _Lock:
        def __init__(self) -> None:
            self.entered = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Model:
        def __init__(self) -> None:
            self.forward_obs: np.ndarray | None = None
            self.forward_actor: np.ndarray | None = None

        def forward_seat_aware(self, obs_tensor, actor_tensor, hidden_tensor):
            self.forward_obs = obs_tensor.detach().cpu().numpy().copy()
            self.forward_actor = actor_tensor.detach().cpu().numpy().copy()
            logits = torch.tensor([[0.0, 2.0, 1.0, 4.0]], dtype=torch.float32)
            values = torch.tensor([7.0], dtype=torch.float32)
            return logits, values, hidden_tensor + 5.0

    model = _Model()
    lock = _Lock()

    apply_central_snapshot_opponent_policy(
        policy_id="snapshot_a",
        entries=[entry],
        opponent_models={"snapshot_a": model},
        opponent_model_locks={"snapshot_a": lock},
        device=torch.device("cpu"),
        amp_enabled=False,
        action_selection="argmax",
        pass_action_id=0,
        action_dim=4,
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert lock.entered == 1
    assert model.forward_obs is not None
    assert np.array_equal(model.forward_obs, np.asarray([[2.0, 0.0]], dtype=np.float32))
    assert np.array_equal(model.forward_actor, np.asarray([1], dtype=np.int64))
    assert values_out.tolist() == [0.0, 7.0]
    assert actor.opponent_hidden[1].tolist() == [5.0, 5.0]
    assert logits_out[1, 3] == pytest.approx(0.0)
    assert logits_out[1, 1] == pytest.approx(-100.0)
    assert logits_out[1, 0] < -1.0e8


def test_apply_central_snapshot_outputs_sample_writes_raw_logits_values_and_hidden() -> None:
    actor = SimpleNamespace(opponent_hidden=torch.zeros((3, 2), dtype=torch.float32))
    logits_out = np.zeros((3, 4), dtype=np.float32)
    values_out = np.zeros((3,), dtype=np.float32)
    entry = _entry(
        actor=actor,
        batch=SimpleNamespace(ids_offsets=None, mask=np.ones((3, 4), dtype=np.bool_)),
        row_indices=[2, 0],
        obs_step=np.zeros((3, 2), dtype=np.float32),
        actor_step=np.zeros((3,), dtype=np.int64),
        logits_out=logits_out,
        values_out=values_out,
    )
    outputs = CentralSnapshotModelOutputs(
        logits=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        values=np.asarray([9.0, 10.0], dtype=np.float32),
        next_hidden=torch.tensor([[11.0, 12.0], [13.0, 14.0]], dtype=torch.float32),
    )

    apply_central_snapshot_outputs(
        entries=[entry],
        outputs=outputs,
        action_selection="sample",
        pass_action_id=0,
        action_dim=4,
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert logits_out[2].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert logits_out[0].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert values_out.tolist() == [10.0, 0.0, 9.0]
    assert actor.opponent_hidden[2].tolist() == [11.0, 12.0]
    assert actor.opponent_hidden[0].tolist() == [13.0, 14.0]


def test_apply_central_snapshot_outputs_argmax_rewrites_packed_logits() -> None:
    actor = SimpleNamespace(opponent_hidden=torch.zeros((3, 2), dtype=torch.float32))
    logits_out = np.full((3, 5), -3.0, dtype=np.float32)
    values_out = np.zeros((3,), dtype=np.float32)
    entry = _entry(
        actor=actor,
        batch=SimpleNamespace(
            ids_offsets=(
                np.asarray([1, 3, 4, 0, 2], dtype=np.uint32),
                np.asarray([0, 2, 3, 5], dtype=np.uint32),
            ),
            legal_action_meta=None,
            mask=None,
        ),
        row_indices=[0, 2],
        obs_step=np.zeros((3, 2), dtype=np.float32),
        actor_step=np.zeros((3,), dtype=np.int64),
        logits_out=logits_out,
        values_out=values_out,
    )
    outputs = CentralSnapshotModelOutputs(
        logits=np.asarray(
            [
                [0.0, 3.0, 1.0, 2.0, 4.0],
                [4.0, 0.0, 5.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
        values=np.asarray([6.0, 7.0], dtype=np.float32),
        next_hidden=torch.ones((2, 2), dtype=torch.float32),
    )

    apply_central_snapshot_outputs(
        entries=[entry],
        outputs=outputs,
        action_selection="argmax",
        pass_action_id=0,
        action_dim=5,
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert logits_out[0, 1] == pytest.approx(0.0)
    assert logits_out[0, 3] == pytest.approx(-100.0)
    assert logits_out[0, 2] < -1.0e8
    assert logits_out[2, 2] == pytest.approx(0.0)
    assert logits_out[2, 0] == pytest.approx(-100.0)
    assert values_out.tolist() == [6.0, 0.0, 7.0]


def test_apply_central_snapshot_outputs_argmax_rewrites_mask_logits() -> None:
    actor = SimpleNamespace(opponent_hidden=torch.zeros((2, 2), dtype=torch.float32))
    logits_out = np.full((2, 5), -3.0, dtype=np.float32)
    values_out = np.zeros((2,), dtype=np.float32)
    entry = _entry(
        actor=actor,
        batch=SimpleNamespace(
            ids_offsets=None,
            mask=np.asarray(
                [
                    [True, False, True, False, False],
                    [False, True, False, True, False],
                ],
                dtype=np.bool_,
            ),
        ),
        row_indices=[1],
        obs_step=np.zeros((2, 2), dtype=np.float32),
        actor_step=np.zeros((2,), dtype=np.int64),
        logits_out=logits_out,
        values_out=values_out,
    )
    outputs = CentralSnapshotModelOutputs(
        logits=np.asarray([[0.0, 2.0, 1.0, 3.0, 4.0]], dtype=np.float32),
        values=np.asarray([8.0], dtype=np.float32),
        next_hidden=torch.ones((1, 2), dtype=torch.float32),
    )

    apply_central_snapshot_outputs(
        entries=[entry],
        outputs=outputs,
        action_selection="argmax",
        pass_action_id=0,
        action_dim=5,
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert logits_out[1, 3] == pytest.approx(0.0)
    assert logits_out[1, 1] == pytest.approx(-100.0)
    assert logits_out[1, 4] < -1.0e8
    assert values_out.tolist() == [0.0, 8.0]
