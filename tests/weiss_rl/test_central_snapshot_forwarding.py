from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import torch
from weiss_rl.runtime.components.opponents.central_snapshot_opponents import (
    build_central_snapshot_forward_batch,
    run_central_snapshot_model,
)

from tests.weiss_rl.central_snapshot_opponents_test_support import make_central_opponent_entry


def test_build_central_snapshot_forward_batch_preserves_entry_and_row_order() -> None:
    actor_a = SimpleNamespace(opponent_hidden=torch.arange(12, dtype=torch.float32).reshape(4, 3))
    actor_b = SimpleNamespace(opponent_hidden=torch.arange(12, 24, dtype=torch.float32).reshape(4, 3))
    entries = [
        make_central_opponent_entry(
            actor=actor_a,
            batch=SimpleNamespace(),
            row_indices=[2, 0],
            obs_step=np.asarray([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32),
            actor_step=np.asarray([0, 1, 1, 0], dtype=np.int64),
            logits_out=None,
            values_out=np.zeros((4,), dtype=np.float32),
        ),
        make_central_opponent_entry(
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
    entry = make_central_opponent_entry(
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
