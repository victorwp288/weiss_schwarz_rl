from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest
import torch
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.central import value_rows as value_rows_module

from .runtime_central_rows_test_support import bare_queue_runtime


def test_central_value_actor_rows_preserves_sparse_actor_order(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = bare_queue_runtime()

    class _ValueOnlyModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, torch.Tensor]] = []

        def value_seat_aware(self, obs, acting_seat, hidden_state):
            self.calls.append((obs.detach().cpu().numpy(), acting_seat.detach().cpu().numpy(), hidden_state.clone()))
            return obs[:, 0] + (acting_seat.to(obs.dtype) * 100.0) + hidden_state.sum(dim=1)

        def forward_seat_aware(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("value_seat_aware should be preferred")

    model = _ValueOnlyModel()
    monkeypatch.setattr(value_rows_module, "actor_inference_model", lambda actor: model)

    actor_a = SimpleNamespace(seat_hidden=torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    actor_b = SimpleNamespace(seat_hidden=torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]))
    values_a = np.zeros((2,), dtype=np.float32)
    values_b = np.zeros((3,), dtype=np.float32)

    QueueRuntime._central_value_actor_rows(
        runtime,
        actors=[cast(Any, actor_a), cast(Any, actor_b)],
        obs_steps=[
            np.asarray([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32),
            np.asarray([[30.0, 0.0], [40.0, 0.0], [50.0, 0.0]], dtype=np.float32),
        ],
        actor_steps=[
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1, 0, 1], dtype=np.int64),
        ],
        row_indices_by_actor=[
            np.asarray([1], dtype=np.int64),
            np.asarray([0, 2], dtype=np.int64),
        ],
        values_outs=[values_a, values_b],
    )

    assert len(model.calls) == 1
    npt.assert_array_equal(model.calls[0][0][:, 0], np.asarray([20.0, 30.0, 50.0], dtype=np.float32))
    npt.assert_array_equal(values_a, np.asarray([0.0, 124.0], dtype=np.float32))
    npt.assert_array_equal(values_b, np.asarray([136.0, 0.0, 160.0], dtype=np.float32))
    npt.assert_array_equal(actor_a.seat_hidden.numpy(), np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))
    npt.assert_array_equal(
        actor_b.seat_hidden.numpy(),
        np.asarray([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=np.float32),
    )


def test_central_value_and_advance_actor_rows_prefers_packed_trunk(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = bare_queue_runtime()

    class _PackedTrunkModel:
        def __init__(self) -> None:
            self.calls = 0

        def forward_trunk_packed_seat_aware(self, obs, acting_seat, hidden_state):
            self.calls += 1
            value = obs[:, 0] + acting_seat.to(obs.dtype)
            return hidden_state, hidden_state, None, value, hidden_state + 10.0

        def value_seat_aware(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("packed trunk should be preferred")

    model = _PackedTrunkModel()
    monkeypatch.setattr(value_rows_module, "actor_inference_model", lambda actor: model)

    actor_a = SimpleNamespace(seat_hidden=torch.zeros((2, 2), dtype=torch.float32))
    actor_b = SimpleNamespace(seat_hidden=torch.ones((2, 2), dtype=torch.float32))
    values_a = np.zeros((2,), dtype=np.float32)
    values_b = np.zeros((2,), dtype=np.float32)

    QueueRuntime._central_value_and_advance_actor_rows(
        runtime,
        actors=[cast(Any, actor_a), cast(Any, actor_b)],
        obs_steps=[
            np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            np.asarray([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        ],
        actor_steps=[
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1, 0], dtype=np.int64),
        ],
        row_indices_by_actor=[
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        ],
        values_outs=[values_a, values_b],
    )

    assert model.calls == 1
    npt.assert_array_equal(values_a, np.asarray([1.0, 0.0], dtype=np.float32))
    npt.assert_array_equal(values_b, np.asarray([0.0, 4.0], dtype=np.float32))
    npt.assert_array_equal(actor_a.seat_hidden.numpy(), np.asarray([[10.0, 10.0], [0.0, 0.0]], dtype=np.float32))
    npt.assert_array_equal(actor_b.seat_hidden.numpy(), np.asarray([[1.0, 1.0], [11.0, 11.0]], dtype=np.float32))


def test_central_advance_actor_rows_updates_only_selected_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = bare_queue_runtime()

    class _AdvanceOnlyModel:
        def __init__(self) -> None:
            self.calls = 0

        def advance_seat_hidden(self, obs, acting_seat, hidden_state):
            self.calls += 1
            del obs, acting_seat
            return hidden_state + 3.0

        def forward_seat_aware(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("advance_seat_hidden should be preferred")

    model = _AdvanceOnlyModel()
    monkeypatch.setattr(value_rows_module, "actor_inference_model", lambda actor: model)

    actor_a = SimpleNamespace(seat_hidden=torch.zeros((3, 2), dtype=torch.float32))
    actor_b = SimpleNamespace(seat_hidden=torch.ones((2, 2), dtype=torch.float32))

    QueueRuntime._central_advance_actor_rows(
        runtime,
        actors=[cast(Any, actor_a), cast(Any, actor_b)],
        obs_steps=[
            np.zeros((3, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        ],
        actor_steps=[
            np.asarray([0, 1, 0], dtype=np.int64),
            np.asarray([1, 0], dtype=np.int64),
        ],
        row_indices_by_actor=[
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        ],
    )

    assert model.calls == 1
    npt.assert_array_equal(actor_a.seat_hidden.numpy(), np.asarray([[3.0, 3.0], [0.0, 0.0], [3.0, 3.0]]))
    npt.assert_array_equal(actor_b.seat_hidden.numpy(), np.asarray([[1.0, 1.0], [4.0, 4.0]]))
