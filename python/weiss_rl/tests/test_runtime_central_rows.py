from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest
import torch

import weiss_rl.runtime as runtime_module
from weiss_rl.runtime import QueueRuntime


def _runtime() -> QueueRuntime:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    return runtime


def test_central_value_actor_rows_preserves_sparse_actor_order(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()

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
    monkeypatch.setattr(runtime_module, "_actor_inference_model", lambda actor: model)

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
    runtime = _runtime()

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
    monkeypatch.setattr(runtime_module, "_actor_inference_model", lambda actor: model)

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
    runtime = _runtime()

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
    monkeypatch.setattr(runtime_module, "_actor_inference_model", lambda actor: model)

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


def test_central_forward_all_rows_scatters_logits_values_and_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()

    class _ForwardModel:
        def __init__(self) -> None:
            self.supports_legal_candidate_scoring = False
            self.calls: list[tuple[np.ndarray, np.ndarray]] = []

        def forward_seat_aware(self, obs, acting_seat, hidden_state, *, legal_actions=None):
            assert legal_actions is None
            self.calls.append((obs.detach().cpu().numpy(), acting_seat.detach().cpu().numpy()))
            logits = torch.stack((obs[:, 0], acting_seat.to(obs.dtype)), dim=1)
            values = obs[:, 0] + 0.5
            return logits, values, hidden_state + 7.0

    model = _ForwardModel()
    monkeypatch.setattr(runtime_module, "_actor_inference_model", lambda actor: model)

    actor_a = SimpleNamespace(seat_hidden=torch.zeros((2, 2), dtype=torch.float32))
    actor_b = SimpleNamespace(seat_hidden=torch.ones((1, 2), dtype=torch.float32))
    logits_a = np.zeros((2, 2), dtype=np.float32)
    logits_b = np.zeros((1, 2), dtype=np.float32)
    values_a = np.zeros((2,), dtype=np.float32)
    values_b = np.zeros((1,), dtype=np.float32)

    QueueRuntime._central_forward_all_rows(
        runtime,
        actors=[cast(Any, actor_a), cast(Any, actor_b)],
        batches=None,
        obs_steps=[
            np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            np.asarray([[3.0, 0.0]], dtype=np.float32),
        ],
        actor_steps=[
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        ],
        logits_outs=[logits_a, logits_b],
        values_outs=[values_a, values_b],
    )

    assert len(model.calls) == 1
    npt.assert_array_equal(model.calls[0][0][:, 0], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    npt.assert_array_equal(logits_a, np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32))
    npt.assert_array_equal(logits_b, np.asarray([[3.0, 1.0]], dtype=np.float32))
    npt.assert_array_equal(values_a, np.asarray([1.5, 2.5], dtype=np.float32))
    npt.assert_array_equal(values_b, np.asarray([3.5], dtype=np.float32))
    npt.assert_array_equal(actor_a.seat_hidden.numpy(), np.full((2, 2), 7.0, dtype=np.float32))
    npt.assert_array_equal(actor_b.seat_hidden.numpy(), np.full((1, 2), 8.0, dtype=np.float32))


def test_central_sample_policy_rows_routes_fractional_heuristic_and_model_rows() -> None:
    runtime = _runtime()
    runtime_any = cast(Any, runtime)
    runtime_any._actor_policy_backend = "heuristic_public"
    runtime_any._active_actor_heuristic_fraction = lambda: 0.5

    calls: list[tuple[str, list[np.ndarray]]] = []

    def record_model(**kwargs: Any) -> None:
        calls.append(("model", [np.array(rows, copy=True) for rows in kwargs["row_indices_by_actor"]]))

    def record_heuristic(**kwargs: Any) -> None:
        calls.append(("heuristic", [np.array(rows, copy=True) for rows in kwargs["row_indices_by_actor"]]))

    runtime_any._central_sample_policy_rows_ids_model = record_model
    runtime_any._central_sample_policy_rows_ids_heuristic = record_heuristic

    class _FixedRng:
        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = np.asarray(values, dtype=np.float64)

        def random(self, size: int) -> np.ndarray:
            assert size <= self.values.shape[0]
            return self.values[:size]

    actors = [
        SimpleNamespace(rng=_FixedRng((0.1, 0.9, 0.4))),
        SimpleNamespace(rng=_FixedRng((0.6, 0.2))),
        SimpleNamespace(rng=_FixedRng(())),
    ]
    row_indices_by_actor = [
        np.asarray([2, 4, 6], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        np.asarray([], dtype=np.int64),
    ]

    QueueRuntime._central_sample_policy_rows_ids(
        runtime,
        actors=cast(Any, actors),
        batches=[object(), object(), object()],
        obs_steps=[np.empty((0, 1), dtype=np.float32)] * 3,
        actor_steps=[np.empty((0,), dtype=np.int64)] * 3,
        row_indices_by_actor=row_indices_by_actor,
        values_outs=[np.empty((0,), dtype=np.float32)] * 3,
        actions_outs=[np.empty((0,), dtype=np.int64)] * 3,
        logp_outs=[np.empty((0,), dtype=np.float32)] * 3,
    )

    assert [label for label, _rows in calls] == ["heuristic", "model"]
    heuristic_rows = calls[0][1]
    model_rows = calls[1][1]
    npt.assert_array_equal(heuristic_rows[0], np.asarray([2, 6], dtype=np.int64))
    npt.assert_array_equal(heuristic_rows[1], np.asarray([3], dtype=np.int64))
    npt.assert_array_equal(heuristic_rows[2], np.asarray([], dtype=np.int64))
    npt.assert_array_equal(model_rows[0], np.asarray([4], dtype=np.int64))
    npt.assert_array_equal(model_rows[1], np.asarray([1], dtype=np.int64))
    npt.assert_array_equal(model_rows[2], np.asarray([], dtype=np.int64))
