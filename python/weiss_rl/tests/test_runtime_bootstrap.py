from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from weiss_rl.runtime.components import bootstrap as bootstrap_module
from weiss_rl.runtime.components.bootstrap import (
    add_shared_elapsed_ms,
    bootstrap_fields_from_batch,
    bootstrap_fields_from_batches,
    bootstrap_values_for_unroll,
    collector_bootstrap_fields_for_actor,
    live_bootstrap_rows,
    model_bootstrap_values,
)


@dataclass(frozen=True)
class _Unroll:
    bootstrap_obs: np.ndarray
    bootstrap_actor: np.ndarray
    final_hidden_state: np.ndarray


def test_bootstrap_values_for_unroll_prefers_value_only_path() -> None:
    class _Model:
        def __init__(self) -> None:
            self.value_calls = 0
            self.forward_calls = 0

        def value_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            hidden_state: torch.Tensor,
        ) -> torch.Tensor:
            self.value_calls += 1
            assert obs.shape == (2, 3)
            assert acting_seat.tolist() == [0, 1]
            assert hidden_state.shape == (2, 4)
            return torch.tensor([2.5, -1.5], dtype=torch.float32, device=obs.device)

        def forward_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            hidden_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self.forward_calls += 1
            raise AssertionError("forward path should not run when value_seat_aware exists")

    model = _Model()
    values = bootstrap_values_for_unroll(
        unroll=_Unroll(
            bootstrap_obs=np.zeros((3, 3), dtype=np.float32),
            bootstrap_actor=np.array([0, 2, 1], dtype=np.int64),
            final_hidden_state=np.zeros((3, 4), dtype=np.float32),
        ),
        actor_model=model,
        bootstrap_device=torch.device("cpu"),
        actor_amp_enabled=False,
    )

    assert model.value_calls == 1
    assert model.forward_calls == 0
    np.testing.assert_array_equal(values, np.array([2.5, 0.0, -1.5], dtype=np.float32))


def test_bootstrap_values_for_unroll_uses_forward_fallback() -> None:
    class _Model:
        def forward_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            hidden_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.zeros((obs.shape[0], 1), dtype=torch.float32, device=obs.device),
                torch.full((obs.shape[0],), 4.0, dtype=torch.float32, device=obs.device),
                hidden_state,
            )

    values = bootstrap_values_for_unroll(
        unroll=_Unroll(
            bootstrap_obs=np.zeros((2, 3), dtype=np.float32),
            bootstrap_actor=np.array([1, -1], dtype=np.int64),
            final_hidden_state=np.zeros((2, 4), dtype=np.float32),
        ),
        actor_model=_Model(),
        bootstrap_device=torch.device("cpu"),
        actor_amp_enabled=False,
    )

    np.testing.assert_array_equal(values, np.array([4.0, 0.0], dtype=np.float32))


def test_bootstrap_values_for_unroll_returns_zero_when_no_valid_rows() -> None:
    values = bootstrap_values_for_unroll(
        unroll=_Unroll(
            bootstrap_obs=np.zeros((2, 3), dtype=np.float32),
            bootstrap_actor=np.array([-1, 2], dtype=np.int64),
            final_hidden_state=np.zeros((2, 4), dtype=np.float32),
        ),
        actor_model=object(),
        bootstrap_device=torch.device("cpu"),
        actor_amp_enabled=False,
    )

    np.testing.assert_array_equal(values, np.array([0.0, 0.0], dtype=np.float32))


def test_bootstrap_fields_from_batch_coerces_collector_dtypes() -> None:
    batch = SimpleNamespace(
        obs=np.asarray([[1.0, 2.0]], dtype=np.float64),
        actor=np.asarray([1], dtype=np.int32),
    )

    fields = bootstrap_fields_from_batch(batch)

    assert fields.obs.dtype == np.float32
    assert fields.actor.dtype == np.int64
    assert fields.value.dtype == np.float32
    assert fields.value.tolist() == [0.0]


def test_bootstrap_fields_from_batches_preserves_batch_order() -> None:
    batches = [
        SimpleNamespace(obs=np.asarray([[1.0]], dtype=np.float32), actor=np.asarray([0], dtype=np.int64)),
        SimpleNamespace(obs=np.asarray([[2.0], [3.0]], dtype=np.float32), actor=np.asarray([1, 2], dtype=np.int64)),
    ]

    obs_steps, actor_steps, values = bootstrap_fields_from_batches(cast_list(batches))

    assert [obs.tolist() for obs in obs_steps] == [[[1.0]], [[2.0], [3.0]]]
    assert [actor.tolist() for actor in actor_steps] == [[0], [1, 2]]
    assert [value.tolist() for value in values] == [[0.0], [0.0, 0.0]]


def test_live_bootstrap_rows_only_allows_live_seats() -> None:
    assert live_bootstrap_rows(np.asarray([-1, 0, 1, 2], dtype=np.int64)).tolist() == [False, True, True, False]


def test_model_bootstrap_values_accepts_actor_seat_hidden_tensor() -> None:
    class _Model:
        def value_seat_aware(
            self,
            obs: torch.Tensor,
            acting_seat: torch.Tensor,
            hidden_state: torch.Tensor,
        ) -> torch.Tensor:
            assert obs.shape == (2, 2)
            assert acting_seat.tolist() == [0, 1]
            assert hidden_state.tolist() == [[10.0, 11.0], [14.0, 15.0]]
            return torch.asarray([3.0, 4.0], dtype=torch.float32, device=obs.device)

    values = model_bootstrap_values(
        bootstrap_obs=np.asarray([[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]], dtype=np.float32),
        bootstrap_actor=np.asarray([0, 2, 1], dtype=np.int64),
        hidden_state=torch.asarray([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]], dtype=torch.float32),
        actor_model=_Model(),
        bootstrap_device=torch.device("cpu"),
        actor_amp_enabled=False,
    )

    np.testing.assert_array_equal(values, np.asarray([3.0, 0.0, 4.0], dtype=np.float32))


def test_collector_bootstrap_fields_for_actor_skips_model_when_values_not_required() -> None:
    class _Model:
        def value_seat_aware(self, *_args: Any) -> torch.Tensor:
            raise AssertionError("model should not run when values are disabled")

    counters = {"actor_bootstrap_ms": 0}
    fields = collector_bootstrap_fields_for_actor(
        batch=SimpleNamespace(
            obs=np.asarray([[1.0, 2.0]], dtype=np.float32),
            actor=np.asarray([0], dtype=np.int64),
        ),
        actor=SimpleNamespace(seat_hidden=torch.zeros((1, 2), dtype=torch.float32)),
        actor_model=_Model(),
        bootstrap_device=torch.device("cpu"),
        actor_amp_enabled=False,
        values_required=False,
        counters=counters,
    )

    assert fields.value.tolist() == [0.0]
    assert counters["actor_bootstrap_ms"] == 0


def test_add_shared_elapsed_ms_distributes_counter_time(monkeypatch) -> None:
    readings = iter([1.5])
    monkeypatch.setattr(bootstrap_module.time, "perf_counter", lambda: next(readings))
    counters = [{"actor_bootstrap_ms": 1}, {"actor_bootstrap_ms": 3}]

    add_shared_elapsed_ms(counters=counters, key="actor_bootstrap_ms", started_at=1.0, divisor=2)

    assert counters == [{"actor_bootstrap_ms": 251}, {"actor_bootstrap_ms": 253}]


def cast_list(values: list[Any]) -> list[Any]:
    return values
