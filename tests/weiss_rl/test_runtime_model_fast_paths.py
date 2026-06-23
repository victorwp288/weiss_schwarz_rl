from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import torch
from weiss_rl.runtime import QueueRuntime

from .runtime_test_support import _make_runtime_unroll


def test_advance_hidden_only_prefers_hidden_only_model_path() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False

    class _FakeModel:
        def __init__(self) -> None:
            self.advance_calls = 0
            self.forward_calls = 0

        def advance_seat_hidden(self, obs, acting_seat, hidden_state):
            self.advance_calls += 1
            assert tuple(obs.shape) == (2, 3)
            assert tuple(acting_seat.shape) == (2,)
            return hidden_state + 5.0

        def forward_seat_aware(self, obs, acting_seat, hidden_state):
            self.forward_calls += 1
            raise AssertionError("forward_seat_aware should not run when advance_seat_hidden exists")

    hidden = torch.zeros((4, 2), dtype=torch.float32)
    model = _FakeModel()

    QueueRuntime._advance_hidden_only(
        runtime,
        model=model,
        hidden_state=hidden,
        row_indices=np.array([1, 3], dtype=np.int64),
        obs_step=np.zeros((4, 3), dtype=np.float32),
        actor_step=np.array([0, 1, 0, 1], dtype=np.int64),
    )

    assert model.advance_calls == 1
    assert model.forward_calls == 0
    npt.assert_array_equal(hidden.numpy(), np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 0.0], [5.0, 5.0]], dtype=np.float32))


def test_bootstrap_values_prefers_value_only_model_path() -> None:
    runtime = object.__new__(QueueRuntime)
    runtime_any = cast(Any, runtime)
    runtime_any._device = torch.device("cpu")
    runtime_any._actor_amp_enabled = False
    runtime_any._bootstrap_models = None
    runtime_any._actors = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.value_calls = 0
            self.forward_calls = 0

        def value_seat_aware(self, obs, acting_seat, hidden_state):
            self.value_calls += 1
            return torch.full((obs.shape[0],), 7.0, dtype=torch.float32, device=obs.device)

        def forward_seat_aware(self, obs, acting_seat, hidden_state):
            self.forward_calls += 1
            raise AssertionError("forward_seat_aware should not run when value_seat_aware exists")

    model = _FakeModel()
    runtime_any._actors[0] = cast(Any, SimpleNamespace(model=model))
    unroll = _make_runtime_unroll(actor_id=0, unroll_seq=0, behavior_policy_version=0)
    unroll = replace(
        unroll,
        bootstrap_obs=np.zeros((3, 2), dtype=np.float32),
        bootstrap_actor=np.array([0, 2, 1], dtype=np.int64),
        final_hidden_state=np.zeros((3, 4), dtype=np.float32),
    )

    values = QueueRuntime._bootstrap_values(runtime, unroll)

    assert model.value_calls == 1
    assert model.forward_calls == 0
    npt.assert_array_equal(values, np.array([7.0, 0.0, 7.0], dtype=np.float32))
