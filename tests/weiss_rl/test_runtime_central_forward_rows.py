from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest
import torch
from weiss_rl.runtime import QueueRuntime
from weiss_rl.runtime.components.central import forward_rows as forward_rows_module

from .runtime_central_rows_test_support import bare_queue_runtime


def test_central_forward_all_rows_scatters_logits_values_and_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = bare_queue_runtime()

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
    monkeypatch.setattr(forward_rows_module, "actor_inference_model", lambda actor: model)

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
