from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from weiss_rl.runtime.components.opponents.central_snapshot_opponents import (
    apply_central_snapshot_opponent_policy,
)

from tests.weiss_rl.central_snapshot_opponents_test_support import make_central_opponent_entry


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
    entry = make_central_opponent_entry(
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

        def __exit__(self, exc_type, exc, _tb):
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
