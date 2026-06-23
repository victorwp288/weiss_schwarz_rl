from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.training.batches import bootstrap_values

from .training_batches_test_support import minimal_rollout


def test_bootstrap_values_only_evaluates_live_actor_rows() -> None:
    class _Model:
        def forward_seat_aware(self, obs, actor, hidden):
            assert obs.cpu().numpy().reshape(-1).tolist() == [0.0, 1.0]
            assert actor.cpu().numpy().tolist() == [0, 1]
            assert tuple(hidden.shape) == (2, 2)
            return None, torch.tensor([3.0, 4.0], dtype=torch.float32), None

    values = bootstrap_values(
        _Model(),
        minimal_rollout(truncated_bootstrap_actor=np.array([0, 1, -1], dtype=np.int64)),
        torch.zeros((3, 2), dtype=torch.float32),
        device=torch.device("cpu"),
    )

    assert values.tolist() == pytest.approx([3.0, 4.0, 0.0])
