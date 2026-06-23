from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from weiss_rl.training.batches import build_learner_batch

from .training_batches_test_support import minimal_rollout


def test_build_learner_batch_requires_training_and_rewards_config_blocks() -> None:
    stack = SimpleNamespace(config=SimpleNamespace(training=None, rewards=None))

    with pytest.raises(RuntimeError, match="requires training and rewards config blocks"):
        build_learner_batch(
            stack,
            minimal_rollout(),
            np.zeros((1,), dtype=np.float32),
            action_dim=2,
            initial_hidden_state=torch.zeros((1, 1), dtype=torch.float32),
            pass_action_id=1,
        )
