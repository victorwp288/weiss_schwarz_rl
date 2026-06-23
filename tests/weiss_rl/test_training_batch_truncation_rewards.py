from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.config import load_stack_config
from weiss_rl.training.train_entrypoint import (
    MinimalRollout,
    _build_learner_batch,
)

from tests.weiss_rl._config_paths import repo_root


def test_train_build_learner_batch_does_not_double_apply_truncation_reward() -> None:
    stack = load_stack_config(repo_root() / "configs" / "presets" / "typed_local.yaml")
    rollout = MinimalRollout(
        obs=np.zeros((2, 1, 1), dtype=np.float32),
        legal_mask=np.ones((2, 1, 2), dtype=np.bool_),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=np.zeros((2, 1), dtype=np.bool_),
        truncated=np.array([[False], [True]], dtype=np.bool_),
        to_play_seat=np.zeros((2, 1), dtype=np.int64),
        behavior_logp=np.zeros((2, 1), dtype=np.float32),
        logits=np.zeros((2, 1, 2), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        bootstrap_obs=np.zeros((1, 1), dtype=np.float32),
        bootstrap_actor=np.zeros((1,), dtype=np.int64),
    )

    batch = _build_learner_batch(
        stack,
        rollout,
        np.zeros((1,), dtype=np.float32),
        action_dim=2,
        initial_hidden_state=torch.zeros((1, 1), dtype=torch.float32),
        pass_action_id=1,
    )

    assert batch["rewards"][:, 0].tolist() == pytest.approx([0.0, 0.0])
