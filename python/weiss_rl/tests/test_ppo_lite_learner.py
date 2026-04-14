from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.learners.ppo_lite_learner import PpoLiteLearner


class TinyPolicyValueModel(nn.Module):
    def __init__(self, observation_dim: int = 2, action_dim: int = 3) -> None:
        super().__init__()
        self.policy = nn.Linear(observation_dim, action_dim)
        self.value = nn.Linear(observation_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy(obs)
        values = self.value(obs).squeeze(-1)
        next_hidden = torch.zeros((int(obs.shape[0]), 1), dtype=obs.dtype, device=obs.device)
        return logits, values, next_hidden


def _packed_ids_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    offsets = [0]
    for row in np.asarray(mask, dtype=bool).reshape(-1, mask.shape[-1]):
        row_ids = np.flatnonzero(row).astype(np.uint32)
        ids.extend(int(value) for value in row_ids.tolist())
        offsets.append(len(ids))
    return np.asarray(ids, dtype=np.uint32), np.asarray(offsets, dtype=np.uint32)


def _ppo_batch() -> dict[str, object]:
    return {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [2]], dtype=np.int64),
        "legal_mask": np.asarray([[[1, 1, 0]], [[0, 1, 1]]], dtype=np.uint8),
        "old_logp": np.asarray([[-0.3], [-0.2]], dtype=np.float32),
        "old_values": np.asarray([[0.2], [0.1]], dtype=np.float32),
        "returns": np.asarray([[1.0], [0.3]], dtype=np.float32),
        "advantages": np.asarray([[0.8], [0.2]], dtype=np.float32),
        "policy_train_mask": np.asarray([[1.0], [1.0]], dtype=np.float32),
    }


def test_ppo_lite_packed_legal_actions_match_dense_mask_loss() -> None:
    torch.manual_seed(0)
    dense_model = TinyPolicyValueModel()
    packed_model = TinyPolicyValueModel()
    packed_model.load_state_dict(dense_model.state_dict())
    dense_learner = PpoLiteLearner(model=dense_model, pass_action_id=2, normalize_advantages=False)
    packed_learner = PpoLiteLearner(model=packed_model, pass_action_id=2, normalize_advantages=False)

    dense_batch = _ppo_batch()
    packed_ids, packed_offsets = _packed_ids_from_mask(np.asarray(dense_batch["legal_mask"]))
    packed_batch = dict(dense_batch)
    packed_batch["legal_actions"] = LegalActionBatch.from_packed(packed_ids, packed_offsets)
    packed_batch["legal_mask"] = None

    dense_loss, dense_metrics, _ = dense_learner._loss_and_metrics_with_context(dense_batch)
    packed_loss, packed_metrics, _ = packed_learner._loss_and_metrics_with_context(packed_batch)

    torch.testing.assert_close(dense_loss, packed_loss)
    assert dense_metrics == pytest.approx(packed_metrics)


def test_ppo_lite_update_reports_completed_epochs() -> None:
    learner = PpoLiteLearner(
        model=TinyPolicyValueModel(),
        pass_action_id=2,
        normalize_advantages=True,
        ppo_epochs=3,
        target_kl=0.0,
    )

    metrics = learner.update(_ppo_batch())

    assert metrics["ppo_epochs_completed"] == pytest.approx(3.0)
    assert "approx_kl" in metrics
    assert "clip_fraction" in metrics
