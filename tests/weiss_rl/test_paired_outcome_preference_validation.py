from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_outcome_preference.loss import paired_outcome_preference_loss


def test_paired_outcome_preference_loss_rejects_bad_pair_weights() -> None:
    with pytest.raises(ValueError, match="preference_pair_weights must be positive"):
        paired_outcome_preference_loss(
            current_action_logp=torch.zeros((1, 2)),
            reference_action_logp=torch.zeros((1, 2)),
            preference_pair_ids=torch.zeros((1, 2), dtype=torch.long),
            preference_role=torch.tensor([[1, 0]], dtype=torch.long),
            preference_pair_weights=torch.tensor([[1.0, 0.0]]),
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
        )


def test_paired_outcome_preference_loss_requires_group_ids_when_balancing() -> None:
    with pytest.raises(ValueError, match="preference_group_ids"):
        paired_outcome_preference_loss(
            current_action_logp=torch.zeros((1, 2)),
            reference_action_logp=torch.zeros((1, 2)),
            preference_pair_ids=torch.zeros((1, 2), dtype=torch.long),
            preference_role=torch.zeros((1, 2), dtype=torch.long),
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
            group_balance=True,
        )


def test_paired_outcome_preference_loss_rejects_invalid_retention_args() -> None:
    with pytest.raises(ValueError, match="retention_role"):
        paired_outcome_preference_loss(
            current_action_logp=torch.zeros((1, 2)),
            reference_action_logp=torch.zeros((1, 2)),
            preference_pair_ids=torch.zeros((1, 2), dtype=torch.long),
            preference_role=torch.zeros((1, 2), dtype=torch.long),
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
            retention_role="mystery",
        )
    with pytest.raises(ValueError, match="retention_coef"):
        paired_outcome_preference_loss(
            current_action_logp=torch.zeros((1, 2)),
            reference_action_logp=torch.zeros((1, 2)),
            preference_pair_ids=torch.zeros((1, 2), dtype=torch.long),
            preference_role=torch.zeros((1, 2), dtype=torch.long),
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
            retention_coef=-1.0,
        )
    with pytest.raises(ValueError, match="top_action_retention_role"):
        paired_outcome_preference_loss(
            current_action_logp=torch.zeros((1, 2)),
            reference_action_logp=torch.zeros((1, 2)),
            preference_pair_ids=torch.zeros((1, 2), dtype=torch.long),
            preference_role=torch.zeros((1, 2), dtype=torch.long),
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
            top_action_retention_role="mystery",
        )
    with pytest.raises(ValueError, match="reference_best_non_target_logp"):
        paired_outcome_preference_loss(
            current_action_logp=torch.zeros((1, 2)),
            reference_action_logp=torch.zeros((1, 2)),
            preference_pair_ids=torch.zeros((1, 2), dtype=torch.long),
            preference_role=torch.tensor([[1, 0]], dtype=torch.long),
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
            retention_reference_top_only=True,
        )


def test_paired_outcome_preference_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="reference_action_logp"):
        paired_outcome_preference_loss(
            current_action_logp=torch.zeros((1, 2)),
            reference_action_logp=torch.zeros((1, 1)),
            preference_pair_ids=torch.zeros((1, 2), dtype=torch.long),
            preference_role=torch.zeros((1, 2), dtype=torch.long),
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
        )
