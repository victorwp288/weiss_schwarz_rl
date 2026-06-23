from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_outcome_preference.loss import paired_outcome_preference_loss


def test_paired_outcome_preference_loss_can_balance_groups() -> None:
    current = torch.tensor([[-2.0, -2.0, -2.0, -2.0, 0.0, -2.0]], requires_grad=True)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[1, 1, 2, 2, 3, 3]])
    roles = torch.tensor([[1, 0, 1, 0, 1, 0]])
    group_ids = torch.tensor([[0, 0, 0, 0, 1, 1]])
    mask = torch.tensor([[True, True, True, True, True, True]])

    unbalanced_loss, unbalanced_metrics, _ = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        preference_group_ids=group_ids,
        loss_mask=mask,
        beta=1.0,
        group_balance=False,
    )
    balanced_loss, balanced_metrics, _ = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        preference_group_ids=group_ids,
        loss_mask=mask,
        beta=1.0,
        group_balance=True,
    )

    assert unbalanced_metrics["paired_outcome_preference_group_balance"] == 0.0
    assert balanced_metrics["paired_outcome_preference_group_balance"] == 1.0
    assert balanced_metrics["paired_outcome_preference_group_count"] == 2.0
    assert balanced_loss.item() < unbalanced_loss.item()


def test_paired_outcome_preference_loss_can_weight_pairs() -> None:
    current = torch.tensor([[0.0, 0.0, -2.0, 0.0]], requires_grad=True)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[1, 1, 2, 2]])
    roles = torch.tensor([[1, 0, 1, 0]])
    mask = torch.ones_like(current, dtype=torch.bool)

    unweighted_loss, _, _ = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=1.0,
    )
    weighted_loss, weighted_metrics, weighted_tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        preference_pair_weights=torch.tensor([[1.0, 1.0, 8.0, 8.0]]),
        loss_mask=mask,
        beta=1.0,
    )

    assert weighted_loss.item() > unweighted_loss.item()
    assert weighted_metrics["paired_outcome_preference_pair_weighted"] == 1.0
    assert weighted_metrics["paired_outcome_preference_pair_weight_nondefault_count"] == 1.0
    assert weighted_tensors["paired_outcome_preference_pair_weights"].tolist() == pytest.approx([1.0, 8.0])
