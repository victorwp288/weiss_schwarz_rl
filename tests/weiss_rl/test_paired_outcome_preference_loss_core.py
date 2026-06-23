from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_outcome_preference.loss import paired_outcome_preference_loss


def test_paired_outcome_preference_loss_pushes_preferred_over_rejected() -> None:
    current = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -2.0]])
    pair_ids = torch.tensor([[7, 7]])
    roles = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, True]])

    loss, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=1.0,
    )
    loss.backward()

    assert metrics["paired_outcome_preference_pair_count"] == 1.0
    assert metrics["paired_outcome_preference_margin_mean"] == pytest.approx(0.0)
    assert tensors["paired_outcome_preference_margins"].tolist() == [0.0]
    assert current.grad is not None
    assert current.grad[0, 0].item() < 0.0
    assert current.grad[0, 1].item() > 0.0


def test_paired_outcome_preference_loss_uses_reference_log_ratio() -> None:
    current = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    reference = torch.tensor([[-3.0, -2.0]])
    pair_ids = torch.tensor([[3, 3]])
    roles = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, True]])

    loss, metrics, _ = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=1.0,
    )

    assert loss.item() < 0.7
    assert metrics["paired_outcome_preference_margin_mean"] == pytest.approx(1.0)
    assert metrics["paired_outcome_preference_satisfied_fraction"] == 1.0


def test_paired_outcome_preference_loss_can_mean_aggregate_spans() -> None:
    current = torch.tensor([[-4.0, -2.0, -4.0, -1.0]], requires_grad=True)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[5, 5, 5, 5]])
    roles = torch.tensor([[1, 1, 0, 0]])
    mask = torch.tensor([[True, True, True, True]])

    _, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=1.0,
        aggregation="mean",
    )

    assert tensors["paired_outcome_preference_margins"].tolist() == pytest.approx([-0.5])
    assert metrics["paired_outcome_preference_current_preferred_logp_mean"] == pytest.approx(-3.0)
    assert metrics["paired_outcome_preference_current_rejected_logp_mean"] == pytest.approx(-2.5)


def test_paired_outcome_preference_loss_can_use_edge_mean_aligned_steps() -> None:
    current = torch.tensor([[-2.0, -2.0], [-1.0, -3.0]], requires_grad=True)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[5, 5], [5, 5]])
    roles = torch.tensor([[1, 0], [1, 0]])
    mask = torch.ones_like(current, dtype=torch.bool)

    loss, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=1.0,
        aggregation="edge_mean",
    )
    loss.backward()

    assert metrics["paired_outcome_preference_aggregation_edge_mean"] == 1.0
    assert metrics["paired_outcome_preference_pair_count"] == 2.0
    assert metrics["paired_outcome_preference_edge_count"] == 2.0
    assert metrics["paired_outcome_preference_candidate_pair_count"] == 1.0
    assert tensors["paired_outcome_preference_margins"].tolist() == pytest.approx([0.0, 2.0])
    assert current.grad is not None
    assert current.grad[0, 0].item() < 0.0
    assert current.grad[0, 1].item() > 0.0


def test_paired_outcome_preference_loss_ignores_masked_and_incomplete_pairs() -> None:
    current = torch.tensor([[-2.0, -2.0, -5.0]], requires_grad=True)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[1, 1, 2]])
    roles = torch.tensor([[1, 0, 1]])
    mask = torch.tensor([[True, True, True]])

    loss, metrics, _ = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
    )

    assert metrics["paired_outcome_preference_pair_count"] == 1.0
    assert metrics["paired_outcome_preference_candidate_pair_count"] == 2.0
    assert metrics["paired_outcome_preference_incomplete_pair_count"] == 1.0
    loss.backward()
    assert current.grad is not None
    assert current.grad[0, 2].item() == 0.0


def test_paired_outcome_preference_loss_returns_graph_zero_for_empty_surface() -> None:
    current = torch.tensor([[-2.0]], requires_grad=True)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[-1]])
    roles = torch.tensor([[1]])
    mask = torch.tensor([[True]])

    loss, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
    )
    loss.backward()

    assert metrics["paired_outcome_preference_pair_count"] == 0.0
    assert tensors == {}
    assert current.grad is not None
    assert current.grad[0, 0].item() == 0.0
