from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_outcome_preference.loss import paired_outcome_preference_loss
from weiss_rl.learners.paired_outcome_preference.retention import (
    preference_top_action_retention_loss_and_metrics,
)


def test_paired_outcome_preference_loss_can_retain_preferred_target_logp() -> None:
    current = torch.tensor([[-3.0, -2.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -2.0]])
    pair_ids = torch.tensor([[3, 3]])
    roles = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, True]])

    loss, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=0.1,
        retention_coef=2.0,
        retention_role="preferred",
    )
    loss.backward()

    assert metrics["paired_outcome_preference_retention_coef"] == 2.0
    assert metrics["paired_outcome_preference_retention_role_preferred"] == 1.0
    assert metrics["paired_outcome_preference_retention_row_count"] == 1.0
    assert metrics["paired_outcome_preference_retention_loss"] == pytest.approx(1.0)
    assert metrics["paired_outcome_preference_retention_violation_fraction"] == 1.0
    assert metrics["paired_outcome_preference_retention_logp_delta_min"] == pytest.approx(-1.0)
    assert tensors["paired_outcome_preference_retention_logp_delta"].tolist() == pytest.approx([-1.0])
    assert current.grad is not None
    assert current.grad[0, 0].item() < 0.0


def test_paired_outcome_preference_loss_can_limit_retention_to_reference_top_rows() -> None:
    current = torch.tensor([[-3.0, -3.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -2.0]])
    reference_best_non_target = torch.tensor([[-2.5, -1.0]])
    pair_ids = torch.tensor([[3, 3]])
    roles = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, True]])

    loss, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        reference_best_non_target_logp=reference_best_non_target,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=0.1,
        retention_coef=1.0,
        retention_role="all",
        retention_reference_top_only=True,
    )
    loss.backward()

    assert metrics["paired_outcome_preference_retention_reference_top_only"] == 1.0
    assert metrics["paired_outcome_preference_retention_row_count"] == 1.0
    assert tensors["paired_outcome_preference_retention_logp_delta"].tolist() == pytest.approx([-1.0])
    assert current.grad is not None
    assert current.grad[0, 0].item() < 0.0
    assert abs(current.grad[0, 1].item()) < abs(current.grad[0, 0].item())


def test_paired_outcome_preference_loss_can_preserve_target_as_top_action() -> None:
    current = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -2.0]])
    best_non_target = torch.tensor([[-1.0, -3.0]])
    pair_ids = torch.tensor([[3, 3]])
    roles = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, True]])

    loss, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        current_best_non_target_logp=best_non_target,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=0.1,
        top_action_retention_coef=1.5,
        top_action_retention_role="all",
    )
    loss.backward()

    assert metrics["paired_outcome_preference_top_action_retention_coef"] == 1.5
    assert metrics["paired_outcome_preference_top_action_retention_role_all"] == 1.0
    assert metrics["paired_outcome_preference_top_action_retention_row_count"] == 2.0
    assert metrics["paired_outcome_preference_top_action_retention_loss"] == pytest.approx(0.5)
    assert metrics["paired_outcome_preference_top_action_retention_violation_fraction"] == pytest.approx(0.5)
    assert metrics["paired_outcome_preference_top_action_retention_gap_min"] == pytest.approx(-1.0)
    assert tensors["paired_outcome_preference_top_action_retention_gap"].tolist() == pytest.approx([-1.0, 1.0])
    assert current.grad is not None
    assert current.grad[0, 0].item() < 0.0


def test_paired_outcome_preference_loss_can_limit_top_retention_to_reference_top_rows() -> None:
    current = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -2.0]])
    best_non_target = torch.tensor([[-1.0, -1.0]])
    reference_best_non_target = torch.tensor([[-3.0, -1.0]])
    pair_ids = torch.tensor([[3, 3]])
    roles = torch.tensor([[1, 0]])
    mask = torch.tensor([[True, True]])

    loss, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        current_best_non_target_logp=best_non_target,
        reference_best_non_target_logp=reference_best_non_target,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        beta=0.1,
        top_action_retention_coef=1.0,
        top_action_retention_role="all",
        top_action_retention_reference_top_only=True,
    )
    loss.backward()

    assert metrics["paired_outcome_preference_top_action_retention_reference_top_only"] == 1.0
    assert metrics["paired_outcome_preference_top_action_retention_row_count"] == 1.0
    assert tensors["paired_outcome_preference_top_action_retention_gap"].tolist() == pytest.approx([-1.0])
    assert current.grad is not None
    assert current.grad[0, 0].item() < 0.0
    assert abs(current.grad[0, 1].item()) < abs(current.grad[0, 0].item())


def test_paired_outcome_preference_loss_can_scope_retention_masks() -> None:
    current = torch.tensor([[-3.0, -1.0, -3.0, -1.0]], requires_grad=True)
    reference = torch.zeros_like(current)
    best_non_target = torch.tensor([[-2.0, -2.0, -2.0, -2.0]])
    pair_ids = torch.tensor([[1, 1, 2, 2]])
    roles = torch.tensor([[1, 0, 1, 0]])
    mask = torch.ones_like(current, dtype=torch.bool)
    scope = torch.tensor([[True, False, False, False]])

    loss, metrics, tensors = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        current_best_non_target_logp=best_non_target,
        reference_best_non_target_logp=best_non_target,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        retention_coef=1.0,
        retention_scope_mask=scope,
        top_action_retention_coef=1.0,
        top_action_retention_scope_mask=scope,
    )
    loss.backward()

    assert metrics["paired_outcome_preference_retention_scoped"] == 1.0
    assert metrics["paired_outcome_preference_retention_row_count"] == 1.0
    assert metrics["paired_outcome_preference_top_action_retention_scoped"] == 1.0
    assert metrics["paired_outcome_preference_top_action_retention_row_count"] == 1.0
    assert tensors["paired_outcome_preference_retention_logp_delta"].tolist() == pytest.approx([-3.0])
    assert tensors["paired_outcome_preference_top_action_retention_gap"].tolist() == pytest.approx([-1.0])
    assert current.grad is not None
    assert current.grad[0, 0].item() < current.grad[0, 2].item()


def test_preference_top_action_retention_helper_preserves_reference_top_filter() -> None:
    current = torch.tensor([-2.0, -2.0], dtype=torch.float32, requires_grad=True)
    best_non_target = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    reference = torch.tensor([-2.0, -2.0], dtype=torch.float32)
    reference_best_non_target = torch.tensor([-3.0, -1.0], dtype=torch.float32)

    loss, metrics, tensors = preference_top_action_retention_loss_and_metrics(
        current=current,
        best_non_target=best_non_target,
        roles=torch.tensor([1, 0], dtype=torch.long),
        valid=torch.tensor([True, True]),
        role="all",
        scope_mask=None,
        margin=0.0,
        reference=reference,
        reference_best_non_target=reference_best_non_target,
        reference_top_only=True,
        dtype=current.dtype,
        metric_prefix="paired_outcome_preference",
    )
    loss.backward()

    assert metrics["paired_outcome_preference_top_action_retention_row_count"] == 1.0
    assert metrics["paired_outcome_preference_top_action_retention_reference_top_only"] == 1.0
    assert tensors["paired_outcome_preference_top_action_retention_gap"].tolist() == pytest.approx([-1.0])
    assert current.grad is not None
    assert current.grad[0].item() < 0.0
    assert current.grad[1].item() == pytest.approx(0.0)


def test_paired_outcome_preference_loss_keeps_zero_retention_finite_with_invalid_infinities() -> None:
    current = torch.tensor([[-2.0, -torch.inf, -2.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -torch.inf, -2.0]])
    best_non_target = torch.tensor([[-1.0, -torch.inf, -1.0]])
    pair_ids = torch.tensor([[1, -1, 1]])
    roles = torch.tensor([[1, 1, 0]])
    mask = torch.tensor([[True, False, True]])

    loss, metrics, _ = paired_outcome_preference_loss(
        current_action_logp=current,
        reference_action_logp=reference,
        current_best_non_target_logp=best_non_target,
        reference_best_non_target_logp=best_non_target,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        loss_mask=mask,
        retention_coef=1.0,
        retention_reference_top_only=True,
        top_action_retention_coef=1.0,
        top_action_retention_reference_top_only=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["paired_outcome_preference_retention_row_count"] == 0.0
    assert metrics["paired_outcome_preference_top_action_retention_row_count"] == 0.0
    assert current.grad is not None
    assert torch.isfinite(current.grad[0, 0])
    assert current.grad[0, 1].item() == 0.0
