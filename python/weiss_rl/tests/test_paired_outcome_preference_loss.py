from __future__ import annotations

import pytest
import torch

from weiss_rl.learners.paired_outcome_preference_loss import paired_outcome_preference_loss


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


def test_paired_outcome_preference_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="reference_action_logp"):
        paired_outcome_preference_loss(
            current_action_logp=torch.zeros((1, 2)),
            reference_action_logp=torch.zeros((1, 1)),
            preference_pair_ids=torch.zeros((1, 2), dtype=torch.long),
            preference_role=torch.zeros((1, 2), dtype=torch.long),
            loss_mask=torch.ones((1, 2), dtype=torch.bool),
        )
