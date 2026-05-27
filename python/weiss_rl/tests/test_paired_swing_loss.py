from __future__ import annotations

import pytest
import torch

from weiss_rl.learners.paired_swing_loss import (
    packed_paired_swing_margin_loss,
    packed_target_action_retention_loss,
    packed_top_action_retention_loss,
)


def test_packed_paired_swing_margin_loss_penalizes_positive_below_negative() -> None:
    packed_logits = torch.tensor([0.0, 1.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2], dtype=torch.long)
    positive_actions = torch.tensor([[1]], dtype=torch.long)
    negative_actions = torch.tensor([[2]], dtype=torch.long)
    valid = torch.tensor([[True]])
    loss_mask = torch.tensor([[1.0]], dtype=torch.float32)

    loss, metrics, context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.25,
        pass_action_id=0,
    )

    assert loss.item() == pytest.approx(1.25)
    assert metrics["paired_swing_rows"] == 1.0
    assert metrics["paired_swing_candidate_rows"] == 1.0
    assert metrics["paired_swing_margin_mean"] == pytest.approx(-1.0)
    assert metrics["paired_swing_satisfied_fraction"] == 0.0
    assert "paired_swing_margins" in context


def test_packed_paired_swing_margin_loss_can_compare_positive_to_top_other() -> None:
    packed_logits = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    legal_offsets = torch.tensor([0, 3], dtype=torch.long)
    positive_actions = torch.tensor([[1]], dtype=torch.long)
    negative_actions = torch.tensor([[2]], dtype=torch.long)
    valid = torch.tensor([[True]])
    loss_mask = torch.tensor([[1.0]], dtype=torch.float32)

    loss, metrics, context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.25,
        pass_action_id=0,
        compare_to="top_other",
    )

    assert loss.item() == pytest.approx(2.25)
    assert metrics["paired_swing_compare_to_top_other"] == 1.0
    assert metrics["paired_swing_margin_mean"] == pytest.approx(-2.0)
    assert context["paired_swing_margins"].tolist() == pytest.approx([-2.0])


def test_packed_paired_swing_margin_loss_can_retain_reference_margin() -> None:
    packed_logits = torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True)
    reference_logits = torch.tensor([1.0, 0.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2], dtype=torch.long)
    positive_actions = torch.tensor([[1]], dtype=torch.long)
    negative_actions = torch.tensor([[2]], dtype=torch.long)
    valid = torch.tensor([[True]])
    loss_mask = torch.tensor([[1.0]], dtype=torch.float32)

    loss, metrics, context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        reference_packed_logits=reference_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.0,
        pass_action_id=0,
        margin_retention_coef=0.5,
    )
    loss.backward()

    assert metrics["paired_swing_margin_retention_coef"] == 0.5
    assert metrics["paired_swing_margin_retention_loss"] == pytest.approx(2.0)
    assert metrics["paired_swing_margin_delta_mean"] == pytest.approx(-2.0)
    assert metrics["paired_swing_margin_retention_violation_fraction"] == 1.0
    assert context["paired_swing_margin_delta"].tolist() == pytest.approx([-2.0])
    assert packed_logits.grad is not None
    assert packed_logits.grad[0].item() < 0.0
    assert packed_logits.grad[1].item() > 0.0


def test_packed_paired_swing_margin_loss_can_retain_reference_top_action() -> None:
    packed_logits = torch.tensor([0.0, 1.0], dtype=torch.float32, requires_grad=True)
    reference_logits = torch.tensor([1.0, 0.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2], dtype=torch.long)
    positive_actions = torch.tensor([[2]], dtype=torch.long)
    negative_actions = torch.tensor([[1]], dtype=torch.long)
    valid = torch.tensor([[True]])
    loss_mask = torch.tensor([[1.0]], dtype=torch.float32)

    loss, metrics, context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        reference_packed_logits=reference_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.0,
        pass_action_id=0,
        top_action_retention_coef=0.5,
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.5)
    assert metrics["paired_swing_top_action_retention_coef"] == 0.5
    assert metrics["paired_swing_top_action_retention_loss"] == pytest.approx(1.0)
    assert metrics["paired_swing_top_action_retention_rows"] == 1.0
    assert metrics["paired_swing_top_action_retention_violation_fraction"] == 1.0
    assert metrics["paired_swing_top_action_retention_gap_min"] == pytest.approx(-1.0)
    assert metrics["paired_swing_top_action_retention_agreement_fraction"] == 0.0
    assert context["paired_swing_top_action_retention_gap"].tolist() == pytest.approx([-1.0])
    assert packed_logits.grad is not None
    assert packed_logits.grad[0].item() < 0.0
    assert packed_logits.grad[1].item() > 0.0


def test_packed_top_action_retention_loss_protects_all_masked_rows() -> None:
    packed_logits = torch.tensor([0.0, 1.0, 1.0, 0.0, 3.0, 0.0], dtype=torch.float32, requires_grad=True)
    reference_logits = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 3.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    loss_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    loss, metrics, context = packed_top_action_retention_loss(
        packed_logits=packed_logits,
        reference_packed_logits=reference_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        loss_mask=loss_mask,
    )
    loss.backward()

    assert loss.item() == pytest.approx(2.0)
    assert metrics["paired_swing_full_surface_top_action_retention_rows"] == 2.0
    assert metrics["paired_swing_full_surface_top_action_retention_violation_fraction"] == 1.0
    assert metrics["paired_swing_full_surface_top_action_retention_gap_min"] == pytest.approx(-3.0)
    assert context["paired_swing_full_surface_top_action_retention_gap"].tolist() == pytest.approx([-1.0, -3.0])
    assert packed_logits.grad is not None
    assert packed_logits.grad[0].item() < 0.0
    assert packed_logits.grad[1].item() > 0.0
    assert packed_logits.grad[2].item() == pytest.approx(0.0)
    assert packed_logits.grad[3].item() == pytest.approx(0.0)
    assert packed_logits.grad[4].item() > 0.0
    assert packed_logits.grad[5].item() < 0.0


def test_packed_target_action_retention_loss_pushes_targets_above_best_other() -> None:
    packed_logits = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32, requires_grad=True)
    legal_ids = torch.tensor([1, 2, 1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2, 4], dtype=torch.long)
    target_actions = torch.tensor([[1, 2]], dtype=torch.long)
    loss_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    loss, metrics, context = packed_target_action_retention_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        target_actions=target_actions,
        target_valid=None,
        loss_mask=loss_mask,
        retention_margin=0.25,
    )
    loss.backward()

    assert loss.item() == pytest.approx(1.75)
    assert metrics["paired_swing_full_surface_target_retention_rows"] == 2.0
    assert metrics["paired_swing_full_surface_target_retention_target_top_fraction"] == 0.0
    assert metrics["paired_swing_full_surface_target_retention_margin_min"] == pytest.approx(-2.0)
    assert context["paired_swing_full_surface_target_retention_margin"].tolist() == pytest.approx([-1.0, -2.0])
    assert packed_logits.grad is not None
    assert packed_logits.grad[0].item() < 0.0
    assert packed_logits.grad[1].item() > 0.0
    assert packed_logits.grad[2].item() > 0.0
    assert packed_logits.grad[3].item() < 0.0


def test_packed_paired_swing_margin_loss_ignores_matching_actions() -> None:
    loss, metrics, context = packed_paired_swing_margin_loss(
        packed_logits=torch.tensor([1.0, 0.0], dtype=torch.float32),
        legal_ids=torch.tensor([1, 2], dtype=torch.long),
        legal_offsets=torch.tensor([0, 2], dtype=torch.long),
        positive_actions=torch.tensor([[1]], dtype=torch.long),
        negative_actions=torch.tensor([[1]], dtype=torch.long),
        negative_valid=torch.tensor([[True]]),
        loss_mask=torch.tensor([[1.0]], dtype=torch.float32),
        margin=0.25,
        pass_action_id=0,
    )

    assert loss.item() == 0.0
    assert metrics["paired_swing_candidate_rows"] == 0.0
    assert metrics["paired_swing_rows"] == 0.0
    assert context == {}


def test_packed_paired_swing_episode_mean_loss_hinges_once_per_episode() -> None:
    packed_logits = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2, 1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2, 4], dtype=torch.long)
    positive_actions = torch.tensor([[1], [1]], dtype=torch.long)
    negative_actions = torch.tensor([[2], [2]], dtype=torch.long)
    valid = torch.tensor([[True], [True]])
    loss_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

    loss, metrics, _context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.25,
        pass_action_id=0,
        loss_scope="episode_mean",
    )

    assert loss.item() == pytest.approx(0.25)
    assert metrics["paired_swing_loss_scope_episode_mean"] == 1.0
    assert metrics["paired_swing_episode_count"] == 1.0
    assert metrics["paired_swing_margin_mean"] == pytest.approx(0.0)


def test_packed_paired_swing_label_mean_loss_balances_source_labels() -> None:
    packed_logits = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    legal_ids = torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.long)
    legal_offsets = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    positive_actions = torch.tensor([[1, 1, 1]], dtype=torch.long)
    negative_actions = torch.tensor([[2, 2, 2]], dtype=torch.long)
    valid = torch.tensor([[True, True, True]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    group_ids = torch.tensor([[1, 1, 2]], dtype=torch.long)

    loss, metrics, _context = packed_paired_swing_margin_loss(
        packed_logits=packed_logits,
        legal_ids=legal_ids,
        legal_offsets=legal_offsets,
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        negative_valid=valid,
        loss_mask=loss_mask,
        margin=0.25,
        pass_action_id=0,
        loss_scope="label_mean",
        group_ids=group_ids,
    )

    assert loss.item() == pytest.approx(0.625)
    assert metrics["paired_swing_loss_scope_label_mean"] == 1.0
    assert metrics["paired_swing_label_count"] == 2.0
    assert metrics["paired_swing_margin_mean"] == pytest.approx(0.0)
