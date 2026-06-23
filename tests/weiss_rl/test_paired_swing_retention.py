from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_swing.loss import (
    packed_paired_swing_margin_loss,
    packed_target_action_retention_loss,
    packed_top_action_retention_loss,
)
from weiss_rl.learners.paired_swing.margin_retention import paired_swing_margin_retention_loss_and_metrics
from weiss_rl.learners.paired_swing.top_retention import paired_swing_top_action_retention_rows


def test_paired_swing_margin_retention_preserves_negative_reference_path() -> None:
    positive_actions = torch.tensor([[1, 2, 1]], dtype=torch.long)
    negative_actions = torch.tensor([[2, 1, 2]], dtype=torch.long)

    loss, metrics, tensors = paired_swing_margin_retention_loss_and_metrics(
        current_margin_by_row=torch.tensor([0.0, -1.0, 9.0], dtype=torch.float32),
        reference_packed_logits=torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0, 1.0], dtype=torch.float32),
        legal_ids=torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.long),
        legal_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.long),
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        flat_positive_actions=positive_actions.reshape(-1),
        flat_negative_actions=negative_actions.reshape(-1),
        active_rows=torch.tensor([True, True, True]),
        supported=torch.tensor([True, True, False]),
        supported_weight=torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32),
        pass_action_id=0,
        compare_to="negative",
        retention_margin=1.5,
        metric_prefix="swing",
    )

    assert loss.item() == pytest.approx(0.5)
    assert metrics["swing_margin_retention_rows"] == 2.0
    assert metrics["swing_margin_retention_violation_fraction"] == 1.0
    assert metrics["swing_margin_delta_mean"] == pytest.approx(1.0)
    assert metrics["swing_margin_delta_min"] == pytest.approx(1.0)
    assert tensors["paired_swing_margin_delta"].tolist() == pytest.approx([1.0, 1.0])


def test_paired_swing_margin_retention_preserves_top_other_reference_support() -> None:
    positive_actions = torch.tensor([[1, 1]], dtype=torch.long)
    negative_actions = torch.tensor([[2, 2]], dtype=torch.long)

    loss, metrics, tensors = paired_swing_margin_retention_loss_and_metrics(
        current_margin_by_row=torch.tensor([-1.0, 0.0], dtype=torch.float32),
        reference_packed_logits=torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float32),
        legal_ids=torch.tensor([1, 2, 3, 1], dtype=torch.long),
        legal_offsets=torch.tensor([0, 3, 4], dtype=torch.long),
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        flat_positive_actions=positive_actions.reshape(-1),
        flat_negative_actions=negative_actions.reshape(-1),
        active_rows=torch.tensor([True, True]),
        supported=torch.tensor([True, True]),
        supported_weight=torch.tensor([2.0, 5.0], dtype=torch.float32),
        pass_action_id=0,
        compare_to="top_other",
        retention_margin=0.25,
        metric_prefix="swing",
    )

    assert loss.item() == 0.0
    assert metrics["swing_margin_retention_rows"] == 1.0
    assert metrics["swing_margin_retention_violation_fraction"] == 0.0
    assert metrics["swing_margin_delta_mean"] == pytest.approx(1.0)
    assert tensors["paired_swing_margin_delta"].tolist() == pytest.approx([1.0])


def test_paired_swing_top_action_retention_rows_preserve_gap_weights_and_skips() -> None:
    rows = paired_swing_top_action_retention_rows(
        packed_logits=torch.tensor([0.0, 1.0, 5.0, 4.0, 1.0], dtype=torch.float32),
        reference_packed_logits=torch.tensor([1.0, 0.0, 5.0, 0.0, 2.0], dtype=torch.float32),
        legal_offsets=torch.tensor([0, 2, 3, 5], dtype=torch.long),
        supported=torch.tensor([True, True, True]),
        supported_weight=torch.tensor([2.0, 5.0, 3.0], dtype=torch.float32),
    )

    assert rows is not None
    assert rows.gaps.tolist() == pytest.approx([-1.0, -3.0])
    assert rows.weights.tolist() == pytest.approx([2.0, 3.0])
    assert rows.agreements.tolist() == pytest.approx([0.0, 0.0])


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
