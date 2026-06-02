from __future__ import annotations

import pytest
import torch

from weiss_rl.learners.paired_swing.comparison import paired_swing_margin_comparison_rows
from weiss_rl.learners.paired_swing.inputs import prepare_paired_swing_loss_inputs
from weiss_rl.learners.paired_swing.loss import (
    packed_paired_swing_margin_loss,
    packed_target_action_retention_loss,
    packed_top_action_retention_loss,
)
from weiss_rl.learners.paired_swing.margin_retention import paired_swing_margin_retention_loss_and_metrics
from weiss_rl.learners.paired_swing.metrics import (
    paired_swing_final_metrics,
    paired_swing_supported_rows,
)
from weiss_rl.learners.paired_swing.rows import positive_vs_top_other_margin_by_row
from weiss_rl.learners.paired_swing.scope import paired_swing_scoped_margin_loss
from weiss_rl.learners.paired_swing.top_retention import paired_swing_top_action_retention_rows


def test_prepare_paired_swing_loss_inputs_normalizes_options_and_counts_active_rows() -> None:
    prepared = prepare_paired_swing_loss_inputs(
        packed_logits=torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32),
        reference_packed_logits=None,
        positive_actions=torch.tensor([[1, 2, 3]], dtype=torch.long),
        negative_actions=torch.tensor([[2, 2, -1]], dtype=torch.long),
        negative_valid=torch.tensor([[True, True, True]]),
        loss_mask=torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        loss_scope=" Episode_Mean ",
        compare_to=" Top_Other ",
        margin_retention_coef=0.0,
        margin_retention_margin=0.0,
        top_action_retention_coef=0.0,
        top_action_retention_margin=0.0,
        metric_prefix="swing",
    )

    assert prepared.options.loss_scope == "episode_mean"
    assert prepared.options.compare_to == "top_other"
    assert prepared.active_rows.tolist() == [True, False, False]
    assert prepared.raw_weight_total == pytest.approx(1.0)
    assert prepared.train_weight_total == pytest.approx(2.0)
    assert prepared.candidate_metrics == {"swing_candidate_rows": 1.0, "swing_distinct_fraction": 0.5}


def test_paired_swing_margin_comparison_rows_preserves_negative_action_path() -> None:
    positive_actions = torch.tensor([[1, 2]], dtype=torch.long)
    negative_actions = torch.tensor([[2, 1]], dtype=torch.long)

    comparison = paired_swing_margin_comparison_rows(
        packed_logits=torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32),
        legal_ids=torch.tensor([1, 2, 1, 2], dtype=torch.long),
        legal_offsets=torch.tensor([0, 2, 4], dtype=torch.long),
        flat_positive_actions=positive_actions.reshape(-1),
        flat_negative_actions=negative_actions.reshape(-1),
        positive_actions=positive_actions,
        negative_actions=negative_actions,
        active_rows=torch.tensor([True, True]),
        pass_action_id=0,
        compare_to="negative",
    )

    assert comparison.supported.tolist() == [True, True]
    assert comparison.margin_by_row.tolist() == pytest.approx([-1.0, -2.0])
    assert comparison.positive_logp_by_row.tolist() == pytest.approx([-1.3132616, -2.126928])
    assert comparison.negative_logp_by_row.tolist() == pytest.approx([-0.3132616, -0.126928])


def test_paired_swing_supported_rows_preserves_weighted_support_contract() -> None:
    supported_rows = paired_swing_supported_rows(
        margin_by_row=torch.tensor([-1.0, 5.0, 2.0], dtype=torch.float64),
        positive_logp_by_row=torch.tensor([-2.0, -99.0, -1.0], dtype=torch.float32),
        negative_logp_by_row=torch.tensor([-1.0, -99.0, -3.0], dtype=torch.float32),
        supported=torch.tensor([True, False, True]),
        flat_loss_mask=torch.tensor([1.0, 10.0, 3.0], dtype=torch.float32),
        raw_weight_total=8.0,
        packed_logits=torch.tensor([0.0], dtype=torch.float32),
        candidate_metrics={"swing_candidate_rows": 3.0},
        metric_prefix="swing",
    )

    assert supported_rows.margins.dtype == torch.float32
    assert supported_rows.margins.tolist() == pytest.approx([-1.0, 2.0])
    assert supported_rows.positive_metric_logp.tolist() == pytest.approx([-2.0, -1.0])
    assert supported_rows.negative_metric_logp.tolist() == pytest.approx([-1.0, -3.0])
    assert supported_rows.supported_weight.tolist() == pytest.approx([1.0, 3.0])
    assert supported_rows.supported_weight_total == pytest.approx(4.0)
    assert supported_rows.metrics["swing_rows"] == 2.0
    assert supported_rows.metrics["swing_supported_fraction"] == pytest.approx(0.5)
    assert supported_rows.metrics["swing_candidate_rows"] == 3.0
    assert supported_rows.has_weight


def test_paired_swing_final_metrics_preserves_weighted_logp_and_retention_metrics() -> None:
    metrics = paired_swing_final_metrics(
        row_metrics={"swing_rows": 2.0, "swing_supported_fraction": 1.0},
        loss=torch.tensor(0.75),
        margin_mean=torch.tensor(-0.25),
        satisfied_fraction=torch.tensor(0.5),
        normalized_scope="label_mean",
        normalized_compare_to="top_other",
        margin_retention_coef=0.25,
        margin_retention_margin=0.5,
        top_action_retention_coef=0.75,
        top_action_retention_margin=1.0,
        positive_metric_logp=torch.tensor([-1.0, -3.0]),
        negative_metric_logp=torch.tensor([0.0, -2.0]),
        supported_weight=torch.tensor([1.0, 3.0]),
        scope_metrics={"paired_swing_label_count": 2.0},
        retention_metrics={"swing_margin_retention_rows": 2.0},
        top_retention_metrics={"swing_top_action_retention_rows": 1.0},
        metric_prefix="swing",
    )

    assert metrics["swing_loss"] == pytest.approx(0.75)
    assert metrics["swing_margin_mean"] == pytest.approx(-0.25)
    assert metrics["swing_satisfied_fraction"] == pytest.approx(0.5)
    assert metrics["swing_loss_scope_episode_mean"] == 0.0
    assert metrics["swing_loss_scope_label_mean"] == 1.0
    assert metrics["swing_compare_to_top_other"] == 1.0
    assert metrics["swing_positive_logp_mean"] == pytest.approx(-2.5)
    assert metrics["swing_negative_logp_mean"] == pytest.approx(-1.5)
    assert metrics["paired_swing_label_count"] == 2.0
    assert metrics["swing_margin_retention_rows"] == 2.0
    assert metrics["swing_top_action_retention_rows"] == 1.0


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


def test_positive_vs_top_other_margin_by_row_preserves_packed_row_support_contract() -> None:
    margin_by_row, supported, positive_logp, top_other_logp = positive_vs_top_other_margin_by_row(
        packed_logits=torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float32),
        legal_ids=torch.tensor([1, 2, 3, 1], dtype=torch.long),
        legal_offsets=torch.tensor([0, 3, 4], dtype=torch.long),
        flat_positive_actions=torch.tensor([1, 1], dtype=torch.long),
        active_rows=torch.tensor([True, True]),
    )

    assert supported.tolist() == [True, False]
    assert margin_by_row[0].item() == pytest.approx(-2.0)
    assert torch.isneginf(margin_by_row[1])
    assert positive_logp[0].item() < top_other_logp[0].item()


def test_paired_swing_scoped_margin_loss_preserves_label_mean_contract() -> None:
    loss, margin_mean, satisfied_fraction, metrics = paired_swing_scoped_margin_loss(
        margins=torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32),
        supported_weight=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        supported=torch.tensor([True, True, True]),
        positive_actions=torch.tensor([[1, 1, 1]], dtype=torch.long),
        group_ids=torch.tensor([1, 1, 2], dtype=torch.long),
        normalized_scope="label_mean",
        margin=0.25,
    )

    assert loss.item() == pytest.approx(0.625)
    assert margin_mean.item() == pytest.approx(0.0)
    assert satisfied_fraction.item() == pytest.approx(0.5)
    assert metrics["paired_swing_label_count"] == 2.0
    assert metrics["paired_swing_label_rows_mean"] == pytest.approx(1.5)


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
