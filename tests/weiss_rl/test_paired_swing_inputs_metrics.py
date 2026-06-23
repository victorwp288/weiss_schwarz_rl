from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_swing.inputs import prepare_paired_swing_loss_inputs
from weiss_rl.learners.paired_swing.metrics import (
    paired_swing_final_metrics,
    paired_swing_supported_rows,
)


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
