from __future__ import annotations

import torch
from weiss_rl.learners.paired_outcome_preference.inputs import (
    prepare_paired_outcome_preference_loss_inputs,
)
from weiss_rl.learners.paired_outcome_preference.metrics import empty_paired_outcome_preference_metrics


def test_empty_paired_outcome_preference_metrics_preserve_aggregation_and_retention_defaults() -> None:
    metrics = empty_paired_outcome_preference_metrics(metric_prefix="pref", aggregation="edge_mean")

    assert metrics["pref_loss"] == 0.0
    assert metrics["pref_pair_count"] == 0.0
    assert metrics["pref_aggregation_sum"] == 0.0
    assert metrics["pref_aggregation_edge_mean"] == 1.0
    assert metrics["pref_retention_loss"] == 0.0
    assert metrics["pref_top_action_retention_loss"] == 0.0


def test_prepare_paired_outcome_preference_inputs_normalizes_options_and_ignores_invalid_masked_weights() -> None:
    current = torch.tensor([[-2.0, -2.0, -5.0]], dtype=torch.float32)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[9, 9, 10]])
    roles = torch.tensor([[1, 0, 1]])
    mask = torch.tensor([[True, True, False]])
    weights = torch.tensor([[3.0, 3.0, float("nan")]])

    prepared = prepare_paired_outcome_preference_loss_inputs(
        current_action_logp=current,
        reference_action_logp=reference,
        current_best_non_target_logp=None,
        reference_best_non_target_logp=None,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        preference_group_ids=None,
        preference_pair_weights=weights,
        loss_mask=mask,
        aggregation=" Edge_Mean ",
        group_balance=False,
        retention_coef=0.0,
        retention_margin=0.0,
        retention_role=" Preferred ",
        retention_scope_mask=None,
        top_action_retention_coef=0.0,
        top_action_retention_margin=0.0,
        top_action_retention_role=" ALL ",
        top_action_retention_scope_mask=None,
    )

    assert prepared.options.aggregation == "edge_mean"
    assert prepared.options.retention_role == "preferred"
    assert prepared.options.top_action_retention_role == "all"
    assert prepared.valid_row_count == 2
    assert prepared.unique_pair_ids.tolist() == [9]
