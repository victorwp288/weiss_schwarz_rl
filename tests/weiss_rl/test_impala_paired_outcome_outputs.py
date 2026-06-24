from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.impala.auxiliary.paired_outcome_candidates import PairedOutcomeCandidateLogps
from weiss_rl.learners.impala.auxiliary.paired_outcome_outputs import (
    build_paired_outcome_preference_context,
    build_paired_outcome_preference_metrics,
)


def test_build_paired_outcome_preference_context_preserves_detached_logp_surface() -> None:
    current_candidates = torch.tensor([0.0, 1.0], requires_grad=True)
    reference_candidates = torch.tensor([1.0, 0.0], requires_grad=True)
    candidate_logps = PairedOutcomeCandidateLogps(
        current_candidate_log_probs=current_candidates,
        reference_candidate_log_probs=reference_candidates,
        current_action_logp=torch.tensor([[0.1]], requires_grad=True),
        current_best_non_target_logp=torch.tensor([[0.2]], requires_grad=True),
        reference_action_logp=torch.tensor([[0.3]], requires_grad=True),
        reference_best_non_target_logp=torch.tensor([[0.4]], requires_grad=True),
    )

    context = build_paired_outcome_preference_context(
        weighted_loss=torch.tensor(0.75, requires_grad=True),
        loss_mask=torch.tensor([[1.0]], requires_grad=True),
        candidate_logps=candidate_logps,
        preference_context={"paired_outcome_preference_margins": torch.tensor([0.5])},
    )

    assert context["paired_outcome_preference_loss"].item() == pytest.approx(0.75)
    assert context["policy_train_mask"].tolist() == [[1.0]]
    assert context["current_action_logp"].reshape(-1).tolist() == pytest.approx([0.1])
    assert context["current_best_non_target_logp"].reshape(-1).tolist() == pytest.approx([0.2])
    assert context["reference_action_logp"].reshape(-1).tolist() == pytest.approx([0.3])
    assert context["reference_best_non_target_logp"].reshape(-1).tolist() == pytest.approx([0.4])
    assert context["paired_outcome_preference_margins"].tolist() == pytest.approx([0.5])
    assert not context["paired_outcome_preference_loss"].requires_grad
    assert not context["policy_train_mask"].requires_grad
    assert not context["current_action_logp"].requires_grad


def test_build_paired_outcome_preference_metrics_preserves_flags_and_metric_precedence() -> None:
    metrics = build_paired_outcome_preference_metrics(
        weighted_loss=torch.tensor(0.75),
        coef=0.7,
        beta=0.2,
        aggregation=" Sum ",
        group_balance=True,
        retention_coef=0.1,
        retention_margin=0.2,
        retention_reference_top_only=True,
        top_action_retention_coef=0.3,
        top_action_retention_margin=0.4,
        top_action_retention_reference_top_only=True,
        preference_metrics={
            "paired_outcome_preference_pair_count": 2.0,
            "paired_outcome_preference_weighted_loss": 99.0,
        },
    )

    assert metrics["loss"] == pytest.approx(0.75)
    assert metrics["paired_outcome_preference_weighted_loss"] == 99.0
    assert metrics["paired_outcome_preference_coef"] == pytest.approx(0.7)
    assert metrics["paired_outcome_preference_beta"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_aggregation_sum"] == 1.0
    assert metrics["paired_outcome_preference_group_balance"] == 1.0
    assert metrics["paired_outcome_preference_retention_coef"] == pytest.approx(0.1)
    assert metrics["paired_outcome_preference_retention_margin"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_retention_reference_top_only"] == 1.0
    assert metrics["paired_outcome_preference_top_action_retention_coef"] == pytest.approx(0.3)
    assert metrics["paired_outcome_preference_top_action_retention_margin"] == pytest.approx(0.4)
    assert metrics["paired_outcome_preference_top_action_retention_reference_top_only"] == 1.0
    assert metrics["paired_outcome_preference_pair_count"] == 2.0
