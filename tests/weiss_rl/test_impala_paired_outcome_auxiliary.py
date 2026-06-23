from __future__ import annotations

import pytest
import torch

from tests.weiss_rl.impala_paired_auxiliary_test_support import make_factorized_paired_outcome_loss_case


def test_impala_learner_paired_outcome_auxiliary_preserves_factorized_metrics() -> None:
    model, learner, batch = make_factorized_paired_outcome_loss_case()

    loss, metrics, context = learner._paired_outcome_preference_loss_and_metrics(
        batch,
        beta=0.2,
        coef=0.7,
        aggregation="sum",
        group_balance=True,
    )

    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(metrics["paired_outcome_preference_weighted_loss"])
    assert metrics["paired_outcome_preference_coef"] == pytest.approx(0.7)
    assert metrics["paired_outcome_preference_beta"] == pytest.approx(0.2)
    assert metrics["paired_outcome_preference_aggregation_sum"] == 1.0
    assert metrics["paired_outcome_preference_group_balance"] == 1.0
    assert metrics["paired_outcome_preference_pair_count"] == 1.0
    assert metrics["paired_outcome_preference_group_count"] == 1.0
    assert context["current_action_logp"].shape == (2, 1)
    assert context["reference_action_logp"].shape == (2, 1)
    assert context["current_best_non_target_logp"].shape == (2, 1)
    assert context["reference_best_non_target_logp"].shape == (2, 1)
    assert context["paired_outcome_preference_margins"].tolist() == pytest.approx([0.0])
    assert model.factorized_candidate_logp_calls == 1
