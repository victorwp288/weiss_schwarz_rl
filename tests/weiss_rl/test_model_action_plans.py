from __future__ import annotations

import torch
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.model import (
    _FactorizedConditionalLogProbs,
    _FactorizedEvaluationResult,
    _FactorizedFamilyPlan,
    _FactorizedLegalityPlan,
    _PackedScoringPlan,
)
from weiss_rl.models.actions.action_plans import (
    FactorizedConditionalLogProbs,
    FactorizedEvaluationResult,
    FactorizedFamilyPlan,
    FactorizedLegalityPlan,
    PackedScoringPlan,
    build_factorized_legality_plan,
)


def test_packed_scoring_plan_slice_preserves_fields_and_candidate_count() -> None:
    plan = PackedScoringPlan(
        row_indices=torch.tensor([0, 0, 1]),
        family_ids=torch.tensor([1, 2, 3]),
        arg0=torch.tensor([10, 20, 30]),
        arg1=torch.tensor([100, 200, 300]),
    )

    sliced = plan.slice(1, 3)

    assert plan.candidate_count == 3
    assert sliced.candidate_count == 2
    assert sliced.row_indices.tolist() == [0, 1]
    assert sliced.family_ids.tolist() == [2, 3]
    assert sliced.arg0.tolist() == [20, 30]
    assert sliced.arg1.tolist() == [200, 300]


def test_factorized_plan_containers_preserve_payloads() -> None:
    family_plan = FactorizedFamilyPlan(
        row_indices=torch.tensor([1, 3]),
        arg0_mask=torch.tensor([[True, False], [False, True]]),
        arg1_mask=None,
    )
    legality = FactorizedLegalityPlan(
        row_count=4,
        family_mask=torch.tensor([[True], [False], [True], [False]]),
        family_candidate_counts=torch.tensor([[1], [0], [1], [0]]),
        family_plans={7: family_plan},
    )
    conditional = FactorizedConditionalLogProbs(
        row_indices=torch.tensor([1, 3]),
        log_probs=torch.tensor([[0.0, -1.0], [-2.0, -3.0]]),
        mask=torch.tensor([[True, True], [True, False]]),
    )
    result = FactorizedEvaluationResult(
        values=torch.tensor([0.5]),
        action_logp=None,
        entropy=None,
        family_log_probs=torch.tensor([[0.0]]),
        play_slot_log_probs=None,
        move_source_log_probs=None,
        move_slot_log_probs=None,
        attack_slot_log_probs=None,
        attack_type_log_probs=None,
    )

    assert legality.family_plans[7] is family_plan
    assert legality.family_candidate_counts.tolist() == [[1], [0], [1], [0]]
    assert conditional.log_probs.shape == (2, 2)
    assert result.values.tolist() == [0.5]


def test_model_private_action_plan_aliases_are_preserved() -> None:
    assert _PackedScoringPlan is PackedScoringPlan
    assert _FactorizedEvaluationResult is FactorizedEvaluationResult
    assert _FactorizedFamilyPlan is FactorizedFamilyPlan
    assert _FactorizedConditionalLogProbs is FactorizedConditionalLogProbs
    assert _FactorizedLegalityPlan is FactorizedLegalityPlan


def test_build_factorized_legality_plan_preserves_packed_row_order() -> None:
    # Action ids are deliberately not sorted by family within each packed row.
    legal_actions = LegalActionBatch.from_packed(
        ids=[2, 0, 1, 3, 4],
        offsets=[0, 3, 5],
        action_space=5,
    )
    family_ids_by_action = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    action_arg0 = torch.tensor([1, 0, 0, 1, 2], dtype=torch.long)
    action_arg1 = torch.tensor([0, 1, 2, 0, -1], dtype=torch.long)
    family_arg0_size = torch.tensor([3, 2], dtype=torch.long)
    family_arg1_size = torch.tensor([3, 2], dtype=torch.long)

    plan = build_factorized_legality_plan(
        legal_actions,
        device=torch.device("cpu"),
        family_ids_by_action=family_ids_by_action,
        action_arg0=action_arg0,
        action_arg1=action_arg1,
        family_arg0_size=family_arg0_size,
        family_arg1_size=family_arg1_size,
        family_count=2,
    )

    assert plan.row_count == 2
    assert plan.family_mask.tolist() == [[True, True], [True, True]]

    family0 = plan.family_plans[0]
    assert family0.row_indices.tolist() == [0, 1]
    assert family0.arg0_mask is not None
    assert family0.arg0_mask.tolist() == [
        [True, True, False],
        [False, False, True],
    ]
    assert family0.arg1_mask is not None
    assert family0.arg1_mask.tolist() == [
        [[False, False, True], [True, False, False], [False, False, False]],
        [[False, False, False], [False, False, False], [False, False, False]],
    ]

    family1 = plan.family_plans[1]
    assert family1.row_indices.tolist() == [0, 1]
    assert family1.arg0_mask is not None
    assert family1.arg0_mask.tolist() == [
        [True, False],
        [False, True],
    ]
    assert family1.arg1_mask is not None
    assert family1.arg1_mask.tolist() == [
        [[False, True], [False, False]],
        [[False, False], [True, False]],
    ]
