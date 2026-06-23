from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_teacher.auxiliary import compute_structured_teacher_auxiliary_metrics
from weiss_rl.learners.structured_teacher.common import (
    StructuredTeacherAuxiliaryCoefficients,
    StructuredTeacherAuxiliaryLossTerms,
    exact_action_family_rows,
    finalize_structured_teacher_auxiliary_loss,
    flatten_structured_teacher_labels,
)

from .structured_auxiliary_test_support import _catalog


def test_flatten_structured_teacher_labels_normalizes_time_major_labels() -> None:
    labels = flatten_structured_teacher_labels(
        loss_mask=torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
        teacher_family=torch.tensor([[0, 1], [2, -1]], dtype=torch.int16),
        teacher_slot=torch.tensor([[3, 4], [5, -1]], dtype=torch.int16),
        teacher_move_source=torch.tensor([[1, 2], [3, -1]], dtype=torch.int16),
        teacher_attack_type=torch.tensor([[0, 1], [2, -1]], dtype=torch.int16),
        teacher_action=torch.tensor([[10, 11], [12, -1]], dtype=torch.int16),
        teacher_valid=torch.tensor([[1, 1], [0, 1]], dtype=torch.int64),
    )

    assert labels.loss_mask.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert labels.loss_mask.dtype == torch.float32
    assert labels.family.dtype == torch.long
    assert labels.slot.dtype == torch.long
    assert labels.move_source is not None
    assert labels.move_source.tolist() == [1, 2, 3, -1]
    assert labels.attack_type.tolist() == [0, 1, 2, -1]
    assert labels.action is not None
    assert labels.action.tolist() == [10, 11, 12, -1]
    assert labels.valid.dtype == torch.bool
    assert labels.valid.tolist() == [True, True, False, True]


def test_exact_action_family_rows_uses_one_shared_mask_for_teacher_branches() -> None:
    family_names = ("main_play_character", "attack", "main_move", "pass")
    flat_teacher_family = torch.tensor([0, 1, 2, 3, -1], dtype=torch.long)

    expected = torch.tensor([False, True, True, False, False])
    for _branch_name in ("dense", "packed", "factorized"):
        rows = exact_action_family_rows(
            flat_teacher_family=flat_teacher_family,
            family_names=family_names,
            exact_action_families=("attack", "main_move"),
        )
        assert rows is not None
        assert torch.equal(rows, expected)

    assert (
        exact_action_family_rows(
            flat_teacher_family=flat_teacher_family,
            family_names=family_names,
            exact_action_families=(),
        )
        is None
    )


def test_finalize_structured_teacher_auxiliary_loss_uses_all_nonzero_coefficients() -> None:
    terms = StructuredTeacherAuxiliaryLossTerms(
        family=torch.tensor(1.0),
        slot=torch.tensor(2.0),
        hand=torch.tensor(3.0),
        move_source=torch.tensor(4.0),
        attack_type=torch.tensor(5.0),
        action=torch.tensor(6.0),
        same_family_action=torch.tensor(7.0),
        action_margin=torch.tensor(8.0),
        same_family_action_margin=torch.tensor(9.0),
        public_heuristic=torch.tensor(10.0),
        public_nonpass_over_pass=torch.tensor(11.0),
    )
    coefs = StructuredTeacherAuxiliaryCoefficients(
        family=0.1,
        slot=0.2,
        hand=0.3,
        move_source=0.4,
        attack_type=0.5,
        action=0.6,
        same_family_action=0.7,
        action_margin=0.8,
        same_family_action_margin=0.9,
        public_heuristic=1.0,
        public_nonpass_over_pass=1.1,
    )
    metrics: dict[str, float] = {}
    context: dict[str, torch.Tensor] = {}

    total = finalize_structured_teacher_auxiliary_loss(
        terms=terms,
        coefs=coefs,
        metrics=metrics,
        context=context,
        value_dtype=torch.float64,
    )

    expected = sum(
        value * coef
        for value, coef in zip(range(1, 12), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1), strict=True)
    )
    assert total.dtype == torch.float64
    assert total.item() == pytest.approx(expected)
    assert metrics["teacher_aux_loss"] == pytest.approx(expected)
    assert context["teacher_aux_loss"].item() == pytest.approx(expected)


def test_structured_teacher_missing_labels_return_zero_on_logits_dtype_and_device() -> None:
    logits = torch.ones((2, 3), dtype=torch.float64)
    legal_mask = torch.ones((2, 3), dtype=torch.bool)

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=None,
        teacher_slot=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        loss_mask=torch.ones((2,), dtype=torch.float32),
        action_catalog=_catalog(),
        family_coef=1.0,
        slot_coef=1.0,
        attack_type_coef=1.0,
        action_coef=1.0,
        same_family_action_coef=1.0,
    )

    assert aux_loss.item() == pytest.approx(0.0)
    assert aux_loss.dtype == logits.dtype
    assert aux_loss.device == logits.device
    assert metrics["teacher_aux_loss"] == pytest.approx(0.0)
    assert context == {}
