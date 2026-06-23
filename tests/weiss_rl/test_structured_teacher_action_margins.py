from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    packed_structured_legal_view,
)
from weiss_rl.learners.structured_teacher.auxiliary import compute_structured_teacher_auxiliary_metrics

from .structured_auxiliary_test_support import _catalog, action_margin_packed_inputs


def test_packed_teacher_action_margin_loss_rewards_explicit_logit_separation() -> None:
    catalog = _catalog()
    logits, packed_ids, packed_offsets, packed_meta = action_margin_packed_inputs()

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        teacher_family=torch.tensor([1, 0], dtype=torch.long),
        teacher_slot=torch.tensor([0, 0], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 0], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 3.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=2.0,
        action_margin=0.5,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.1))
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.05)
    assert metrics["teacher_action_margin_mean"] == pytest.approx(2.325)
    assert metrics["teacher_action_margin_satisfied_fraction"] == pytest.approx(0.75)
    torch.testing.assert_close(context["teacher_action_margins"], torch.tensor([0.3, 3.0]))


def test_factorized_teacher_action_margin_uses_packed_student_logits() -> None:
    catalog = _catalog()
    logits, packed_ids, packed_offsets, packed_meta = action_margin_packed_inputs()
    packed_view = packed_structured_legal_view(
        logits=logits,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([1, 0], dtype=torch.long),
        teacher_slot=torch.tensor([0, 0], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 0], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 3.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=2.0,
        action_margin=0.5,
        packed_view=packed_view,
        factorized_family_log_probs=torch.zeros((2, 4), dtype=torch.float32),
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.1))
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.05)
    torch.testing.assert_close(context["teacher_action_margins"], torch.tensor([0.3, 3.0]))


def test_exact_action_family_filter_skips_public_unstable_candidate_ids() -> None:
    catalog = _catalog()
    logits, packed_ids, packed_offsets, packed_meta = action_margin_packed_inputs()

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        teacher_family=torch.tensor([1, 0], dtype=torch.long),
        teacher_slot=torch.tensor([0, 0], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 0], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 100.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=0.0,
        exact_action_families=("attack",),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    expected = -(logits[1] - torch.logsumexp(logits[:3], dim=0))
    torch.testing.assert_close(aux_loss, expected)
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)


def test_public_nonpass_over_pass_margin_loss_targets_pass_collapse_only_when_public_teacher_prefers_nonpass() -> None:
    catalog = _catalog()
    logits, packed_ids, packed_offsets, packed_meta = action_margin_packed_inputs()
    target_logits = torch.tensor([0.0, 3.0, 1.0, 0.0, 5.0], dtype=torch.float32)

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        teacher_family=torch.tensor([1, 0], dtype=torch.long),
        teacher_slot=torch.tensor([0, 0], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 0], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 3.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        public_heuristic_target_logits=target_logits,
        public_nonpass_over_pass_coef=2.0,
        public_nonpass_over_pass_margin=0.5,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.4))
    assert metrics["teacher_public_nonpass_over_pass_supported_fraction"] == pytest.approx(0.25)
    assert metrics["teacher_public_nonpass_over_pass_loss"] == pytest.approx(0.2)
    assert metrics["teacher_public_nonpass_over_pass_margin_mean"] == pytest.approx(0.3)
    assert metrics["teacher_public_nonpass_over_pass_satisfied_fraction"] == pytest.approx(0.0)
    torch.testing.assert_close(context["teacher_public_nonpass_over_pass_margins"], torch.tensor([0.3]))


def test_dense_teacher_action_margin_loss_matches_packed_branch() -> None:
    catalog = _catalog()
    logits = torch.full((1, 2, 20), -7.0, dtype=torch.float32)
    logits[0, 0, 0] = 1.0
    logits[0, 0, 10] = 2.0
    logits[0, 0, 19] = 1.7
    logits[0, 1, 0] = 4.0
    logits[0, 1, 19] = 1.0
    legal_mask = torch.zeros((1, 2, 20), dtype=torch.bool)
    legal_mask[0, 0, [0, 10, 19]] = True
    legal_mask[0, 1, [0, 19]] = True

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor([[1, 0]], dtype=torch.long),
        teacher_slot=torch.tensor([[0, 0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[0, -1]], dtype=torch.long),
        teacher_action=torch.tensor([[10, 0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True, True]]),
        loss_mask=torch.tensor([[1.0, 3.0]], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=2.0,
        action_margin=0.5,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.1))
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.05)
    assert metrics["teacher_action_margin_mean"] == pytest.approx(2.325)
    assert metrics["teacher_action_margin_satisfied_fraction"] == pytest.approx(0.75)
    torch.testing.assert_close(context["teacher_action_margins"], torch.tensor([0.3, 3.0]))
