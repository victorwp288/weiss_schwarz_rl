from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    packed_structured_legal_view as _packed_structured_legal_view,
)
from weiss_rl.learners.structured_teacher.dispatch import (
    resolve_structured_teacher_branch,
    resolve_structured_teacher_required_labels,
    resolve_structured_teacher_zero_context,
)

from .impala_test_support import _packed_meta_from_ids, _teacher_aux_catalog


def test_resolve_structured_teacher_zero_context_uses_packed_view_before_loss_mask() -> None:
    action_catalog = _teacher_aux_catalog()
    packed_ids = torch.as_tensor([0, 5], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 2], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor([1.0, 2.0], dtype=torch.float64),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    packed_zero = resolve_structured_teacher_zero_context(
        logits=None,
        packed_view=packed_view,
        loss_mask=torch.ones((1, 1), dtype=torch.float64),
    )
    mask_zero = resolve_structured_teacher_zero_context(
        logits=None,
        packed_view=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float64),
    )

    assert packed_zero.value_dtype == packed_view.logits.dtype
    assert packed_zero.zero.dtype == packed_view.logits.dtype
    assert packed_zero.empty_metrics["teacher_aux_loss"] == pytest.approx(0.0)
    assert mask_zero.value_dtype == torch.float64


def test_resolve_structured_teacher_required_labels_names_missing_label_gate() -> None:
    family = torch.tensor([[0]], dtype=torch.long)
    slot = torch.tensor([[1]], dtype=torch.long)
    attack_type = torch.tensor([[-1]], dtype=torch.long)
    valid = torch.tensor([[True]], dtype=torch.bool)

    labels = resolve_structured_teacher_required_labels(
        teacher_family=family,
        teacher_slot=slot,
        teacher_attack_type=attack_type,
        teacher_valid=valid,
    )
    missing = resolve_structured_teacher_required_labels(
        teacher_family=family,
        teacher_slot=None,
        teacher_attack_type=attack_type,
        teacher_valid=valid,
    )

    assert labels is not None
    assert labels.family is family
    assert labels.slot is slot
    assert labels.attack_type is attack_type
    assert labels.valid is valid
    assert missing is None


def test_resolve_structured_teacher_branch_prioritizes_factorized_then_packed_then_dense() -> None:
    factorized = resolve_structured_teacher_branch(
        factorized_family_log_probs=torch.zeros((1, 1, 2)),
        packed_view=object(),
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    packed = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=object(),
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    dense = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=None,
        logits=torch.zeros((1, 1, 3)),
        legal_mask=torch.ones((1, 1, 3), dtype=torch.bool),
    )
    inactive = resolve_structured_teacher_branch(
        factorized_family_log_probs=None,
        packed_view=None,
        logits=torch.zeros((1, 1, 3)),
        legal_mask=None,
    )

    assert factorized.use_factorized is True
    assert factorized.use_packed is False
    assert factorized.use_dense is False
    assert packed.use_factorized is False
    assert packed.use_packed is True
    assert packed.use_dense is False
    assert dense.use_factorized is False
    assert dense.use_packed is False
    assert dense.use_dense is True
    assert inactive.use_factorized is False
    assert inactive.use_packed is False
    assert inactive.use_dense is False
