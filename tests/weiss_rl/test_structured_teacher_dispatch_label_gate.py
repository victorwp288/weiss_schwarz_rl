from __future__ import annotations

import torch
from weiss_rl.learners.structured_teacher.dispatch import resolve_structured_teacher_dispatch

from .impala_test_support import _packed_meta_from_ids, _teacher_aux_catalog


def test_resolve_structured_teacher_dispatch_preserves_label_gate_before_packed_view_build() -> None:
    action_catalog = _teacher_aux_catalog()
    logits = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float64)
    legal_mask = torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    packed_ids = torch.as_tensor([0, 5], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 2], dtype=torch.long)
    invalid_packed_meta = torch.zeros((2, 3), dtype=torch.long)
    labels = {
        "teacher_family": torch.tensor([[0]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
    }

    missing_label_dispatch = resolve_structured_teacher_dispatch(
        logits=logits,
        legal_mask=legal_mask,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=invalid_packed_meta,
        packed_view=None,
        factorized_family_log_probs=None,
        teacher_family=labels["teacher_family"],
        teacher_slot=None,
        teacher_attack_type=labels["teacher_attack_type"],
        teacher_valid=labels["teacher_valid"],
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
    )

    assert missing_label_dispatch.labels is None
    assert missing_label_dispatch.packed_view is None
    assert missing_label_dispatch.branch.use_factorized is False
    assert missing_label_dispatch.branch.use_packed is False
    assert missing_label_dispatch.branch.use_dense is False
    assert missing_label_dispatch.zero_context.value_dtype == torch.float64

    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    factorized_dispatch = resolve_structured_teacher_dispatch(
        logits=logits,
        legal_mask=legal_mask,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
        packed_view=None,
        factorized_family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        **labels,
    )

    assert factorized_dispatch.labels is not None
    assert factorized_dispatch.packed_view is not None
    assert factorized_dispatch.branch.use_factorized is True
    assert factorized_dispatch.branch.use_packed is False
    assert factorized_dispatch.branch.use_dense is False
    assert factorized_dispatch.zero_context.value_dtype == torch.float64
