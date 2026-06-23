from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import (
    packed_structured_legal_view as _packed_structured_legal_view,
)

from .impala_test_support import _packed_meta_from_ids, _teacher_aux_catalog


@dataclass(frozen=True)
class PackedTeacherPublicMarginCase:
    action_catalog: ActionCatalog
    family_index: dict[str, int]
    packed_offsets: torch.Tensor
    packed_view: Any
    teacher_family: torch.Tensor
    teacher_action: torch.Tensor
    teacher_valid: torch.Tensor
    loss_mask: torch.Tensor
    public_target_logits: torch.Tensor
    zero: torch.Tensor


def _packed_teacher_public_margin_case(
    *,
    logits: tuple[float, float, float],
    teacher_action_id: int,
) -> PackedTeacherPublicMarginCase:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(action_catalog, packed_ids.numpy()), dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=torch.tensor(logits, dtype=torch.float32),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    teacher_family = torch.tensor([[family_index["main_play_character"]]], dtype=torch.long)
    teacher_action = torch.tensor([[teacher_action_id]], dtype=torch.long)
    teacher_valid = torch.tensor([[True]], dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)
    public_target_logits = torch.tensor([4.0, 5.0, -5.0], dtype=torch.float32)

    return PackedTeacherPublicMarginCase(
        action_catalog=action_catalog,
        family_index=family_index,
        packed_offsets=packed_offsets,
        packed_view=packed_view,
        teacher_family=teacher_family,
        teacher_action=teacher_action,
        teacher_valid=teacher_valid,
        loss_mask=loss_mask,
        public_target_logits=public_target_logits,
        zero=packed_view.logits.sum() * 0.0,
    )
