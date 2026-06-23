from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from weiss_rl.learners.structured_auxiliary import (
    packed_structured_legal_view as _packed_structured_legal_view,
)
from weiss_rl.learners.structured_teacher.auxiliary import (
    compute_structured_teacher_auxiliary_metrics,
)
from weiss_rl.learners.structured_teacher.common import empty_structured_teacher_metrics
from weiss_rl.learners.structured_teacher.packed import compute_packed_structured_teacher_auxiliary_metrics

from .impala_test_support import (
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_compute_packed_structured_teacher_auxiliary_metrics_matches_dispatcher() -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, 5] = 3.0
    logits[0, 0, action_catalog.pass_action_id] = -2.0
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    packed_ids_tensor = torch.as_tensor(packed_ids, dtype=torch.long)
    packed_offsets_tensor = torch.as_tensor(packed_offsets, dtype=torch.long)
    packed_meta_tensor = torch.as_tensor(packed_meta, dtype=torch.long)
    packed_view = _packed_structured_legal_view(
        logits=logits[legal_mask],
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=packed_meta_tensor,
    )
    teacher_kwargs = {
        "teacher_family": torch.tensor([[family_index["main_play_character"]]], dtype=torch.long),
        "teacher_slot": torch.tensor([[0]], dtype=torch.long),
        "teacher_attack_type": torch.tensor([[-1]], dtype=torch.long),
        "teacher_action": torch.tensor([[5]], dtype=torch.long),
        "teacher_valid": torch.tensor([[True]], dtype=torch.bool),
        "loss_mask": torch.ones((1, 1), dtype=torch.float32),
        "action_catalog": action_catalog,
        "family_coef": 0.2,
        "slot_coef": 0.1,
        "attack_type_coef": 0.0,
        "action_coef": 0.3,
        "same_family_action_coef": 0.4,
    }

    dispatch_loss, dispatch_metrics, dispatch_context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        packed_ids=packed_ids_tensor,
        packed_offsets=packed_offsets_tensor,
        packed_meta=packed_meta_tensor,
        packed_view=packed_view,
        **cast(Any, teacher_kwargs),
    )
    direct_loss, direct_metrics, direct_context = compute_packed_structured_teacher_auxiliary_metrics(
        packed_view=packed_view,
        packed_offsets=packed_offsets_tensor,
        teacher_move_source=None,
        action_margin_coef=0.0,
        action_margin=0.5,
        same_family_action_margin_coef=0.0,
        same_family_action_margin=0.5,
        exact_action_families=(),
        move_source_coef=0.0,
        public_heuristic_coef=0.0,
        public_heuristic_temperature=32.0,
        public_nonpass_over_pass_coef=0.0,
        public_nonpass_over_pass_margin=0.5,
        public_heuristic_families=(),
        public_heuristic_target_logits=None,
        zero=packed_view.logits.sum() * 0.0,
        value_dtype=packed_view.logits.dtype,
        empty_metrics=empty_structured_teacher_metrics(),
        **cast(Any, teacher_kwargs),
    )

    torch.testing.assert_close(direct_loss, dispatch_loss)
    assert direct_metrics == pytest.approx(dispatch_metrics)
    assert direct_context.keys() == dispatch_context.keys()
    for key in direct_context:
        torch.testing.assert_close(direct_context[key], dispatch_context[key])
