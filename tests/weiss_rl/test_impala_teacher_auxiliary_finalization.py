from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from weiss_rl.learners.impala.losses.loss_finalization import apply_impala_teacher_auxiliary

from .impala_test_support import (
    ImpalaLearner,
    TinyPolicyValueModel,
    _teacher_aux_catalog,
)


def test_apply_impala_teacher_auxiliary_returns_unchanged_loss_when_inactive() -> None:
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    total_loss = torch.tensor(2.0)

    result = apply_impala_teacher_auxiliary(
        learner=object(),
        batch={},
        total_loss=total_loss,
        context=context,
        teacher_aux_active=False,
        logits=None,
        legal_mask=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=None,
        expected_shape=torch.Size((1, 1)),
        packed_legal=None,
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        resolve_legal_mask=lambda _batch, _shape, _action_dim: pytest.fail(
            "inactive teacher aux must not resolve mask"
        ),
        batch_value=lambda batch, key: getattr(batch, key),
    )

    assert result.total_loss is total_loss
    assert result.teacher_metrics == {}
    assert list(context) == ["existing"]
    torch.testing.assert_close(context["existing"], torch.tensor(1.0))


def test_apply_impala_teacher_auxiliary_resolves_dense_mask_for_packed_without_meta() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        teacher_family_coef=0.5,
        teacher_action_coef=0.25,
    )
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    logits = torch.full((1, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, 0, [0, 5, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 3.0
    logits[0, 0, 5] = 0.5
    logits[0, 0, action_catalog.pass_action_id] = -1.0
    packed_ids = torch.as_tensor([0, 5, action_catalog.pass_action_id], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    batch = {
        "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
        "teacher_slot": np.asarray([[0]], dtype=np.int64),
        "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
        "teacher_action": np.asarray([[0]], dtype=np.int64),
        "teacher_valid": np.asarray([[True]], dtype=np.bool_),
    }
    context: dict[str, Any] = {}
    resolver_calls: list[tuple[Any, torch.Size, int]] = []

    result = apply_impala_teacher_auxiliary(
        learner=learner,
        batch=batch,
        total_loss=torch.tensor(1.0),
        context=context,
        teacher_aux_active=True,
        logits=logits,
        legal_mask=None,
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        action_catalog=action_catalog,
        expected_shape=torch.Size((1, 1)),
        packed_legal=(packed_ids, packed_offsets, None),
        packed_view=None,
        factorized_result=None,
        public_heuristic_target_logits=None,
        resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
            resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
    )

    assert resolver_calls == [(batch, torch.Size((1, 1)), action_catalog.action_space_size)]
    assert result.total_loss.item() > 1.0
    assert result.teacher_metrics["teacher_valid_fraction"] == pytest.approx(1.0)
    assert result.teacher_metrics["teacher_family_accuracy"] == pytest.approx(1.0)
    assert result.teacher_metrics["teacher_action_accuracy"] == pytest.approx(1.0)
    assert "teacher_aux_loss" in context
