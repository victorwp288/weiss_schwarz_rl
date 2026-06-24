from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from weiss_rl.learners.impala.losses.loss_inputs import (
    prepare_impala_loss_inputs,
    resolve_impala_loss_forward_flags,
)

from .impala_test_support import (
    ImpalaLearner,
    TinyPolicyValueModel,
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


def test_prepare_impala_loss_inputs_restricts_packed_forward_to_policy_and_retention_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="off",
        trajectory_retention_coef=0.4,
    )
    packed_ids = np.asarray(
        [0, 5, action_catalog.pass_action_id, 0, 5, action_catalog.pass_action_id],
        dtype=np.uint32,
    )
    packed_offsets = np.asarray([0, 3, 6], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [5]], dtype=np.int64),
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "trajectory_retention_valid": np.asarray([[False], [True]], dtype=np.bool_),
    }
    captured_policy_masks: list[torch.Tensor | None] = []

    def fake_forward(
        obs: torch.Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
        legal_actions: Any = None,
        policy_train_mask: torch.Tensor | None = None,
        reset_before_step: torch.Tensor | None = None,
        opponent_context_index: Any = None,
    ) -> SimpleNamespace:
        del initial_hidden_state, to_play_seat, actor, legal_actions, reset_before_step, opponent_context_index
        captured_policy_masks.append(None if policy_train_mask is None else policy_train_mask.detach().clone())
        return SimpleNamespace(
            logits=None,
            packed_logits=torch.zeros((int(packed_ids.shape[0]),), dtype=torch.float32),
            values=torch.zeros(obs.shape[:2], dtype=torch.float32),
            observation_context={"rows": obs.reshape(-1, obs.shape[-1])},
        )

    cast(Any, learner)._forward_time_major = fake_forward

    prepared = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))

    assert prepared.packed_legal is not None
    assert prepared.legal_mask is None
    assert prepared.teacher_aux_active is False
    assert prepared.emit_structured_metrics is False
    assert captured_policy_masks
    assert captured_policy_masks[0] is not None
    assert captured_policy_masks[0].tolist() == [[1.0], [1.0]]
    assert prepared.context["packed_logits"].shape == (int(packed_ids.shape[0]),)
    assert prepared.context["values"].tolist() == [[0.0], [0.0]]


def test_resolve_impala_loss_forward_flags_only_restricts_safe_packed_policy_rows() -> None:
    action_catalog = _teacher_aux_catalog()
    packed_legal = (
        torch.as_tensor([0, action_catalog.pass_action_id], dtype=torch.long),
        torch.as_tensor([0, 2], dtype=torch.long),
        torch.as_tensor(_packed_meta_from_ids(action_catalog, np.asarray([0, action_catalog.pass_action_id]))),
    )
    loss_mask = torch.as_tensor([[1.0], [0.0]], dtype=torch.float32)

    plain = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="off",
    )
    teacher_model = TinyPolicyValueModel(action_dim=action_catalog.action_space_size)
    teacher_model.action_catalog = action_catalog
    teacher = ImpalaLearner(model=teacher_model, teacher_action_coef=0.5, structured_metrics_mode="off")
    structured = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=action_catalog.action_space_size),
        structured_metrics_mode="full",
    )

    plain_flags = resolve_impala_loss_forward_flags(learner=plain, packed_legal=packed_legal, loss_mask=loss_mask)
    teacher_flags = resolve_impala_loss_forward_flags(learner=teacher, packed_legal=packed_legal, loss_mask=loss_mask)
    structured_flags = resolve_impala_loss_forward_flags(
        learner=structured,
        packed_legal=packed_legal,
        loss_mask=loss_mask,
    )
    dense_flags = resolve_impala_loss_forward_flags(learner=plain, packed_legal=None, loss_mask=loss_mask)

    assert plain_flags.teacher_aux_active is False
    assert plain_flags.emit_structured_metrics is False
    assert plain_flags.restrict_packed_policy_rows is True
    assert teacher_flags.teacher_aux_active is True
    assert teacher_flags.restrict_packed_policy_rows is False
    assert structured_flags.emit_structured_metrics is True
    assert structured_flags.restrict_packed_policy_rows is False
    assert dense_flags.restrict_packed_policy_rows is False
