from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import weiss_rl.learners.impala.auxiliary.teacher_auxiliary_call as impala_teacher_auxiliary_call
from weiss_rl.learners.impala.auxiliary.teacher_auxiliary_request import resolve_impala_teacher_auxiliary_inputs

from .impala_test_support import (
    ImpalaLearner,
    TinyStructuredTeacherModel,
    _teacher_aux_catalog,
)


def test_compute_structured_teacher_auxiliary_from_impala_inputs_maps_all_fields(monkeypatch) -> None:
    action_catalog = _teacher_aux_catalog()
    family_index = {family.name: index for index, family in enumerate(action_catalog.families)}
    learner = ImpalaLearner(
        model=TinyStructuredTeacherModel(action_catalog),
        teacher_family_coef=0.11,
        teacher_slot_coef=0.12,
        teacher_hand_coef=0.13,
        teacher_move_source_coef=0.14,
        teacher_attack_type_coef=0.15,
        teacher_action_coef=0.16,
        teacher_same_family_action_coef=0.17,
        teacher_action_margin_coef=0.18,
        teacher_action_margin=0.19,
        teacher_same_family_action_margin_coef=0.20,
        teacher_same_family_action_margin=0.21,
        teacher_exact_action_families=("attack",),
        teacher_public_heuristic_coef=0.22,
        teacher_public_heuristic_temperature=0.23,
        teacher_public_nonpass_over_pass_coef=0.24,
        teacher_public_nonpass_over_pass_margin=0.25,
        teacher_public_heuristic_families=("main_play_character",),
    )
    ids = torch.tensor([0, 5], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    meta = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
    packed_view = object()
    factorized_result = SimpleNamespace(
        family_log_probs=torch.zeros((1, 1, len(action_catalog.families))),
        play_slot_log_probs=torch.ones((1, 1, int(action_catalog.max_stage))),
        move_source_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 1.5),
        move_slot_log_probs=torch.full((1, 1, int(action_catalog.max_stage)), 2.0),
        attack_slot_log_probs=torch.full((1, 1, int(action_catalog.attack_slot_count)), 3.0),
        attack_type_log_probs=torch.full((1, 1, len(action_catalog.attack_type_names)), 4.0),
        top_action_ids=torch.tensor([[0]], dtype=torch.long),
        same_family_action_logp=torch.tensor([[-0.5]]),
        same_family_top_action_ids=torch.tensor([[5]], dtype=torch.long),
        same_family_arg0_logp=torch.tensor([[-0.6]]),
        same_family_top_arg0=torch.tensor([[1]], dtype=torch.long),
    )
    inputs = resolve_impala_teacher_auxiliary_inputs(
        learner=learner,
        batch={
            "teacher_family": np.asarray([[family_index["main_play_character"]]], dtype=np.int64),
            "teacher_slot": np.asarray([[0]], dtype=np.int64),
            "teacher_move_source": np.asarray([[1]], dtype=np.int64),
            "teacher_attack_type": np.asarray([[-1]], dtype=np.int64),
            "teacher_action": np.asarray([[0]], dtype=np.int64),
            "teacher_valid": np.asarray([[True]], dtype=np.bool_),
        },
        batch_value=lambda batch, key: batch.get(key),
        expected_shape=torch.Size((1, 1)),
        packed_legal=(ids, offsets, meta),
        packed_view=packed_view,
        factorized_result=factorized_result,
    )
    logits = torch.zeros((1, 1, action_catalog.action_space_size), dtype=torch.float32)
    legal_mask = torch.ones((1, 1, action_catalog.action_space_size), dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)
    public_target = torch.arange(2, dtype=torch.float32)
    sentinel_loss = torch.tensor(9.0)
    sentinel_metrics = {"teacher_aux_loss": 9.0}
    sentinel_context = {"teacher_family_log_probs": torch.tensor([1.0])}
    captured: dict[str, Any] = {}

    def fake_compute_structured_teacher_auxiliary_metrics(
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        captured.update(kwargs)
        return sentinel_loss, sentinel_metrics, sentinel_context

    monkeypatch.setattr(
        impala_teacher_auxiliary_call,
        "compute_structured_teacher_auxiliary_metrics",
        fake_compute_structured_teacher_auxiliary_metrics,
    )

    loss, metrics, context = impala_teacher_auxiliary_call.compute_structured_teacher_auxiliary_from_impala_inputs(
        inputs=inputs,
        logits=logits,
        legal_mask=legal_mask,
        loss_mask=loss_mask,
        action_catalog=action_catalog,
        public_heuristic_target_logits=public_target,
    )

    assert loss is sentinel_loss
    assert metrics is sentinel_metrics
    assert context is sentinel_context
    assert captured["logits"] is logits
    assert captured["legal_mask"] is legal_mask
    assert captured["teacher_family"] is inputs.labels.family
    assert captured["teacher_slot"] is inputs.labels.slot
    assert captured["teacher_move_source"] is inputs.labels.move_source
    assert captured["teacher_attack_type"] is inputs.labels.attack_type
    assert captured["teacher_action"] is inputs.labels.action
    assert captured["teacher_valid"] is inputs.labels.valid
    assert captured["loss_mask"] is loss_mask
    assert captured["action_catalog"] is action_catalog
    assert captured["family_coef"] == pytest.approx(0.11)
    assert captured["slot_coef"] == pytest.approx(0.12)
    assert captured["hand_coef"] == pytest.approx(0.13)
    assert captured["move_source_coef"] == pytest.approx(0.14)
    assert captured["attack_type_coef"] == pytest.approx(0.15)
    assert captured["action_coef"] == pytest.approx(0.16)
    assert captured["same_family_action_coef"] == pytest.approx(0.17)
    assert captured["action_margin_coef"] == pytest.approx(0.18)
    assert captured["action_margin"] == pytest.approx(0.19)
    assert captured["same_family_action_margin_coef"] == pytest.approx(0.20)
    assert captured["same_family_action_margin"] == pytest.approx(0.21)
    assert captured["exact_action_families"] == ("attack",)
    assert captured["public_heuristic_coef"] == pytest.approx(0.22)
    assert captured["public_heuristic_temperature"] == pytest.approx(0.23)
    assert captured["public_nonpass_over_pass_coef"] == pytest.approx(0.24)
    assert captured["public_nonpass_over_pass_margin"] == pytest.approx(0.25)
    assert captured["public_heuristic_families"] == ("main_play_character",)
    assert captured["public_heuristic_target_logits"] is public_target
    assert captured["packed_ids"] is ids
    assert captured["packed_offsets"] is offsets
    assert captured["packed_meta"] is meta
    assert captured["packed_view"] is packed_view
    assert captured["factorized_family_log_probs"] is factorized_result.family_log_probs
    assert captured["factorized_play_slot_log_probs"] is factorized_result.play_slot_log_probs
    assert captured["factorized_move_source_log_probs"] is factorized_result.move_source_log_probs
    assert captured["factorized_move_slot_log_probs"] is factorized_result.move_slot_log_probs
    assert captured["factorized_attack_slot_log_probs"] is factorized_result.attack_slot_log_probs
    assert captured["factorized_attack_type_log_probs"] is factorized_result.attack_type_log_probs
    assert captured["factorized_top_action_ids"] is factorized_result.top_action_ids
    assert captured["factorized_same_family_action_logp"] is factorized_result.same_family_action_logp
    assert captured["factorized_same_family_top_action_ids"] is factorized_result.same_family_top_action_ids
    assert captured["factorized_same_family_arg0_logp"] is factorized_result.same_family_arg0_logp
    assert captured["factorized_same_family_top_arg0"] is factorized_result.same_family_top_arg0
