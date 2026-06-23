from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import weiss_rl.learners.impala.loss_teacher_targets_stage as impala_loss_teacher_targets_stage
from weiss_rl.learners.impala.loss_teacher_targets_stage import prepare_impala_loss_teacher_target_inputs


def test_prepare_impala_loss_teacher_target_inputs_maps_forward_state_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = object()
    batch = {"teacher_target_batch": True}
    forward_model = object()
    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    packed_logits = torch.arange(4, dtype=torch.float32)
    packed_legal = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    factorized_result = object()
    forward_observation_context = {"encoded": torch.ones((2, 1), dtype=torch.float32)}
    masks = SimpleNamespace(loss_mask=loss_mask)
    forward_flags = SimpleNamespace(emit_structured_metrics=False, teacher_aux_active=True)
    forward_result = SimpleNamespace(
        logits=logits,
        packed_logits=packed_logits,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        forward_observation_context=forward_observation_context,
    )
    packed_view = object()
    teacher_view = object()
    public_targets = torch.ones((4,), dtype=torch.float32)
    captured: dict[str, Any] = {}

    def fake_prepare_impala_teacher_target_inputs(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            packed_view=packed_view,
            teacher_aux_packed_view=teacher_view,
            public_heuristic_target_logits=public_targets,
        )

    monkeypatch.setattr(
        impala_loss_teacher_targets_stage,
        "prepare_impala_teacher_target_inputs",
        fake_prepare_impala_teacher_target_inputs,
    )

    result = prepare_impala_loss_teacher_target_inputs(
        learner=learner,
        batch=batch,
        forward_model=forward_model,
        obs=obs,
        masks=masks,
        forward_flags=forward_flags,
        forward_result=forward_result,
    )

    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["forward_model"] is forward_model
    assert captured["obs"] is obs
    assert captured["logits"] is logits
    assert captured["packed_logits"] is packed_logits
    assert captured["packed_legal"] is packed_legal
    assert captured["loss_mask"] is loss_mask
    assert captured["factorized_result"] is factorized_result
    assert captured["forward_observation_context"] is forward_observation_context
    assert captured["need_packed_view"] is True
    assert captured["teacher_aux_enabled"] is True
    assert result.packed_view is packed_view
    assert result.teacher_aux_packed_view is teacher_view
    assert result.public_heuristic_target_logits is public_targets


def test_prepare_impala_loss_teacher_target_inputs_needs_packed_view_for_structured_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_prepare_impala_teacher_target_inputs(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            packed_view=None,
            teacher_aux_packed_view=None,
            public_heuristic_target_logits=None,
        )

    monkeypatch.setattr(
        impala_loss_teacher_targets_stage,
        "prepare_impala_teacher_target_inputs",
        fake_prepare_impala_teacher_target_inputs,
    )

    prepare_impala_loss_teacher_target_inputs(
        learner=object(),
        batch={},
        forward_model=object(),
        obs=torch.zeros((1, 1, 2), dtype=torch.float32),
        masks=SimpleNamespace(loss_mask=torch.ones((1, 1), dtype=torch.float32)),
        forward_flags=SimpleNamespace(emit_structured_metrics=True, teacher_aux_active=False),
        forward_result=SimpleNamespace(
            logits=None,
            packed_logits=None,
            packed_legal=None,
            factorized_result=None,
            forward_observation_context=None,
        ),
    )

    assert captured["need_packed_view"] is True
    assert captured["teacher_aux_enabled"] is False
