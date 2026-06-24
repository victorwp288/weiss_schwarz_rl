from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from weiss_rl.learners.impala.auxiliary.teacher_target_inputs import (
    prepare_impala_teacher_target_inputs,
    resolve_impala_teacher_target_plan,
)

from .impala_test_support import (
    _packed_meta_from_ids,
    _teacher_aux_catalog,
)


class _TeacherTargetInputLearner:
    def __init__(self) -> None:
        self.teacher_public_heuristic_coef = 0.0
        self.teacher_public_nonpass_over_pass_coef = 0.0
        self.teacher_action_margin_coef = 0.0
        self.teacher_same_family_action_margin_coef = 0.0
        self.timings: list[tuple[str, float]] = []
        self.packed_public_target_calls = 0
        self.factorized_teacher_view_calls: list[bool] = []
        self.factorized_view = SimpleNamespace(row_has_candidates=torch.ones((1,), dtype=torch.bool))

    def _record_timing_ms(self, name: str, duration: float) -> None:
        self.timings.append((name, duration))

    def _packed_public_heuristic_target_logits(
        self,
        *,
        forward_model: Any,
        obs: torch.Tensor,
        loss_mask: torch.Tensor,
        packed_legal: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        observation_context: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        del forward_model, obs, loss_mask, observation_context
        self.packed_public_target_calls += 1
        return torch.arange(int(packed_legal[0].numel()), dtype=torch.float32)

    def _factorized_public_heuristic_teacher_view(
        self,
        batch: Any,
        *,
        obs: torch.Tensor,
        loss_mask: torch.Tensor,
        packed_legal: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        score_public_target: bool,
    ) -> tuple[Any, torch.Tensor | None]:
        del batch, obs, loss_mask, packed_legal
        self.factorized_teacher_view_calls.append(score_public_target)
        target_logits = torch.ones((3,), dtype=torch.float32) if score_public_target else None
        return self.factorized_view, target_logits


def test_prepare_impala_teacher_target_inputs_builds_packed_view_and_public_target() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)
    obs = torch.ones((1, 1, 2), dtype=torch.float32)
    packed_logits = torch.as_tensor([0.0, 1.0, -1.0], dtype=torch.float32)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={},
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        obs=obs,
        logits=None,
        packed_logits=packed_logits,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=None,
        forward_observation_context={"obs": obs.reshape(1, 2)},
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is not None
    assert result.teacher_aux_packed_view is result.packed_view
    assert result.public_heuristic_target_logits is not None
    torch.testing.assert_close(result.public_heuristic_target_logits, torch.as_tensor([0.0, 1.0, 2.0]))
    assert learner.packed_public_target_calls == 1
    assert [name for name, _duration in learner.timings] == ["learner_packed_view", "learner_public_heuristic_target"]


def test_prepare_impala_teacher_target_inputs_respects_teacher_aux_gate() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={},
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=torch.zeros((1, 1, _teacher_aux_catalog().action_space_size), dtype=torch.float32),
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=None,
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=False,
    )

    assert result.packed_view is not None
    assert result.teacher_aux_packed_view is result.packed_view
    assert result.public_heuristic_target_logits is None
    assert learner.packed_public_target_calls == 0
    assert [name for name, _duration in learner.timings] == ["learner_packed_view"]


def test_resolve_impala_teacher_target_plan_names_candidate_target_gates() -> None:
    packed_legal = (
        torch.as_tensor([0, 5, 19], dtype=torch.long),
        torch.as_tensor([0, 3], dtype=torch.long),
        None,
    )
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0

    disabled = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=False,
    )
    unsupported = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=object(),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=True,
    )
    supported = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=SimpleNamespace(score_packed_public_heuristic_candidates=object()),
        packed_legal=packed_legal,
        factorized_result=None,
        teacher_aux_enabled=True,
    )

    assert disabled.public_candidate_target_active is True
    assert disabled.factorized_candidate_teacher_view_active is False
    assert disabled.can_prepare_candidate_targets is False
    assert unsupported.can_prepare_candidate_targets is False
    assert supported.can_prepare_candidate_targets is True


def test_resolve_impala_teacher_target_plan_allows_factorized_margin_without_public_model_support() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_action_margin_coef = 1.0
    packed_legal = (
        torch.as_tensor([0, 5, 19], dtype=torch.long),
        torch.as_tensor([0, 3], dtype=torch.long),
        None,
    )

    plan = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=object(),
        packed_legal=packed_legal,
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        teacher_aux_enabled=True,
    )

    assert plan.public_candidate_target_active is False
    assert plan.factorized_candidate_teacher_view_active is True
    assert plan.can_prepare_candidate_targets is True


def test_prepare_impala_teacher_target_inputs_scores_factorized_public_target_when_active() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_public_heuristic_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={"sample": True},
        forward_model=object(),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=None,
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is None
    assert result.teacher_aux_packed_view is learner.factorized_view
    assert result.public_heuristic_target_logits is not None
    torch.testing.assert_close(result.public_heuristic_target_logits, torch.ones((3,), dtype=torch.float32))
    assert learner.factorized_teacher_view_calls == [True]


def test_prepare_impala_teacher_target_inputs_requests_factorized_margin_view_without_public_target() -> None:
    learner = _TeacherTargetInputLearner()
    learner.teacher_action_margin_coef = 1.0
    packed_ids = torch.as_tensor([0, 5, 19], dtype=torch.long)
    packed_offsets = torch.as_tensor([0, 3], dtype=torch.long)
    packed_meta = torch.as_tensor(_packed_meta_from_ids(_teacher_aux_catalog(), packed_ids.numpy()), dtype=torch.long)

    result = prepare_impala_teacher_target_inputs(
        learner=learner,
        batch={"sample": True},
        forward_model=object(),
        obs=torch.ones((1, 1, 2), dtype=torch.float32),
        logits=None,
        packed_logits=None,
        packed_legal=(packed_ids, packed_offsets, packed_meta),
        loss_mask=torch.ones((1, 1), dtype=torch.float32),
        factorized_result=SimpleNamespace(values=torch.zeros((1, 1))),
        forward_observation_context=None,
        need_packed_view=True,
        teacher_aux_enabled=True,
    )

    assert result.packed_view is None
    assert result.teacher_aux_packed_view is learner.factorized_view
    assert result.public_heuristic_target_logits is None
    assert learner.factorized_teacher_view_calls == [False]
    assert learner.timings == []
