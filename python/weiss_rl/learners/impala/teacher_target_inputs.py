"""IMPALA structured-teacher packed views and public-heuristic targets."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.structured_auxiliary import PackedStructuredLegalView, packed_structured_legal_view


@dataclass(frozen=True, slots=True)
class ImpalaTeacherTargetInputs:
    packed_view: PackedStructuredLegalView | None
    teacher_aux_packed_view: PackedStructuredLegalView | None
    public_heuristic_target_logits: Tensor | None


@dataclass(frozen=True, slots=True)
class ImpalaTeacherTargetPlan:
    public_candidate_target_active: bool
    factorized_candidate_teacher_view_active: bool
    can_prepare_candidate_targets: bool


@dataclass(frozen=True, slots=True)
class ImpalaTeacherCandidateTargets:
    teacher_aux_packed_view: PackedStructuredLegalView | None
    public_heuristic_target_logits: Tensor | None


def prepare_impala_teacher_target_inputs(
    *,
    learner: Any,
    batch: Any,
    forward_model: Any,
    obs: Tensor,
    logits: Tensor | None,
    packed_logits: Tensor | None,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    loss_mask: Tensor,
    factorized_result: Any,
    forward_observation_context: Mapping[str, Tensor] | None,
    need_packed_view: bool,
    teacher_aux_enabled: bool,
) -> ImpalaTeacherTargetInputs:
    packed_view = build_impala_packed_structured_view(
        learner=learner,
        logits=logits,
        packed_logits=packed_logits,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        need_packed_view=need_packed_view,
    )
    plan = resolve_impala_teacher_target_plan(
        learner=learner,
        forward_model=forward_model,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        teacher_aux_enabled=teacher_aux_enabled,
    )
    candidate_targets = resolve_impala_teacher_candidate_targets(
        learner=learner,
        batch=batch,
        forward_model=forward_model,
        obs=obs,
        loss_mask=loss_mask,
        packed_legal=packed_legal,
        factorized_result=factorized_result,
        forward_observation_context=forward_observation_context,
        initial_teacher_aux_packed_view=packed_view,
        plan=plan,
    )

    return ImpalaTeacherTargetInputs(
        packed_view=packed_view,
        teacher_aux_packed_view=candidate_targets.teacher_aux_packed_view,
        public_heuristic_target_logits=candidate_targets.public_heuristic_target_logits,
    )


def resolve_impala_teacher_target_plan(
    *,
    learner: Any,
    forward_model: Any,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    factorized_result: Any,
    teacher_aux_enabled: bool,
) -> ImpalaTeacherTargetPlan:
    public_candidate_target_active = (
        float(learner.teacher_public_heuristic_coef) != 0.0
        or float(learner.teacher_public_nonpass_over_pass_coef) != 0.0
    )
    factorized_candidate_teacher_view_active = factorized_result is not None and (
        public_candidate_target_active
        or float(learner.teacher_action_margin_coef) != 0.0
        or float(learner.teacher_same_family_action_margin_coef) != 0.0
    )
    can_prepare_candidate_targets = bool(
        teacher_aux_enabled
        and packed_legal is not None
        and (public_candidate_target_active or factorized_candidate_teacher_view_active)
        and (factorized_result is not None or hasattr(forward_model, "score_packed_public_heuristic_candidates"))
    )
    return ImpalaTeacherTargetPlan(
        public_candidate_target_active=public_candidate_target_active,
        factorized_candidate_teacher_view_active=factorized_candidate_teacher_view_active,
        can_prepare_candidate_targets=can_prepare_candidate_targets,
    )


def resolve_impala_teacher_candidate_targets(
    *,
    learner: Any,
    batch: Any,
    forward_model: Any,
    obs: Tensor,
    loss_mask: Tensor,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    factorized_result: Any,
    forward_observation_context: Mapping[str, Tensor] | None,
    initial_teacher_aux_packed_view: PackedStructuredLegalView | None,
    plan: ImpalaTeacherTargetPlan,
) -> ImpalaTeacherCandidateTargets:
    teacher_aux_packed_view = initial_teacher_aux_packed_view
    public_heuristic_target_logits = None
    if not plan.can_prepare_candidate_targets:
        return ImpalaTeacherCandidateTargets(
            teacher_aux_packed_view=teacher_aux_packed_view,
            public_heuristic_target_logits=public_heuristic_target_logits,
        )

    assert packed_legal is not None
    if factorized_result is not None:
        teacher_aux_packed_view, public_heuristic_target_logits = learner._factorized_public_heuristic_teacher_view(
            batch,
            obs=obs,
            loss_mask=loss_mask,
            packed_legal=packed_legal,
            score_public_target=plan.public_candidate_target_active,
        )
    elif plan.public_candidate_target_active:
        heuristic_started = time.perf_counter()
        with torch.no_grad():
            public_heuristic_target_logits = learner._packed_public_heuristic_target_logits(
                forward_model=forward_model,
                obs=obs,
                loss_mask=loss_mask,
                packed_legal=packed_legal,
                observation_context=forward_observation_context,
            )
        learner._record_timing_ms("learner_public_heuristic_target", time.perf_counter() - heuristic_started)

    return ImpalaTeacherCandidateTargets(
        teacher_aux_packed_view=teacher_aux_packed_view,
        public_heuristic_target_logits=public_heuristic_target_logits,
    )


def build_impala_packed_structured_view(
    *,
    learner: Any,
    logits: Tensor | None,
    packed_logits: Tensor | None,
    packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
    factorized_result: Any,
    need_packed_view: bool,
) -> PackedStructuredLegalView | None:
    if packed_legal is None or factorized_result is not None or not need_packed_view:
        return None
    packed_view_started = time.perf_counter()
    packed_view = packed_structured_legal_view(
        logits=packed_logits if packed_logits is not None else logits,
        packed_ids=packed_legal[0],
        packed_offsets=packed_legal[1],
        packed_meta=packed_legal[2],
    )
    learner._record_timing_ms("learner_packed_view", time.perf_counter() - packed_view_started)
    return packed_view


__all__ = [
    "ImpalaTeacherCandidateTargets",
    "ImpalaTeacherTargetInputs",
    "ImpalaTeacherTargetPlan",
    "build_impala_packed_structured_view",
    "prepare_impala_teacher_target_inputs",
    "resolve_impala_teacher_candidate_targets",
    "resolve_impala_teacher_target_plan",
]
