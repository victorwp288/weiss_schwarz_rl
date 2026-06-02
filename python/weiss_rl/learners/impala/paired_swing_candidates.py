"""Candidate-view assembly for IMPALA paired-swing replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.impala.paired_auxiliary_batch import (
    PackedLegalWithMeta,
    batch_value,
    resolve_paired_auxiliary_reset_before_step,
)
from weiss_rl.learners.structured_auxiliary import packed_structured_legal_view


@dataclass(frozen=True)
class PairedSwingCandidateView:
    packed_view: Any
    zero: Tensor
    logits: Tensor | None
    values: Tensor | None
    reference_packed_logits: Tensor | None


def compute_paired_swing_candidate_view(
    learner: Any,
    batch: Any,
    *,
    obs: Tensor,
    expected_shape: torch.Size,
    packed_legal: PackedLegalWithMeta,
    loss_mask: Tensor,
    margin_retention_coef: float,
    top_action_retention_coef: float,
) -> PairedSwingCandidateView:
    forward_model = learner.compiled_model if learner.compiled_model is not None else learner.model
    if learner._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
        return _factorized_paired_swing_candidate_view(
            learner,
            batch,
            obs=obs,
            expected_shape=expected_shape,
            packed_legal=packed_legal,
            loss_mask=loss_mask,
            margin_retention_coef=margin_retention_coef,
            top_action_retention_coef=top_action_retention_coef,
        )
    return _dense_paired_swing_candidate_view(
        learner,
        batch,
        obs=obs,
        packed_legal=packed_legal,
        loss_mask=loss_mask,
        margin_retention_coef=margin_retention_coef,
        top_action_retention_coef=top_action_retention_coef,
    )


def _factorized_paired_swing_candidate_view(
    learner: Any,
    batch: Any,
    *,
    obs: Tensor,
    expected_shape: torch.Size,
    packed_legal: PackedLegalWithMeta,
    loss_mask: Tensor,
    margin_retention_coef: float,
    top_action_retention_coef: float,
) -> PairedSwingCandidateView:
    factorized_view, _target_logits = learner._factorized_public_heuristic_teacher_view(
        batch,
        obs=obs,
        loss_mask=loss_mask,
        packed_legal=packed_legal,
        score_public_target=False,
        reattach_initial_hidden_context_gradient=True,
    )
    if factorized_view is None:
        raise ValueError("factorized paired-swing replay could not build a packed candidate view")

    reference_packed_logits = None
    if float(margin_retention_coef) != 0.0 or float(top_action_retention_coef) != 0.0:
        anchor_model = learner._ensure_policy_anchor_model()
        with torch.no_grad():
            reference_packed_logits = learner._factorized_candidate_log_probs_for_model(
                anchor_model,
                batch,
                obs=obs,
                packed_legal=packed_legal,
                reset_before_step=resolve_paired_auxiliary_reset_before_step(
                    learner,
                    batch,
                    expected_shape=expected_shape,
                ),
            )
    return PairedSwingCandidateView(
        packed_view=factorized_view,
        zero=factorized_view.logits.sum() * 0.0,
        logits=None,
        values=None,
        reference_packed_logits=reference_packed_logits,
    )


def _dense_paired_swing_candidate_view(
    learner: Any,
    batch: Any,
    *,
    obs: Tensor,
    packed_legal: PackedLegalWithMeta,
    loss_mask: Tensor,
    margin_retention_coef: float,
    top_action_retention_coef: float,
) -> PairedSwingCandidateView:
    if float(margin_retention_coef) != 0.0 or float(top_action_retention_coef) != 0.0:
        raise ValueError("paired-swing retention currently requires the factorized packed learner path")
    legal_actions = batch_value(batch, "legal_actions")
    if legal_actions is None:
        legal_actions = learner._packed_legal_action_view(packed_legal)
    forward = learner._forward_time_major(
        obs,
        initial_hidden_state=batch_value(batch, "initial_hidden_state"),
        to_play_seat=batch_value(batch, "to_play_seat"),
        actor=batch_value(batch, "actor"),
        legal_actions=legal_actions,
        policy_train_mask=loss_mask,
        reset_before_step=batch_value(batch, "reset_before_step"),
        opponent_context_index=batch_value(batch, "opponent_context_index"),
    )
    packed_view = packed_structured_legal_view(
        logits=forward.packed_logits if forward.packed_logits is not None else forward.logits,
        packed_ids=packed_legal[0],
        packed_offsets=packed_legal[1],
        packed_meta=packed_legal[2],
    )
    if packed_view is None:
        raise ValueError("paired-swing replay requires packed legal action metadata")
    return PairedSwingCandidateView(
        packed_view=packed_view,
        zero=forward.values.sum() * 0.0,
        logits=forward.logits,
        values=forward.values,
        reference_packed_logits=None,
    )


__all__ = ["PairedSwingCandidateView", "compute_paired_swing_candidate_view"]
