"""Candidate log-prob assembly for IMPALA paired-outcome preference replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.action_logp import packed_selected_action_logp
from weiss_rl.learners.impala.paired_auxiliary_batch import PackedLegalWithMeta
from weiss_rl.learners.tensor_ops import segment_max


@dataclass(frozen=True)
class PairedOutcomeCandidateLogps:
    current_candidate_log_probs: Tensor
    reference_candidate_log_probs: Tensor
    current_action_logp: Tensor
    current_best_non_target_logp: Tensor
    reference_action_logp: Tensor
    reference_best_non_target_logp: Tensor


def compute_paired_outcome_candidate_logps(
    learner: Any,
    batch: Any,
    *,
    obs: Tensor,
    packed_legal: PackedLegalWithMeta,
    actions: Tensor,
    reset_before_step: Tensor | None,
) -> PairedOutcomeCandidateLogps:
    forward_model = learner.compiled_model if learner.compiled_model is not None else learner.model
    if not learner._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
        raise ValueError("paired outcome preference replay currently requires the factorized packed learner path")
    anchor_model = learner._ensure_policy_anchor_model()
    if not learner._should_use_factorized_legal_policy(anchor_model, packed_legal=packed_legal):
        raise ValueError("paired outcome preference replay requires a factorized structured reference model")

    current_candidate_log_probs = learner._factorized_candidate_log_probs_for_model(
        forward_model,
        batch,
        obs=obs,
        packed_legal=packed_legal,
        reset_before_step=reset_before_step,
    )
    with torch.no_grad():
        reference_candidate_log_probs = learner._factorized_candidate_log_probs_for_model(
            anchor_model,
            batch,
            obs=obs,
            packed_legal=packed_legal,
            reset_before_step=reset_before_step,
        )

    expected_shape = obs.shape[:2]
    current_action_logp = packed_selected_action_logp(
        current_candidate_log_probs,
        packed_legal[0],
        packed_legal[1],
        actions,
        pass_action_id=learner.pass_action_id,
        strict=False,
    ).reshape(expected_shape)
    current_best_non_target_logp = packed_best_non_target_logp(
        current_candidate_log_probs,
        packed_legal[0],
        packed_legal[1],
        actions,
    ).reshape(expected_shape)
    reference_action_logp = packed_selected_action_logp(
        reference_candidate_log_probs,
        packed_legal[0],
        packed_legal[1],
        actions,
        pass_action_id=learner.pass_action_id,
        strict=False,
    ).reshape(expected_shape)
    reference_best_non_target_logp = packed_best_non_target_logp(
        reference_candidate_log_probs,
        packed_legal[0],
        packed_legal[1],
        actions,
    ).reshape(expected_shape)
    return PairedOutcomeCandidateLogps(
        current_candidate_log_probs=current_candidate_log_probs,
        reference_candidate_log_probs=reference_candidate_log_probs,
        current_action_logp=current_action_logp,
        current_best_non_target_logp=current_best_non_target_logp,
        reference_action_logp=reference_action_logp,
        reference_best_non_target_logp=reference_best_non_target_logp,
    )


def packed_best_non_target_logp(
    candidate_log_probs: Tensor,
    packed_ids: Tensor,
    packed_offsets: Tensor,
    actions: Tensor,
) -> Tensor:
    offsets = packed_offsets.to(device=candidate_log_probs.device, dtype=torch.long)
    ids = packed_ids.to(device=candidate_log_probs.device, dtype=torch.long)
    row_count = int(offsets.numel() - 1)
    lengths = offsets[1:] - offsets[:-1]
    row_indices = torch.repeat_interleave(
        torch.arange(row_count, dtype=torch.long, device=candidate_log_probs.device),
        lengths,
    )
    if int(row_indices.numel()) != int(candidate_log_probs.numel()):
        raise ValueError("packed offsets do not match packed candidate log-probs")
    flat_actions = actions.reshape(-1).to(device=candidate_log_probs.device, dtype=torch.long)
    if int(flat_actions.numel()) != row_count:
        raise ValueError(f"actions row count {int(flat_actions.numel())} does not match packed row count {row_count}")
    row_targets = flat_actions.index_select(0, row_indices)
    non_target_scores = torch.where(
        ids != row_targets,
        candidate_log_probs.to(dtype=torch.float32),
        torch.full_like(candidate_log_probs.to(dtype=torch.float32), -torch.inf),
    )
    return segment_max(non_target_scores, row_indices, row_count)


__all__ = [
    "PairedOutcomeCandidateLogps",
    "compute_paired_outcome_candidate_logps",
    "packed_best_non_target_logp",
]
