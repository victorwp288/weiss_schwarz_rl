"""IMPALA learner orchestration for paired-swing auxiliary losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.impala.paired_auxiliary_batch import (
    batch_value,
    resolve_paired_auxiliary_batch_inputs,
    resolve_paired_auxiliary_reset_before_step,
)
from weiss_rl.learners.impala.paired_swing_candidates import compute_paired_swing_candidate_view
from weiss_rl.learners.impala.paired_swing_outputs import (
    build_paired_swing_auxiliary_context,
    build_paired_swing_auxiliary_metrics,
)
from weiss_rl.learners.paired_swing.loss import (
    packed_paired_swing_margin_loss,
    packed_target_action_retention_loss,
    packed_top_action_retention_loss,
)


class ImpalaPairedSwingAuxiliaryMixin:
    def _paired_swing_loss_and_metrics(
        self: Any,
        batch: Any,
        *,
        margin: float,
        coef: float,
        positive_action_source: str,
        negative_action_source: str,
        loss_scope: str = "row",
        compare_to: str = "negative",
        margin_retention_coef: float = 0.0,
        margin_retention_margin: float = 0.0,
        top_action_retention_coef: float = 0.0,
        top_action_retention_margin: float = 0.0,
    ) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute paired-swing losses")

        inputs = resolve_paired_auxiliary_batch_inputs(
            self,
            batch,
            packed_legal_error="paired-swing replay requires packed legal_ids/legal_offsets",
        )
        obs = inputs.obs
        expected_shape = inputs.expected_shape
        packed_legal = inputs.packed_legal
        loss_mask = inputs.loss_mask

        positive_actions = self._paired_swing_action_tensor(
            batch,
            source=positive_action_source,
            expected_shape=expected_shape,
        )
        negative_actions = self._paired_swing_action_tensor(
            batch,
            source=negative_action_source,
            expected_shape=expected_shape,
        )
        negative_valid = self._optional_time_major_bool_field(
            batch_value(batch, "teacher_valid"),
            field_name="teacher_valid",
            expected_shape=expected_shape,
        )
        if negative_valid is None:
            negative_valid = torch.ones(expected_shape, device=obs.device, dtype=torch.bool)
        source_label_id = self._optional_time_major_index_field(
            batch_value(batch, "source_label_id"),
            field_name="source_label_id",
            expected_shape=expected_shape,
        )

        candidate_view = compute_paired_swing_candidate_view(
            self,
            batch,
            obs=obs,
            expected_shape=expected_shape,
            packed_legal=packed_legal,
            loss_mask=loss_mask,
            margin_retention_coef=float(margin_retention_coef),
            top_action_retention_coef=float(top_action_retention_coef),
        )

        base_loss, swing_metrics, swing_context = packed_paired_swing_margin_loss(
            packed_logits=candidate_view.packed_view.logits,
            legal_ids=packed_legal[0],
            legal_offsets=packed_legal[1],
            positive_actions=positive_actions,
            negative_actions=negative_actions,
            negative_valid=negative_valid,
            loss_mask=loss_mask,
            margin=float(margin),
            pass_action_id=self.pass_action_id,
            loss_scope=str(loss_scope),
            compare_to=str(compare_to),
            group_ids=source_label_id,
            reference_packed_logits=candidate_view.reference_packed_logits,
            margin_retention_coef=float(margin_retention_coef),
            margin_retention_margin=float(margin_retention_margin),
            top_action_retention_coef=float(top_action_retention_coef),
            top_action_retention_margin=float(top_action_retention_margin),
        )
        weighted_loss = base_loss * float(coef)
        context = build_paired_swing_auxiliary_context(
            weighted_loss=weighted_loss,
            candidate_view=candidate_view,
            loss_mask=loss_mask,
            swing_context=swing_context,
        )
        self._ensure_finite_tensor("paired_swing_loss", weighted_loss, batch=batch, context=context)
        metrics = build_paired_swing_auxiliary_metrics(
            weighted_loss=weighted_loss,
            coef=float(coef),
            margin=float(margin),
            positive_action_source=positive_action_source,
            negative_action_source=negative_action_source,
            loss_scope=loss_scope,
            compare_to=compare_to,
            margin_retention_coef=float(margin_retention_coef),
            margin_retention_margin=float(margin_retention_margin),
            top_action_retention_coef=float(top_action_retention_coef),
            top_action_retention_margin=float(top_action_retention_margin),
            swing_metrics=swing_metrics,
        )
        return weighted_loss if float(coef) != 0.0 else candidate_view.zero, metrics, context

    def _paired_swing_full_surface_top_action_retention_loss_and_metrics(
        self: Any,
        batch: Any,
        *,
        coef: float,
        margin: float,
        mode: str = "reference_top",
    ) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute full-surface paired-swing retention")
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"reference_top", "target_action"}:
            raise ValueError("full-surface paired-swing retention mode must be one of: reference_top, target_action")

        inputs = resolve_paired_auxiliary_batch_inputs(
            self,
            batch,
            packed_legal_error="full-surface paired-swing retention requires packed legal_ids/legal_offsets",
        )
        obs = inputs.obs
        expected_shape = inputs.expected_shape
        packed_legal = inputs.packed_legal
        loss_mask = inputs.loss_mask

        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        if not self._should_use_factorized_legal_policy(forward_model, packed_legal=packed_legal):
            raise ValueError("full-surface paired-swing retention requires the factorized packed learner path")
        factorized_view, _target_logits = self._factorized_public_heuristic_teacher_view(
            batch,
            obs=obs,
            loss_mask=loss_mask,
            packed_legal=packed_legal,
            score_public_target=False,
            reattach_initial_hidden_context_gradient=True,
        )
        if factorized_view is None:
            raise ValueError("full-surface paired-swing retention could not build a packed candidate view")
        reference_packed_logits = None
        if normalized_mode == "reference_top":
            anchor_model = self._ensure_policy_anchor_model()
            with torch.no_grad():
                reference_packed_logits = self._factorized_candidate_log_probs_for_model(
                    anchor_model,
                    batch,
                    obs=obs,
                    packed_legal=packed_legal,
                    reset_before_step=resolve_paired_auxiliary_reset_before_step(
                        self,
                        batch,
                        expected_shape=expected_shape,
                    ),
                )
            base_loss, retention_metrics, retention_context = packed_top_action_retention_loss(
                packed_logits=factorized_view.logits,
                reference_packed_logits=reference_packed_logits,
                legal_ids=packed_legal[0],
                legal_offsets=packed_legal[1],
                loss_mask=loss_mask,
                retention_margin=float(margin),
                metric_prefix="paired_swing_full_surface",
            )
        else:
            target_actions = self._require_actions(batch_value(batch, "actions"), expected_shape=expected_shape)
            target_valid = target_actions >= 0
            base_loss, retention_metrics, retention_context = packed_target_action_retention_loss(
                packed_logits=factorized_view.logits,
                legal_ids=packed_legal[0],
                legal_offsets=packed_legal[1],
                target_actions=target_actions,
                target_valid=target_valid,
                loss_mask=loss_mask,
                retention_margin=float(margin),
                metric_prefix="paired_swing_full_surface_target",
            )
        weighted_loss = base_loss * float(coef)
        context: dict[str, Any] = {
            "paired_swing_full_surface_top_action_retention_loss": weighted_loss.detach(),
            "paired_swing_full_surface_packed_logits": factorized_view.logits.detach(),
            "policy_train_mask": loss_mask.detach(),
            **retention_context,
        }
        self._ensure_finite_tensor(
            "paired_swing_full_surface_top_action_retention_loss",
            weighted_loss,
            batch=batch,
            context=context,
        )
        metrics = {
            "paired_swing_full_surface_top_action_retention_weighted_loss": float(weighted_loss.detach().item()),
            "paired_swing_full_surface_top_action_retention_coef": float(coef),
            "paired_swing_full_surface_top_action_retention_margin": float(margin),
            "paired_swing_full_surface_top_action_retention_mode_reference_top": 1.0
            if normalized_mode == "reference_top"
            else 0.0,
            "paired_swing_full_surface_top_action_retention_mode_target_action": 1.0
            if normalized_mode == "target_action"
            else 0.0,
        }
        metrics.update(retention_metrics)
        zero = factorized_view.logits.sum() * 0.0
        return weighted_loss if float(coef) != 0.0 else zero, metrics, context

    def _paired_swing_action_tensor(self: Any, batch: Any, *, source: str, expected_shape: torch.Size) -> Tensor:
        normalized = str(source).strip().lower()
        if normalized == "actions":
            return self._require_actions(batch_value(batch, "actions"), expected_shape=expected_shape)
        if normalized == "teacher_action":
            teacher_action = self._optional_time_major_index_field(
                batch_value(batch, "teacher_action"),
                field_name="teacher_action",
                expected_shape=expected_shape,
            )
            if teacher_action is None:
                raise ValueError("paired-swing replay source teacher_action requires batch.teacher_action")
            return teacher_action
        raise ValueError("paired-swing action source must be one of: actions, teacher_action")


__all__ = ["ImpalaPairedSwingAuxiliaryMixin"]
