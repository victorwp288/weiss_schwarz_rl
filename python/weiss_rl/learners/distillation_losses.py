"""Distillation and counterfactual auxiliary losses for IMPALA."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.impala_helpers import (
    _batch_value,
    _packed_selected_action_logp,
    _segment_logsumexp,
)


class ImpalaDistillationLossMixin:
    def _set_public_heuristic_bias_scale_if_supported(self, model: Any, value: float) -> tuple[float, float] | None:
        get_bias_scale = getattr(model, "get_public_heuristic_logit_bias_scale", None)
        set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
        if not callable(get_bias_scale) or not callable(set_bias_scale):
            return None
        previous = (
            float(get_bias_scale(scoring_mode="learner")),
            float(get_bias_scale(scoring_mode="actor")),
        )
        set_bias_scale(float(value), actor_value=float(value))
        return previous

    def _restore_public_heuristic_bias_scale_if_supported(
        self,
        model: Any,
        previous: tuple[float, float] | None,
    ) -> None:
        if previous is None:
            return
        set_bias_scale = getattr(model, "set_public_heuristic_logit_bias_scale", None)
        if callable(set_bias_scale):
            set_bias_scale(float(previous[0]), actor_value=float(previous[1]))

    def _packed_raw_distill_kl_and_metrics(
        self,
        *,
        student_logits: Tensor,
        teacher_logits: Tensor,
        packed_ids: Tensor,
        packed_offsets: Tensor,
        loss_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if student_logits.ndim != 1 or teacher_logits.ndim != 1:
            raise ValueError("raw B1 distill packed logits must be 1D")
        if int(student_logits.numel()) != int(teacher_logits.numel()):
            raise ValueError("raw B1 distill student/teacher packed logits must align")
        ids = packed_ids.reshape(-1).to(device=student_logits.device, dtype=torch.long)
        offsets = packed_offsets.reshape(-1).to(device=student_logits.device, dtype=torch.long)
        if int(offsets.numel()) != int(loss_mask.numel()) + 1:
            raise ValueError("raw B1 distill packed offsets must align with the loss mask rows")
        if int(offsets[-1].item()) != int(ids.numel()) or int(ids.numel()) != int(student_logits.numel()):
            raise ValueError("raw B1 distill packed ids, offsets, and logits must align")

        row_count = int(offsets.numel()) - 1
        widths = offsets[1:] - offsets[:-1]
        row_indices = torch.repeat_interleave(
            torch.arange(row_count, device=student_logits.device, dtype=torch.long),
            widths,
        )
        zero = student_logits.new_zeros(())
        if row_count <= 0 or int(student_logits.numel()) == 0:
            return zero, {
                "raw_b1_distill_row_fraction": zero,
                "raw_b1_top1_match": zero,
                "raw_b1_topk_overlap": zero,
                "raw_b1_family_match": zero,
                "raw_b1_kl": zero,
                "raw_b1_top_action_ce": zero,
            }

        temperature = max(float(self.raw_b1_distill_temperature), 1.0e-6)
        scaled_student = student_logits / temperature
        scaled_teacher = teacher_logits.detach() / temperature
        student_log_z = _segment_logsumexp(scaled_student, row_indices, row_count)
        teacher_log_z = _segment_logsumexp(scaled_teacher, row_indices, row_count)
        student_logp = scaled_student - student_log_z.index_select(0, row_indices)
        teacher_logp = scaled_teacher - teacher_log_z.index_select(0, row_indices)
        teacher_probs = torch.exp(teacher_logp)
        row_kl_terms = teacher_probs * (teacher_logp - student_logp)
        row_kl = torch.zeros((row_count,), dtype=student_logits.dtype, device=student_logits.device)
        row_kl.scatter_add_(0, row_indices, row_kl_terms)

        flat_mask = loss_mask.reshape(-1).to(device=student_logits.device, dtype=student_logits.dtype)
        active_mask = flat_mask * (widths > 0).to(device=student_logits.device, dtype=student_logits.dtype)
        denominator = torch.clamp(active_mask.sum(), min=1.0)
        kl_loss = (row_kl * active_mask).sum() / denominator
        loss = kl_loss
        teacher_top_actions = self._packed_top_action_ids(teacher_logits.detach(), ids, offsets)
        top_action_ce = student_logits.new_zeros(())
        if float(self.raw_b1_distill_top_action_ce_coef) != 0.0:
            teacher_top_logp = _packed_selected_action_logp(
                scaled_student,
                ids,
                offsets,
                teacher_top_actions,
                pass_action_id=self.pass_action_id,
                strict=False,
            )
            valid_teacher_top = torch.isfinite(teacher_top_logp).to(
                dtype=student_logits.dtype, device=student_logits.device
            )
            ce_mask = active_mask * valid_teacher_top
            ce_denominator = torch.clamp(ce_mask.sum(), min=1.0)
            top_action_ce = (
                -(torch.where(ce_mask > 0.0, teacher_top_logp, torch.zeros_like(teacher_top_logp)) * ce_mask).sum()
                / ce_denominator
            )
            loss = loss + (float(self.raw_b1_distill_top_action_ce_coef) * top_action_ce)

        with torch.no_grad():
            student_top_actions = self._packed_top_action_ids(student_logits.detach(), ids, offsets)
            valid_top = (active_mask > 0.0) & (teacher_top_actions >= 0) & (student_top_actions >= 0)
            top_denominator = torch.clamp(valid_top.to(dtype=student_logits.dtype).sum(), min=1.0)
            top1_match = ((teacher_top_actions == student_top_actions) & valid_top).to(
                dtype=student_logits.dtype
            ).sum() / top_denominator
            row_fraction = active_mask.sum() / torch.clamp(
                torch.as_tensor(row_count, device=student_logits.device, dtype=student_logits.dtype), min=1.0
            )
            action_family_ids = self._action_family_ids_tensor(
                action_dim=int(getattr(getattr(self.model, "action_catalog", None), "action_space_size", 0)),
                device=student_logits.device,
            )
            if action_family_ids is None:
                family_match = zero
            else:
                safe_teacher = torch.clamp(teacher_top_actions, min=0, max=action_family_ids.numel() - 1)
                safe_student = torch.clamp(student_top_actions, min=0, max=action_family_ids.numel() - 1)
                teacher_family = action_family_ids.index_select(0, safe_teacher)
                student_family = action_family_ids.index_select(0, safe_student)
                family_match = ((teacher_family == student_family) & valid_top).to(
                    dtype=student_logits.dtype
                ).sum() / top_denominator
            top_k = max(1, int(self.raw_b1_distill_top_k))
            overlap_sum = student_logits.new_zeros(())
            overlap_rows = 0
            valid_rows = torch.nonzero(active_mask > 0.0, as_tuple=False).squeeze(1)
            for row_tensor in valid_rows[:2048]:
                row = int(row_tensor.item())
                start = int(offsets[row].item())
                end = int(offsets[row + 1].item())
                width = max(end - start, 0)
                if width <= 0:
                    continue
                k = min(top_k, width)
                teacher_order = torch.topk(teacher_logits[start:end].detach(), k=k).indices
                student_order = torch.topk(student_logits[start:end].detach(), k=k).indices
                teacher_ids = set(int(value) for value in ids[start:end].index_select(0, teacher_order).tolist())
                student_ids = set(int(value) for value in ids[start:end].index_select(0, student_order).tolist())
                overlap_sum = overlap_sum + (len(teacher_ids & student_ids) / float(k))
                overlap_rows += 1
            topk_overlap = overlap_sum / float(overlap_rows) if overlap_rows > 0 else student_logits.new_zeros(())

        return loss, {
            "raw_b1_distill_row_fraction": row_fraction.detach(),
            "raw_b1_top1_match": top1_match.detach(),
            "raw_b1_topk_overlap": topk_overlap.detach(),
            "raw_b1_family_match": family_match.detach(),
            "raw_b1_kl": kl_loss.detach(),
            "raw_b1_top_action_ce": top_action_ce.detach(),
        }

    def _dense_raw_distill_kl_and_metrics(
        self,
        *,
        student_logits: Tensor,
        teacher_logits: Tensor,
        legal_mask: Tensor,
        loss_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        mask = legal_mask.to(device=student_logits.device, dtype=torch.bool)
        if student_logits.shape != teacher_logits.shape or student_logits.shape != mask.shape:
            raise ValueError("raw B1 distill dense logits and legal mask must align")
        temperature = max(float(self.raw_b1_distill_temperature), 1.0e-6)
        masked_student = torch.where(mask, student_logits / temperature, torch.full_like(student_logits, -torch.inf))
        masked_teacher = torch.where(
            mask,
            teacher_logits.detach() / temperature,
            torch.full_like(teacher_logits, -torch.inf),
        )
        student_logp = torch.log_softmax(masked_student, dim=-1)
        teacher_logp = torch.log_softmax(masked_teacher, dim=-1)
        teacher_probs = torch.exp(teacher_logp)
        row_kl = torch.where(mask, teacher_probs * (teacher_logp - student_logp), torch.zeros_like(student_logits)).sum(
            dim=-1
        )
        active_mask = loss_mask.to(device=student_logits.device, dtype=student_logits.dtype) * mask.any(dim=-1).to(
            dtype=student_logits.dtype
        )
        denominator = torch.clamp(active_mask.sum(), min=1.0)
        kl_loss = (row_kl * active_mask).sum() / denominator
        loss = kl_loss
        with torch.no_grad():
            teacher_top = masked_teacher.argmax(dim=-1)
            student_top = masked_student.argmax(dim=-1)
            valid = active_mask > 0.0
            valid_denom = torch.clamp(valid.to(dtype=student_logits.dtype).sum(), min=1.0)
            top1_match = ((teacher_top == student_top) & valid).to(dtype=student_logits.dtype).sum() / valid_denom
        top_action_ce = student_logits.new_zeros(())
        if float(self.raw_b1_distill_top_action_ce_coef) != 0.0:
            student_logp_selected = student_logp.gather(-1, teacher_top.unsqueeze(-1)).squeeze(-1)
            top_action_ce = (
                -(
                    torch.where(active_mask > 0.0, student_logp_selected, torch.zeros_like(student_logp_selected))
                    * active_mask
                ).sum()
                / denominator
            )
            loss = loss + (float(self.raw_b1_distill_top_action_ce_coef) * top_action_ce)
        return loss, {
            "raw_b1_distill_row_fraction": active_mask.sum()
            / torch.clamp(
                torch.as_tensor(active_mask.numel(), device=student_logits.device, dtype=student_logits.dtype), min=1.0
            ),
            "raw_b1_top1_match": top1_match.detach(),
            "raw_b1_topk_overlap": top1_match.detach(),
            "raw_b1_family_match": top1_match.detach(),
            "raw_b1_kl": kl_loss.detach(),
            "raw_b1_top_action_ce": top_action_ce.detach(),
        }

    def _raw_b1_distill_loss_and_metrics(
        self,
        batch: Any,
        *,
        obs: Tensor,
        loss_mask: Tensor,
        packed_legal: tuple[Tensor, Tensor, Tensor | None] | None,
        legal_mask: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero = loss_mask.new_zeros(())
        if self.reference_policy_model is None or float(self.raw_b1_distill_coef) == 0.0:
            return zero, {
                "raw_b1_distill_row_fraction": zero,
                "raw_b1_top1_match": zero,
                "raw_b1_topk_overlap": zero,
                "raw_b1_family_match": zero,
                "raw_b1_kl": zero,
                "raw_b1_top_action_ce": zero,
            }
        teacher_model = self.reference_policy_model
        student_model = self.model
        if student_model is None:
            return zero, {
                "raw_b1_distill_row_fraction": zero,
                "raw_b1_top1_match": zero,
                "raw_b1_topk_overlap": zero,
                "raw_b1_family_match": zero,
                "raw_b1_kl": zero,
            }
        teacher_previous = self._set_public_heuristic_bias_scale_if_supported(
            teacher_model,
            float(self.raw_b1_distill_teacher_bias_scale),
        )
        student_previous = self._set_public_heuristic_bias_scale_if_supported(
            student_model,
            float(self.raw_b1_distill_student_bias_scale),
        )
        try:
            if packed_legal is not None:
                legal_actions = self._packed_legal_action_view(packed_legal)
                with torch.no_grad():
                    teacher_forward = self._forward_time_major(
                        obs,
                        initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                        to_play_seat=_batch_value(batch, "to_play_seat"),
                        actor=_batch_value(batch, "actor"),
                        legal_actions=legal_actions,
                        policy_train_mask=loss_mask,
                        forward_model_override=teacher_model,
                    )
                student_forward = self._forward_time_major(
                    obs,
                    initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                    to_play_seat=_batch_value(batch, "to_play_seat"),
                    actor=_batch_value(batch, "actor"),
                    legal_actions=legal_actions,
                    policy_train_mask=loss_mask,
                    forward_model_override=student_model,
                )
                if teacher_forward.packed_logits is None or student_forward.packed_logits is None:
                    return zero, {
                        "raw_b1_distill_row_fraction": zero,
                        "raw_b1_top1_match": zero,
                        "raw_b1_topk_overlap": zero,
                        "raw_b1_family_match": zero,
                        "raw_b1_kl": zero,
                        "raw_b1_top_action_ce": zero,
                    }
                return self._packed_raw_distill_kl_and_metrics(
                    student_logits=student_forward.packed_logits,
                    teacher_logits=teacher_forward.packed_logits,
                    packed_ids=packed_legal[0],
                    packed_offsets=packed_legal[1],
                    loss_mask=loss_mask,
                )
            if legal_mask is None:
                return zero, {
                    "raw_b1_distill_row_fraction": zero,
                    "raw_b1_top1_match": zero,
                    "raw_b1_topk_overlap": zero,
                    "raw_b1_family_match": zero,
                    "raw_b1_kl": zero,
                    "raw_b1_top_action_ce": zero,
                }
            with torch.no_grad():
                teacher_forward = self._forward_time_major(
                    obs,
                    initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                    to_play_seat=_batch_value(batch, "to_play_seat"),
                    actor=_batch_value(batch, "actor"),
                    legal_actions=_batch_value(batch, "legal_actions"),
                    policy_train_mask=loss_mask,
                    forward_model_override=teacher_model,
                )
            student_forward = self._forward_time_major(
                obs,
                initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
                to_play_seat=_batch_value(batch, "to_play_seat"),
                actor=_batch_value(batch, "actor"),
                legal_actions=_batch_value(batch, "legal_actions"),
                policy_train_mask=loss_mask,
                forward_model_override=student_model,
            )
            if teacher_forward.logits is None or student_forward.logits is None:
                return zero, {
                    "raw_b1_distill_row_fraction": zero,
                    "raw_b1_top1_match": zero,
                    "raw_b1_topk_overlap": zero,
                    "raw_b1_family_match": zero,
                    "raw_b1_kl": zero,
                    "raw_b1_top_action_ce": zero,
                }
            return self._dense_raw_distill_kl_and_metrics(
                student_logits=student_forward.logits,
                teacher_logits=teacher_forward.logits,
                legal_mask=legal_mask,
                loss_mask=loss_mask,
            )
        finally:
            self._restore_public_heuristic_bias_scale_if_supported(student_model, student_previous)
            self._restore_public_heuristic_bias_scale_if_supported(teacher_model, teacher_previous)

    def _load_counterfactual_positive_records(
        self,
        label_dirs: tuple[str, ...],
        *,
        max_labels: int = 0,
    ) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        for raw_dir in label_dirs:
            label_dir = Path(raw_dir)
            labels_path = label_dir / "counterfactual_labels.jsonl"
            if not labels_path.is_file():
                raise FileNotFoundError(f"counterfactual positive label file not found: {labels_path}")
            with labels_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    tensor_ref = row.get("tensor_ref")
                    if not isinstance(tensor_ref, str) or not tensor_ref:
                        continue
                    tensor_path = label_dir / tensor_ref
                    if not tensor_path.is_file():
                        raise FileNotFoundError(f"counterfactual positive tensor not found: {tensor_path}")
                    loaded = torch.load(tensor_path, map_location="cpu", weights_only=False)
                    if not isinstance(loaded, Mapping):
                        raise TypeError(f"counterfactual tensor must load as a mapping: {tensor_path}")
                    obs = torch.as_tensor(loaded["obs"], dtype=torch.float32).detach().cpu()
                    legal_ids = torch.as_tensor(loaded["legal_ids"], dtype=torch.long).detach().cpu().reshape(-1)
                    positive_action_id = int(torch.as_tensor(loaded["positive_action_id"]).item())
                    baseline_action_id = int(
                        torch.as_tensor(loaded.get("baseline_action_id", positive_action_id)).item()
                    )
                    actor_seat = int(torch.as_tensor(loaded.get("actor_seat", row.get("target_seat", 0))).item())
                    if positive_action_id not in set(int(value) for value in legal_ids.tolist()):
                        raise ValueError(
                            f"counterfactual positive action {positive_action_id} is not legal in {tensor_path}"
                        )
                    label_weight = float(
                        torch.as_tensor(loaded.get("label_weight", row.get("label_weight", 1.0))).item()
                    )
                    records.append(
                        {
                            "obs": obs,
                            "legal_ids": legal_ids,
                            "positive_action_id": positive_action_id,
                            "baseline_action_id": baseline_action_id,
                            "actor_seat": actor_seat,
                            "label_weight": max(label_weight, 0.0),
                            "source_path": tensor_path.as_posix(),
                        }
                    )
                    if max_labels > 0 and len(records) >= max_labels:
                        return tuple(records)
        return tuple(records)

    def _counterfactual_positive_loss_and_metrics(self) -> tuple[Tensor, dict[str, Tensor]]:
        reference = self._model_parameter()
        zero = reference.new_zeros(())
        records = self._counterfactual_positive_records
        if float(self.counterfactual_positive_coef) == 0.0 or not records:
            return zero, {
                "counterfactual_positive_ce_loss": zero,
                "counterfactual_positive_margin_loss": zero,
                "counterfactual_positive_label_count": zero,
                "counterfactual_positive_weight_mean": zero,
                "counterfactual_positive_prob_mean": zero,
                "counterfactual_positive_top1_match": zero,
                "counterfactual_positive_logit_margin_mean": zero,
            }
        obs = torch.stack(
            [record["obs"].to(device=reference.device, dtype=reference.dtype) for record in records],
            dim=0,
        )
        actor_seat = torch.as_tensor(
            [int(record["actor_seat"]) for record in records],
            device=reference.device,
            dtype=torch.long,
        )
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        supports_seat_aware = bool(
            hasattr(forward_model, "forward_seat_aware")
            or hasattr(forward_model, "forward_sequence_seat_aware")
            or hasattr(forward_model, "forward_trunk_sequence_seat_aware")
        )
        forward = self._forward_time_major(
            obs.unsqueeze(0),
            to_play_seat=actor_seat.unsqueeze(0) if supports_seat_aware else None,
        )
        if forward.logits is None:
            raise ValueError("counterfactual positive auxiliary requires dense logits from the learner model")
        logits = forward.logits.squeeze(0)
        if logits.ndim != 2:
            raise ValueError(f"counterfactual positive logits must be 2D, got {tuple(logits.shape)}")
        action_dim = int(logits.shape[-1])
        legal_mask = torch.zeros((len(records), action_dim), device=logits.device, dtype=torch.bool)
        positive_ids: list[int] = []
        baseline_ids: list[int] = []
        weights: list[float] = []
        for row_index, record in enumerate(records):
            legal_ids = record["legal_ids"].to(device=logits.device, dtype=torch.long)
            if bool((legal_ids < 0).any().item()) or bool((legal_ids >= action_dim).any().item()):
                raise ValueError("counterfactual positive legal_ids are outside the model action space")
            legal_mask[row_index, legal_ids] = True
            positive_ids.append(int(record["positive_action_id"]))
            baseline_ids.append(int(record["baseline_action_id"]))
            weights.append(float(record["label_weight"]))
        positive = torch.as_tensor(positive_ids, device=logits.device, dtype=torch.long)
        baseline = torch.as_tensor(baseline_ids, device=logits.device, dtype=torch.long)
        label_weights = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype).clamp_min(0.0)
        denominator = torch.clamp(label_weights.sum(), min=1.0)
        masked_logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        row_indices = torch.arange(len(records), device=logits.device)
        positive_logp = log_probs[row_indices, positive]
        ce_loss = -((positive_logp * label_weights).sum() / denominator)
        positive_logits = logits[row_indices, positive]
        baseline_logits = logits[row_indices, baseline]
        margin_gap = positive_logits - baseline_logits
        margin_loss = torch.relu(float(self.counterfactual_positive_margin) - margin_gap)
        margin_loss = (margin_loss * label_weights).sum() / denominator
        total = ce_loss + (float(self.counterfactual_positive_margin_coef) * margin_loss)
        positive_prob = torch.exp(positive_logp)
        top1 = masked_logits.argmax(dim=-1)
        metrics = {
            "counterfactual_positive_ce_loss": ce_loss.detach(),
            "counterfactual_positive_margin_loss": margin_loss.detach(),
            "counterfactual_positive_label_count": reference.new_tensor(float(len(records))),
            "counterfactual_positive_weight_mean": label_weights.mean().detach(),
            "counterfactual_positive_prob_mean": ((positive_prob * label_weights).sum() / denominator).detach(),
            "counterfactual_positive_top1_match": (
                ((top1 == positive).to(dtype=logits.dtype) * label_weights).sum() / denominator
            ).detach(),
            "counterfactual_positive_logit_margin_mean": ((margin_gap * label_weights).sum() / denominator).detach(),
        }
        return total, metrics
