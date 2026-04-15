"""IMPALA learner helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.learners.vtrace import VTraceTargets, VtraceMetrics, compute_vtrace_metrics
from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask
from weiss_rl.replay.bundles import write_fault_bundle
from weiss_rl.training_logger import TrainingLogger, TrainingMetrics


VTRACE_RHO_PERCENTILES = (50, 90, 95, 99)
_MAX_LOG_RHO_TORCH = float(np.log(np.finfo(np.float32).max))


@dataclass(frozen=True, slots=True)
class _StructuredCatalogMetadata:
    family_names: tuple[str, ...]
    attack_type_names: tuple[str, ...]
    family_ids: tuple[int, ...]
    play_slots: tuple[int, ...]
    attack_slots: tuple[int, ...]
    attack_types: tuple[int, ...]
    main_move_02_action_id: int | None


@dataclass(frozen=True, slots=True)
class _PackedStructuredLegalView:
    row_count: int
    row_indices: Tensor
    action_ids: Tensor
    logits: Tensor
    row_log_z: Tensor
    row_has_candidates: Tensor
    family_ids: Tensor
    arg0: Tensor
    arg1: Tensor
    arg2: Tensor


@dataclass(frozen=True, slots=True)
class _ForwardTimeMajorResult:
    values: Tensor
    logits: Tensor | None = None
    packed_logits: Tensor | None = None

    def __iter__(self):
        yield self.logits if self.logits is not None else self.packed_logits
        yield self.values


@lru_cache(maxsize=8)
def _structured_catalog_metadata(action_catalog: ActionCatalog) -> _StructuredCatalogMetadata:
    family_names = tuple(family.name for family in action_catalog.families)
    attack_type_names = tuple(action_catalog.attack_type_names)
    family_index = {name: index for index, name in enumerate(family_names)}
    action_space = int(action_catalog.action_space_size)
    family_ids = np.full((action_space,), -1, dtype=np.int64)
    play_slots = np.full((action_space,), -1, dtype=np.int64)
    attack_slots = np.full((action_space,), -1, dtype=np.int64)
    attack_types = np.full((action_space,), -1, dtype=np.int64)
    main_move_02_action_id: int | None = None
    attack_type_index = {name: index for index, name in enumerate(attack_type_names)}
    for action_id in range(action_space):
        decoded = action_catalog.decode(action_id)
        family_ids[action_id] = int(family_index.get(decoded.family, -1))
        if decoded.family == "main_play_character" and decoded.stage_slot is not None:
            play_slots[action_id] = int(decoded.stage_slot)
        if decoded.family == "attack":
            if decoded.slot is not None:
                attack_slots[action_id] = int(decoded.slot)
            if decoded.attack_type is not None:
                attack_types[action_id] = int(attack_type_index.get(decoded.attack_type, -1))
        if decoded.family == "main_move" and decoded.from_slot == 0 and decoded.to_slot == 2:
            main_move_02_action_id = int(action_id)
    return _StructuredCatalogMetadata(
        family_names=family_names,
        attack_type_names=attack_type_names,
        family_ids=tuple(int(value) for value in family_ids.tolist()),
        play_slots=tuple(int(value) for value in play_slots.tolist()),
        attack_slots=tuple(int(value) for value in attack_slots.tolist()),
        attack_types=tuple(int(value) for value in attack_types.tolist()),
        main_move_02_action_id=main_move_02_action_id,
    )


def _segment_max(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    out = torch.full((int(num_segments),), -torch.inf, dtype=values.dtype, device=values.device)
    if keys.numel() == 0:
        return out
    out.scatter_reduce_(0, keys.to(dtype=torch.long), values, reduce="amax", include_self=True)
    return out


def _segment_logsumexp(values: Tensor, keys: Tensor, num_segments: int) -> Tensor:
    num_segments = int(num_segments)
    max_per = _segment_max(values, keys, num_segments)
    if keys.numel() == 0:
        return max_per
    gathered_max = max_per.index_select(0, keys.to(dtype=torch.long))
    shifted = torch.exp(values - gathered_max)
    sumexp = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
    sumexp.scatter_add_(0, keys.to(dtype=torch.long), shifted)
    valid = torch.isfinite(max_per) & (sumexp > 0)
    out = torch.full((num_segments,), -torch.inf, dtype=values.dtype, device=values.device)
    out[valid] = torch.log(sumexp[valid]) + max_per[valid]
    return out


def _segment_group_sum(
    values: Tensor,
    row_indices: Tensor,
    group_ids: Tensor,
    *,
    row_count: int,
    group_count: int,
) -> Tensor:
    row_count = int(row_count)
    group_count = int(group_count)
    if group_count <= 0:
        return torch.zeros((row_count, 0), dtype=values.dtype, device=values.device)
    out = torch.zeros((row_count * group_count,), dtype=values.dtype, device=values.device)
    if values.numel() == 0:
        return out.view(row_count, group_count)
    valid = (group_ids >= 0) & (group_ids < group_count)
    if not bool(valid.any().item()):
        return out.view(row_count, group_count)
    flat_keys = row_indices[valid].to(dtype=torch.long) * group_count + group_ids[valid].to(dtype=torch.long)
    out.scatter_add_(0, flat_keys, values[valid])
    return out.view(row_count, group_count)


def _packed_structured_legal_view(
    *,
    logits: Tensor | None,
    packed_ids: Tensor | None,
    packed_offsets: Tensor | None,
    packed_meta: Tensor | None,
) -> _PackedStructuredLegalView | None:
    if packed_ids is None or packed_offsets is None or packed_meta is None:
        return None
    if logits is None:
        selected_logits = torch.zeros((int(packed_ids.shape[0]),), device=packed_ids.device, dtype=torch.float32)
        row_count = int(packed_offsets.shape[0] - 1)
        flat_device = packed_ids.device
    elif logits.ndim == 1:
        selected_logits = logits.to(dtype=torch.float32)
        row_count = int(packed_offsets.shape[0] - 1)
        flat_device = selected_logits.device
        if int(selected_logits.shape[0]) != int(packed_ids.shape[0]):
            raise ValueError("packed logits must align 1:1 with packed ids")
    else:
        flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
        row_count = int(flat_logits.shape[0])
        flat_device = flat_logits.device
        lengths = packed_offsets[1:] - packed_offsets[:-1]
        if lengths.ndim != 1 or lengths.numel() != row_count:
            raise ValueError(f"packed legal offsets must describe {row_count} rows")
        row_indices = torch.repeat_interleave(
            torch.arange(row_count, device=flat_device, dtype=torch.long),
            lengths.to(device=flat_device, dtype=torch.long),
        )
        selected_logits = (
            flat_logits[row_indices, packed_ids.to(device=flat_device, dtype=torch.long)]
            if row_indices.numel() > 0
            else flat_logits.new_zeros((0,))
        )
    lengths = packed_offsets[1:] - packed_offsets[:-1]
    if lengths.ndim != 1 or lengths.numel() != row_count:
        raise ValueError(f"packed legal offsets must describe {row_count} rows")
    if packed_meta.ndim != 2 or int(packed_meta.shape[0]) != int(packed_ids.shape[0]) or int(packed_meta.shape[1]) < 4:
        raise ValueError("packed legal metadata must align 1:1 with packed ids and expose 4 fields")
    row_indices = torch.repeat_interleave(
        torch.arange(row_count, device=flat_device, dtype=torch.long),
        lengths.to(device=flat_device, dtype=torch.long),
    )
    meta_long = packed_meta.to(device=flat_device, dtype=torch.long)
    unused = int(np.iinfo(np.uint16).max)

    def _normalize_meta_column(column: Tensor) -> Tensor:
        return torch.where(column == unused, torch.full_like(column, -1), column)

    return _PackedStructuredLegalView(
        row_count=row_count,
        row_indices=row_indices,
        action_ids=packed_ids.to(device=flat_device, dtype=torch.long),
        logits=selected_logits,
        row_log_z=_segment_logsumexp(selected_logits, row_indices, row_count),
        row_has_candidates=lengths.to(device=flat_device, dtype=torch.bool),
        family_ids=_normalize_meta_column(meta_long[:, 0]),
        arg0=_normalize_meta_column(meta_long[:, 1]),
        arg1=_normalize_meta_column(meta_long[:, 2]),
        arg2=_normalize_meta_column(meta_long[:, 3]),
    )


def _packed_group_log_probs(
    packed_view: _PackedStructuredLegalView,
    *,
    group_ids: Tensor,
    group_count: int,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    group_count = int(group_count)
    out = torch.full(
        (packed_view.row_count, max(group_count, 1)),
        -torch.inf,
        dtype=packed_view.logits.dtype,
        device=packed_view.logits.device,
    )[:, :group_count]
    if group_count <= 0 or packed_view.logits.numel() == 0:
        return out
    selected = torch.ones_like(group_ids, dtype=torch.bool) if candidate_mask is None else candidate_mask.to(dtype=torch.bool)
    row_log_z = (
        packed_view.row_log_z
        if candidate_mask is None
        else _segment_logsumexp(
            packed_view.logits[selected],
            packed_view.row_indices[selected],
            packed_view.row_count,
        )
    )
    valid = selected & (group_ids >= 0) & (group_ids < group_count)
    if not bool(valid.any().item()):
        return out
    flat_keys = packed_view.row_indices[valid].to(dtype=torch.long) * group_count + group_ids[valid].to(dtype=torch.long)
    grouped = _segment_logsumexp(packed_view.logits[valid], flat_keys, packed_view.row_count * group_count).view(
        packed_view.row_count,
        group_count,
    )
    finite_rows = torch.isfinite(row_log_z)
    if bool(finite_rows.any().item()):
        out[finite_rows] = grouped[finite_rows] - row_log_z[finite_rows].unsqueeze(1)
    return out


def _nonfinite_indices(values: Tensor | np.ndarray) -> np.ndarray:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    return np.argwhere(~np.isfinite(array)).astype(np.int64, copy=False)


def learner_logp_from_mask(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=pass_action_id)


def learner_logp_from_legal_ids(
    logits: np.ndarray,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    actions: np.ndarray,
    *,
    pass_action_id: int | None = None,
) -> np.ndarray:
    return masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
    )


def summarize_vtrace_diagnostics(
    result: VTraceTargets,
    *,
    rho_bar: float,
    c_bar: float,
) -> dict[str, float]:
    flat_rhos = np.asarray(result.rhos, dtype=np.float64).reshape(-1)
    if flat_rhos.size == 0:
        raise ValueError("result.rhos must not be empty")

    metrics = {
        f"vtrace_rho_p{percentile}": float(np.percentile(flat_rhos, percentile))
        for percentile in VTRACE_RHO_PERCENTILES
    }
    metrics["vtrace_rho_clip_rate"] = float(np.mean(flat_rhos > rho_bar))
    metrics["vtrace_c_clip_rate"] = float(np.mean(flat_rhos > c_bar))
    return metrics


def summarize_structured_policy_metrics(
    logits: Tensor | None,
    legal_mask: Tensor | None,
    *,
    action_catalog: ActionCatalog,
    packed_ids: Tensor | None = None,
    packed_offsets: Tensor | None = None,
    packed_meta: Tensor | None = None,
    packed_view: _PackedStructuredLegalView | None = None,
) -> dict[str, float]:
    packed_view = packed_view if packed_view is not None else _packed_structured_legal_view(
        logits=logits,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    catalog_metadata = _structured_catalog_metadata(action_catalog)
    family_names = catalog_metadata.family_names
    family_index = {name: index for index, name in enumerate(family_names)}
    main_move_02_action_id = catalog_metadata.main_move_02_action_id
    if packed_view is not None and bool(packed_view.row_has_candidates.any().item()):
        row_log_z = packed_view.row_log_z.index_select(0, packed_view.row_indices)
        probs = torch.exp(packed_view.logits - row_log_z)
        top_logits = _segment_max(packed_view.logits, packed_view.row_indices, packed_view.row_count)
        non_empty = packed_view.row_has_candidates
        top1_confidence = torch.exp(top_logits[non_empty] - packed_view.row_log_z[non_empty])
        family_mass = _segment_group_sum(
            probs,
            packed_view.row_indices,
            packed_view.family_ids,
            row_count=packed_view.row_count,
            group_count=len(family_names),
        )
        play_family_id = family_index.get("main_play_character", -1)
        move_family_id = family_index.get("main_move", -1)
        pass_family_id = family_index.get("pass", -1)
        play_mass = (
            family_mass[:, play_family_id]
            if play_family_id >= 0
            else torch.zeros((packed_view.row_count,), dtype=probs.dtype, device=probs.device)
        )
        move_mass = (
            family_mass[:, move_family_id]
            if move_family_id >= 0
            else torch.zeros((packed_view.row_count,), dtype=probs.dtype, device=probs.device)
        )
        pass_mass = (
            family_mass[:, pass_family_id]
            if pass_family_id >= 0
            else torch.zeros((packed_view.row_count,), dtype=probs.dtype, device=probs.device)
        )
        legal_play_available = (
            _segment_group_sum(
                torch.ones_like(probs),
                packed_view.row_indices,
                packed_view.family_ids,
                row_count=packed_view.row_count,
                group_count=len(family_names),
            )[:, play_family_id]
            > 0
            if play_family_id >= 0
            else torch.zeros((packed_view.row_count,), dtype=torch.bool, device=probs.device)
        )

        metrics = {
            "structured_exact_action_concentration": float(top1_confidence.mean().item()),
            "structured_main_play_character_mass": float(play_mass[non_empty].mean().item()),
            "structured_main_move_mass": float(move_mass[non_empty].mean().item()),
            "structured_pass_mass": float(pass_mass[non_empty].mean().item()),
        }
        legal_play_rows = non_empty & legal_play_available
        if bool(legal_play_rows.any().item()):
            metrics["structured_main_move_share_when_play_available"] = float(
                move_mass[legal_play_rows].mean().item()
            )
        if main_move_02_action_id is not None:
            mm_mask = packed_view.action_ids == int(main_move_02_action_id)
            mm_top_rows = torch.zeros((packed_view.row_count,), dtype=torch.bool, device=packed_view.logits.device)
            if bool(mm_mask.any().item()):
                mm_rows = packed_view.row_indices[mm_mask]
                mm_is_top = packed_view.logits[mm_mask] >= top_logits.index_select(0, mm_rows) - 1.0e-6
                mm_top_rows[mm_rows[mm_is_top]] = True
            metrics["structured_main_move_0_2_top1_rate"] = float(mm_top_rows[non_empty].float().mean().item())
        return metrics

    if logits is None or legal_mask is None:
        return {}
    flat_logits = logits.detach().to(dtype=torch.float32).reshape(-1, logits.shape[-1])
    flat_mask = legal_mask.detach().to(dtype=torch.bool).reshape(-1, legal_mask.shape[-1])
    non_empty = flat_mask.any(dim=1)
    if not bool(non_empty.any().item()):
        return {}

    masked_logits = torch.where(
        flat_mask[non_empty],
        flat_logits[non_empty],
        torch.full_like(flat_logits[non_empty], -1.0e9),
    )
    probs = torch.softmax(masked_logits, dim=1)
    top1_ids = probs.argmax(dim=1)
    top1_confidence = probs.max(dim=1).values

    family_ids = torch.as_tensor(catalog_metadata.family_ids, dtype=torch.long, device=flat_logits.device)

    def family_mass(name: str) -> Tensor:
        family_id = family_index.get(name, -1)
        if family_id < 0:
            return torch.zeros((probs.shape[0],), dtype=probs.dtype, device=probs.device)
        mask = family_ids == family_id
        if not bool(mask.any().item()):
            return torch.zeros((probs.shape[0],), dtype=probs.dtype, device=probs.device)
        return probs[:, mask].sum(dim=1)

    play_mass = family_mass("main_play_character")
    move_mass = family_mass("main_move")
    pass_mass = family_mass("pass")
    legal_play_available = flat_mask[non_empty][:, family_ids == family_index.get("main_play_character", -1)].any(dim=1)

    metrics = {
        "structured_exact_action_concentration": float(top1_confidence.mean().item()),
        "structured_main_play_character_mass": float(play_mass.mean().item()),
        "structured_main_move_mass": float(move_mass.mean().item()),
        "structured_pass_mass": float(pass_mass.mean().item()),
    }
    if bool(legal_play_available.any().item()):
        metrics["structured_main_move_share_when_play_available"] = float(
            move_mass[legal_play_available].mean().item()
        )
    if main_move_02_action_id is not None:
        metrics["structured_main_move_0_2_top1_rate"] = float(
            (top1_ids == int(main_move_02_action_id)).to(dtype=torch.float32).mean().item()
        )
    return metrics


def _structured_group_lookup(action_catalog: ActionCatalog, *, device: torch.device) -> dict[str, Any]:
    metadata = _structured_catalog_metadata(action_catalog)
    family_names = metadata.family_names
    family_index = {name: index for index, name in enumerate(family_names)}
    attack_type_names = metadata.attack_type_names

    return {
        "family_ids": torch.as_tensor(metadata.family_ids, dtype=torch.long, device=device),
        "play_slots": torch.as_tensor(metadata.play_slots, dtype=torch.long, device=device),
        "attack_slots": torch.as_tensor(metadata.attack_slots, dtype=torch.long, device=device),
        "attack_types": torch.as_tensor(metadata.attack_types, dtype=torch.long, device=device),
        "family_names": family_names,
        "family_index": family_index,
        "attack_type_names": attack_type_names,
    }


def _group_log_probs(
    *,
    masked_logits: Tensor,
    group_ids: Tensor,
    group_count: int,
) -> Tensor:
    group_scores = torch.full(
        (masked_logits.shape[0], int(group_count)),
        -1.0e9,
        dtype=masked_logits.dtype,
        device=masked_logits.device,
    )
    for group_id in range(int(group_count)):
        group_mask = group_ids == int(group_id)
        if not bool(group_mask.any().item()):
            continue
        group_scores[:, group_id] = torch.logsumexp(
            torch.where(group_mask.unsqueeze(0), masked_logits, torch.full_like(masked_logits, -1.0e9)),
            dim=1,
        )
    row_log_z = torch.logsumexp(masked_logits, dim=1, keepdim=True)
    return group_scores - row_log_z


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    denominator = torch.clamp(weight.sum(), min=1.0)
    return (value * weight).sum() / denominator


def compute_structured_teacher_auxiliary_metrics(
    *,
    logits: Tensor | None,
    legal_mask: Tensor | None,
    teacher_family: Tensor | None,
    teacher_slot: Tensor | None,
    teacher_attack_type: Tensor | None,
    teacher_valid: Tensor | None,
    loss_mask: Tensor,
    action_catalog: ActionCatalog,
    family_coef: float,
    slot_coef: float,
    attack_type_coef: float,
    packed_ids: Tensor | None = None,
    packed_offsets: Tensor | None = None,
    packed_meta: Tensor | None = None,
    packed_view: _PackedStructuredLegalView | None = None,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    zero_source = logits
    if zero_source is None and packed_view is not None:
        zero_source = packed_view.logits
    if zero_source is None:
        zero_source = loss_mask
    zero = zero_source.sum() * 0.0
    value_dtype = zero.dtype
    empty_metrics = {
        "teacher_valid_fraction": 0.0,
        "teacher_family_accuracy": 0.0,
        "teacher_slot_accuracy": 0.0,
        "teacher_attack_type_accuracy": 0.0,
        "teacher_family_loss": 0.0,
        "teacher_slot_loss": 0.0,
        "teacher_attack_type_loss": 0.0,
        "teacher_aux_loss": 0.0,
    }
    if (
        teacher_family is None
        or teacher_slot is None
        or teacher_attack_type is None
        or teacher_valid is None
    ):
        return zero, empty_metrics, {}

    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_family = teacher_family.reshape(-1).to(dtype=torch.long)
    flat_teacher_slot = teacher_slot.reshape(-1).to(dtype=torch.long)
    flat_teacher_attack_type = teacher_attack_type.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
    packed_view = packed_view if packed_view is not None else _packed_structured_legal_view(
        logits=logits,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    if packed_view is not None:
        catalog_metadata = _structured_catalog_metadata(action_catalog)
        family_names = catalog_metadata.family_names
        family_index = {name: index for index, name in enumerate(family_names)}
        attack_type_names = catalog_metadata.attack_type_names
        metrics = dict(empty_metrics)
        metrics["teacher_valid_fraction"] = float(flat_teacher_valid.float().mean().item())
        context: dict[str, Tensor] = {}

        family_loss = zero
        family_rows = packed_view.row_has_candidates & flat_teacher_valid & (flat_teacher_family >= 0)
        family_log_probs = _packed_group_log_probs(
            packed_view,
            group_ids=packed_view.family_ids,
            group_count=len(family_names),
        )
        if bool(family_rows.any().item()):
            valid_targets = flat_teacher_family[family_rows]
            row_weight = flat_loss_mask[family_rows]
            selected_family_log_probs = family_log_probs[family_rows]
            target_log_probs = selected_family_log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
            supported = torch.isfinite(target_log_probs)
            if bool(supported.any().item()):
                valid_targets = valid_targets[supported]
                row_weight = row_weight[supported]
                selected_family_log_probs = selected_family_log_probs[supported]
                family_nll = -target_log_probs[supported]
                family_loss = _weighted_mean(family_nll, row_weight).to(dtype=value_dtype)
                family_predictions = selected_family_log_probs.argmax(dim=1)
                metrics["teacher_family_accuracy"] = float(
                    ((family_predictions == valid_targets).float() * row_weight).sum().item()
                    / max(float(row_weight.sum().item()), 1.0)
                )
                metrics["teacher_family_loss"] = float(family_loss.detach().item())
                context["teacher_family_log_probs"] = selected_family_log_probs.detach()

        slot_loss_terms: list[Tensor] = []
        slot_weight_terms: list[Tensor] = []
        slot_correct = 0.0
        slot_total = 0.0
        play_family_id = int(family_index.get("main_play_character", -1))
        attack_family_id = int(family_index.get("attack", -1))

        play_rows = family_rows & (flat_teacher_family == play_family_id) & (flat_teacher_slot >= 0)
        if play_family_id >= 0 and bool(play_rows.any().item()):
            group_log_probs = _packed_group_log_probs(
                packed_view,
                group_ids=packed_view.arg1,
                group_count=max(int(action_catalog.max_stage), 1),
                candidate_mask=packed_view.family_ids == play_family_id,
            )
            targets = flat_teacher_slot[play_rows]
            row_weight = flat_loss_mask[play_rows]
            selected_group_log_probs = group_log_probs[play_rows]
            target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            supported = torch.isfinite(target_log_probs)
            if bool(supported.any().item()):
                targets = targets[supported]
                row_weight = row_weight[supported]
                selected_group_log_probs = selected_group_log_probs[supported]
                slot_loss_terms.append(-target_log_probs[supported])
                slot_weight_terms.append(row_weight)
                slot_predictions = selected_group_log_probs.argmax(dim=1)
                slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
                slot_total += max(float(row_weight.sum().item()), 0.0)

        attack_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_slot >= 0)
        if attack_family_id >= 0 and bool(attack_rows.any().item()):
            group_log_probs = _packed_group_log_probs(
                packed_view,
                group_ids=packed_view.arg0,
                group_count=max(int(action_catalog.attack_slot_count), 1),
                candidate_mask=packed_view.family_ids == attack_family_id,
            )
            targets = flat_teacher_slot[attack_rows]
            row_weight = flat_loss_mask[attack_rows]
            selected_group_log_probs = group_log_probs[attack_rows]
            target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            supported = torch.isfinite(target_log_probs)
            if bool(supported.any().item()):
                targets = targets[supported]
                row_weight = row_weight[supported]
                selected_group_log_probs = selected_group_log_probs[supported]
                slot_loss_terms.append(-target_log_probs[supported])
                slot_weight_terms.append(row_weight)
                slot_predictions = selected_group_log_probs.argmax(dim=1)
                slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
                slot_total += max(float(row_weight.sum().item()), 0.0)

        slot_loss = zero
        if slot_loss_terms:
            all_slot_losses = torch.cat(slot_loss_terms, dim=0)
            all_slot_weights = torch.cat(slot_weight_terms, dim=0)
            slot_loss = _weighted_mean(all_slot_losses, all_slot_weights).to(dtype=value_dtype)
            metrics["teacher_slot_accuracy"] = float(slot_correct / max(slot_total, 1.0))
            metrics["teacher_slot_loss"] = float(slot_loss.detach().item())

        attack_type_loss = zero
        attack_type_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_attack_type >= 0)
        if attack_family_id >= 0 and bool(attack_type_rows.any().item()) and attack_type_names:
            group_log_probs = _packed_group_log_probs(
                packed_view,
                group_ids=packed_view.arg1,
                group_count=len(attack_type_names),
                candidate_mask=packed_view.family_ids == attack_family_id,
            )
            targets = flat_teacher_attack_type[attack_type_rows]
            row_weight = flat_loss_mask[attack_type_rows]
            selected_group_log_probs = group_log_probs[attack_type_rows]
            target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            supported = torch.isfinite(target_log_probs)
            if bool(supported.any().item()):
                targets = targets[supported]
                row_weight = row_weight[supported]
                selected_group_log_probs = selected_group_log_probs[supported]
                attack_type_nll = -target_log_probs[supported]
                attack_type_loss = _weighted_mean(attack_type_nll, row_weight).to(dtype=value_dtype)
                attack_type_predictions = selected_group_log_probs.argmax(dim=1)
                metrics["teacher_attack_type_accuracy"] = float(
                    ((attack_type_predictions == targets).float() * row_weight).sum().item()
                    / max(float(row_weight.sum().item()), 1.0)
                )
                metrics["teacher_attack_type_loss"] = float(attack_type_loss.detach().item())
                context["teacher_attack_type_log_probs"] = selected_group_log_probs.detach()

        total_aux = (
            family_loss * float(family_coef)
            + slot_loss * float(slot_coef)
            + attack_type_loss * float(attack_type_coef)
        )
        metrics["teacher_aux_loss"] = float(total_aux.detach().item())
        context["teacher_aux_loss"] = total_aux.detach()
        return total_aux.to(dtype=value_dtype), metrics, context

    if logits is None or legal_mask is None:
        return zero, empty_metrics, {}

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_legal_mask = legal_mask.reshape(-1, legal_mask.shape[-1]).to(dtype=torch.bool)
    masked_logits = torch.where(flat_legal_mask, flat_logits, torch.full_like(flat_logits, -1.0e9))

    lookup = _structured_group_lookup(action_catalog, device=masked_logits.device)
    family_ids = lookup["family_ids"]
    play_slots = lookup["play_slots"]
    attack_slots = lookup["attack_slots"]
    attack_types = lookup["attack_types"]
    family_index = lookup["family_index"]
    family_names = lookup["family_names"]
    attack_type_names = lookup["attack_type_names"]

    metrics = dict(empty_metrics)
    metrics["teacher_valid_fraction"] = float(flat_teacher_valid.float().mean().item())
    context: dict[str, Tensor] = {}

    family_loss = zero
    family_rows = flat_teacher_valid & (flat_teacher_family >= 0)
    if bool(family_rows.any().item()):
        family_log_probs = _group_log_probs(
            masked_logits=masked_logits[family_rows],
            group_ids=family_ids,
            group_count=len(family_names),
        )
        valid_targets = flat_teacher_family[family_rows]
        row_weight = flat_loss_mask[family_rows]
        family_nll = -family_log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        family_loss = _weighted_mean(family_nll, row_weight).to(dtype=logits.dtype)
        family_predictions = family_log_probs.argmax(dim=1)
        metrics["teacher_family_accuracy"] = float(
            ((family_predictions == valid_targets).float() * row_weight).sum().item()
            / max(float(row_weight.sum().item()), 1.0)
        )
        metrics["teacher_family_loss"] = float(family_loss.detach().item())
        context["teacher_family_log_probs"] = family_log_probs.detach()

    slot_loss_terms: list[Tensor] = []
    slot_weight_terms: list[Tensor] = []
    slot_correct = 0.0
    slot_total = 0.0
    play_family_id = int(family_index.get("main_play_character", -1))
    attack_family_id = int(family_index.get("attack", -1))

    play_rows = family_rows & (flat_teacher_family == play_family_id) & (flat_teacher_slot >= 0)
    if play_family_id >= 0 and bool(play_rows.any().item()):
        family_logits = masked_logits[play_rows]
        family_mask = flat_legal_mask[play_rows] & (family_ids == play_family_id).unsqueeze(0)
        group_log_probs = _group_log_probs(
            masked_logits=torch.where(family_mask, family_logits, torch.full_like(family_logits, -1.0e9)),
            group_ids=play_slots,
            group_count=max(int(action_catalog.max_stage), 1),
        )
        targets = flat_teacher_slot[play_rows]
        row_weight = flat_loss_mask[play_rows]
        slot_loss_terms.append(-group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        slot_weight_terms.append(row_weight)
        slot_predictions = group_log_probs.argmax(dim=1)
        slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
        slot_total += max(float(row_weight.sum().item()), 0.0)

    attack_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_slot >= 0)
    if attack_family_id >= 0 and bool(attack_rows.any().item()):
        family_logits = masked_logits[attack_rows]
        family_mask = flat_legal_mask[attack_rows] & (family_ids == attack_family_id).unsqueeze(0)
        group_log_probs = _group_log_probs(
            masked_logits=torch.where(family_mask, family_logits, torch.full_like(family_logits, -1.0e9)),
            group_ids=attack_slots,
            group_count=max(int(action_catalog.attack_slot_count), 1),
        )
        targets = flat_teacher_slot[attack_rows]
        row_weight = flat_loss_mask[attack_rows]
        slot_loss_terms.append(-group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        slot_weight_terms.append(row_weight)
        slot_predictions = group_log_probs.argmax(dim=1)
        slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
        slot_total += max(float(row_weight.sum().item()), 0.0)

    slot_loss = zero
    if slot_loss_terms:
        all_slot_losses = torch.cat(slot_loss_terms, dim=0)
        all_slot_weights = torch.cat(slot_weight_terms, dim=0)
        slot_loss = _weighted_mean(all_slot_losses, all_slot_weights).to(dtype=logits.dtype)
        metrics["teacher_slot_accuracy"] = float(slot_correct / max(slot_total, 1.0))
        metrics["teacher_slot_loss"] = float(slot_loss.detach().item())

    attack_type_loss = zero
    attack_type_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_attack_type >= 0)
    if attack_family_id >= 0 and bool(attack_type_rows.any().item()) and attack_type_names:
        family_logits = masked_logits[attack_type_rows]
        family_mask = flat_legal_mask[attack_type_rows] & (family_ids == attack_family_id).unsqueeze(0)
        group_log_probs = _group_log_probs(
            masked_logits=torch.where(family_mask, family_logits, torch.full_like(family_logits, -1.0e9)),
            group_ids=attack_types,
            group_count=len(attack_type_names),
        )
        targets = flat_teacher_attack_type[attack_type_rows]
        row_weight = flat_loss_mask[attack_type_rows]
        attack_type_nll = -group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        attack_type_loss = _weighted_mean(attack_type_nll, row_weight).to(dtype=logits.dtype)
        attack_type_predictions = group_log_probs.argmax(dim=1)
        metrics["teacher_attack_type_accuracy"] = float(
            ((attack_type_predictions == targets).float() * row_weight).sum().item()
            / max(float(row_weight.sum().item()), 1.0)
        )
        metrics["teacher_attack_type_loss"] = float(attack_type_loss.detach().item())
        context["teacher_attack_type_log_probs"] = group_log_probs.detach()

    total_aux = (
        family_loss * float(family_coef)
        + slot_loss * float(slot_coef)
        + attack_type_loss * float(attack_type_coef)
    )
    metrics["teacher_aux_loss"] = float(total_aux.detach().item())
    context["teacher_aux_loss"] = total_aux.detach()
    return total_aux.to(dtype=logits.dtype), metrics, context


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _time_step_legal_actions(legal_actions: LegalActionBatch | None, *, step_index: int, batch_size: int) -> LegalActionBatch | None:
    if legal_actions is None:
        return None
    if legal_actions.mask is not None:
        mask = np.asarray(legal_actions.mask, dtype=np.bool_)
        if mask.ndim != 3 or mask.shape[1] != batch_size:
            raise ValueError("legal mask must have shape (time, batch, action) matching the learner batch")
        if step_index < 0 or step_index >= mask.shape[0]:
            raise ValueError("step_index is outside the legal mask time dimension")
        return LegalActionBatch.from_mask(np.expand_dims(mask[step_index], axis=0))
    if legal_actions.ids is None or legal_actions.offsets is None:
        return None
    ids = np.asarray(legal_actions.ids, dtype=np.uint32)
    offsets = np.asarray(legal_actions.offsets, dtype=np.uint32)
    row_start = int(step_index * batch_size)
    row_stop = int(row_start + batch_size)
    if offsets.ndim != 1 or row_stop + 1 > offsets.size:
        raise ValueError("packed legal offsets must match the learner batch shape")
    start = int(offsets[row_start])
    stop = int(offsets[row_stop])
    slice_ids = np.array(ids[start:stop], copy=True)
    slice_offsets = np.array(offsets[row_start : row_stop + 1] - offsets[row_start], copy=True)
    return LegalActionBatch.from_packed(slice_ids, slice_offsets)


def _compute_vtrace_targets_torch(
    rewards: Tensor,
    values: Tensor,
    discounts: Tensor,
    behavior_logp: Tensor,
    target_logp: Tensor,
    *,
    rho_bar: float,
    c_bar: float,
) -> tuple[Tensor, Tensor, Tensor]:
    log_rhos = torch.clamp(target_logp - behavior_logp, max=_MAX_LOG_RHO_TORCH)
    rhos = torch.exp(log_rhos).clamp(max=torch.finfo(torch.float32).max)
    rho_bar_tensor = torch.full((), float(rho_bar), dtype=rhos.dtype, device=rhos.device)
    c_bar_tensor = torch.full((), float(c_bar), dtype=rhos.dtype, device=rhos.device)
    clipped_rhos = torch.minimum(rho_bar_tensor, rhos)
    clipped_cs = torch.minimum(c_bar_tensor, rhos)

    acc = torch.zeros_like(values[-1])
    vs_minus_v_xs = torch.zeros_like(rewards)
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = clipped_rhos[t] * (rewards[t] + discounts[t] * values[t + 1] - values[t])
        acc = delta + discounts[t] * clipped_cs[t] * acc
        vs_minus_v_xs[t] = acc

    vs = values[:-1] + vs_minus_v_xs
    next_vs = torch.cat((vs[1:], values[-1:].clone()), dim=0)
    pg_advantages = clipped_rhos * (rewards + discounts * next_vs - values[:-1])
    return vs, pg_advantages, rhos


def _masked_log_probs_and_entropy(logits: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (batch, action), got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ValueError("logits and legal_mask shapes must match")

    mask = legal_mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    has_legal = mask.any(dim=1, keepdim=True)
    row_max = masked_logits.max(dim=1, keepdim=True).values
    row_max = torch.where(has_legal, row_max, torch.zeros_like(row_max))

    shifted = torch.where(mask, logits - row_max, torch.full_like(logits, float("-inf")))
    exp_shifted = torch.where(mask, torch.exp(shifted), torch.zeros_like(logits))
    denom = exp_shifted.sum(dim=1, keepdim=True)
    safe_denom = torch.where(has_legal, denom, torch.ones_like(denom))
    log_probs = torch.where(mask, shifted - torch.log(safe_denom), torch.full_like(logits, float("-inf")))

    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    entropy = -(probs * safe_log_probs).sum(dim=1)
    return log_probs, entropy


def _masked_action_logp_and_entropy(
    logits: Tensor,
    legal_mask: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D (time, batch, action), got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ValueError("logits and legal_mask shapes must match")
    if actions.shape != logits.shape[:2]:
        raise ValueError("actions must match logits on time and batch dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_mask = legal_mask.reshape(-1, logits.shape[-1]).to(dtype=torch.bool)
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    action_space = flat_logits.shape[1]

    if bool((flat_actions < 0).any().item()):
        raise ValueError("actions must be >= 0")
    if bool((flat_actions >= action_space).any().item()):
        raise ValueError(f"actions must be < action_space ({action_space})")

    empty_rows = ~flat_mask.any(dim=1)
    row_actions = flat_actions.unsqueeze(1)
    action_is_legal = flat_mask.gather(dim=1, index=row_actions).squeeze(1)
    illegal_rows = (~empty_rows) & (~action_is_legal)
    if bool(illegal_rows.any().item()):
        row_index = int(torch.nonzero(illegal_rows, as_tuple=False)[0].item())
        action = int(flat_actions[row_index].item())
        raise ValueError(f"illegal action {action} for row {row_index}")

    log_probs, entropy = _masked_log_probs_and_entropy(flat_logits, flat_mask)
    selected_logp = log_probs.gather(dim=1, index=row_actions).squeeze(1)

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            raise ValueError("pass_action_id is required when legality contains empty rows")
        if pass_action_id < 0 or pass_action_id >= action_space:
            raise ValueError(f"pass_action_id must be in [0, {action_space})")
        illegal_empty_rows = empty_rows & (flat_actions != int(pass_action_id))
        if bool(illegal_empty_rows.any().item()):
            row_index = int(torch.nonzero(illegal_empty_rows, as_tuple=False)[0].item())
            action = int(flat_actions[row_index].item())
            raise ValueError(
                f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
            )
        selected_logp = torch.where(empty_rows, torch.zeros_like(selected_logp), selected_logp)
        entropy = torch.where(empty_rows, torch.zeros_like(entropy), entropy)

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)


def _packed_action_logp_and_entropy(
    logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D (time, batch, action), got shape {tuple(logits.shape)}")
    if actions.shape != logits.shape[:2]:
        raise ValueError("actions must match logits on time and batch dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1]).to(dtype=torch.float32)
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    action_space = flat_logits.shape[1]
    row_count = flat_logits.shape[0]

    ids = legal_ids.reshape(-1).to(dtype=torch.long, device=flat_logits.device)
    offsets = legal_offsets.reshape(-1).to(dtype=torch.long, device=flat_logits.device)
    if offsets.ndim != 1 or offsets.numel() != row_count + 1:
        raise ValueError(f"legal_offsets must have shape ({row_count + 1},)")
    if int(offsets[0].item()) != 0:
        raise ValueError("legal_offsets must start at 0")
    if int(offsets[-1].item()) != int(ids.numel()):
        raise ValueError("legal_offsets must end at len(legal_ids)")

    widths = offsets[1:] - offsets[:-1]
    if bool((widths < 0).any().item()):
        raise ValueError("legal_offsets must be non-decreasing")
    if bool((flat_actions < 0).any().item()):
        raise ValueError("actions must be >= 0")
    if bool((flat_actions >= action_space).any().item()):
        raise ValueError(f"actions must be < action_space ({action_space})")
    if bool((ids < 0).any().item()) or bool((ids >= action_space).any().item()):
        raise ValueError(f"packed legal ids must be in [0, {action_space})")

    selected_logp = torch.zeros((row_count,), device=flat_logits.device, dtype=flat_logits.dtype)
    entropy = torch.zeros((row_count,), device=flat_logits.device, dtype=flat_logits.dtype)
    empty_rows = widths == 0
    non_empty_rows = torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() > 0:
        non_empty_widths = widths[non_empty_rows]
        row_ids = torch.repeat_interleave(non_empty_rows, non_empty_widths)
        legal_logits = flat_logits[row_ids, ids]
        segment_max = torch.segment_reduce(legal_logits, reduce="max", lengths=non_empty_widths)
        repeated_max = torch.repeat_interleave(segment_max, non_empty_widths)
        shifted = legal_logits - repeated_max
        exp_shifted = torch.exp(shifted)
        segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_widths)
        repeated_sum = torch.repeat_interleave(segment_sum, non_empty_widths)
        log_probs = shifted - torch.log(repeated_sum)
        entropy_terms = -(torch.exp(log_probs) * log_probs)
        entropy_non_empty = torch.segment_reduce(entropy_terms, reduce="sum", lengths=non_empty_widths)
        entropy[non_empty_rows] = entropy_non_empty

        repeated_actions = flat_actions[row_ids]
        matches = ids == repeated_actions
        match_counts = torch.segment_reduce(matches.to(dtype=flat_logits.dtype), reduce="sum", lengths=non_empty_widths)
        illegal_rows = match_counts != 1.0
        if bool(illegal_rows.any().item()):
            bad_row = int(non_empty_rows[torch.nonzero(illegal_rows, as_tuple=False)[0].item()].item())
            bad_action = int(flat_actions[bad_row].item())
            raise ValueError(f"illegal action {bad_action} for row {bad_row}")
        selected_non_empty = torch.segment_reduce(
            torch.where(matches, log_probs, torch.zeros_like(log_probs)),
            reduce="sum",
            lengths=non_empty_widths,
        )
        selected_logp[non_empty_rows] = selected_non_empty

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            raise ValueError("pass_action_id is required when legality contains empty rows")
        if pass_action_id < 0 or pass_action_id >= action_space:
            raise ValueError(f"pass_action_id must be in [0, {action_space})")
        illegal_empty_rows = empty_rows & (flat_actions != int(pass_action_id))
        if bool(illegal_empty_rows.any().item()):
            row_index = int(torch.nonzero(illegal_empty_rows, as_tuple=False)[0].item())
            action = int(flat_actions[row_index].item())
            raise ValueError(
                f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
            )

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)


def _packed_scores_action_logp_and_entropy(
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
) -> tuple[Tensor, Tensor]:
    if packed_logits.ndim != 1:
        raise ValueError("packed_logits must be 1D")
    flat_actions = actions.reshape(-1).to(dtype=torch.long)
    row_count = int(flat_actions.shape[0])
    ids = legal_ids.reshape(-1).to(dtype=torch.long, device=packed_logits.device)
    offsets = legal_offsets.reshape(-1).to(dtype=torch.long, device=packed_logits.device)
    if offsets.ndim != 1 or offsets.numel() != row_count + 1:
        raise ValueError(f"legal_offsets must have shape ({row_count + 1},)")
    if int(offsets[0].item()) != 0:
        raise ValueError("legal_offsets must start at 0")
    if int(offsets[-1].item()) != int(ids.numel()) or int(ids.numel()) != int(packed_logits.numel()):
        raise ValueError("packed logits, ids, and offsets must align exactly")

    widths = offsets[1:] - offsets[:-1]
    if bool((widths < 0).any().item()):
        raise ValueError("legal_offsets must be non-decreasing")

    selected_logp = torch.zeros((row_count,), device=packed_logits.device, dtype=packed_logits.dtype)
    entropy = torch.zeros((row_count,), device=packed_logits.device, dtype=packed_logits.dtype)
    empty_rows = widths == 0
    non_empty_rows = torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() > 0:
        non_empty_widths = widths[non_empty_rows]
        row_ids = torch.repeat_interleave(non_empty_rows, non_empty_widths)
        segment_max = torch.segment_reduce(packed_logits, reduce="max", lengths=non_empty_widths)
        repeated_max = torch.repeat_interleave(segment_max, non_empty_widths)
        shifted = packed_logits - repeated_max
        exp_shifted = torch.exp(shifted)
        segment_sum = torch.segment_reduce(exp_shifted, reduce="sum", lengths=non_empty_widths)
        repeated_sum = torch.repeat_interleave(segment_sum, non_empty_widths)
        log_probs = shifted - torch.log(repeated_sum)
        entropy_terms = -(torch.exp(log_probs) * log_probs)
        entropy_non_empty = torch.segment_reduce(entropy_terms, reduce="sum", lengths=non_empty_widths)
        entropy[non_empty_rows] = entropy_non_empty

        repeated_actions = flat_actions[row_ids]
        matches = ids == repeated_actions
        match_counts = torch.segment_reduce(matches.to(dtype=packed_logits.dtype), reduce="sum", lengths=non_empty_widths)
        illegal_rows = match_counts != 1.0
        if bool(illegal_rows.any().item()):
            bad_row = int(non_empty_rows[torch.nonzero(illegal_rows, as_tuple=False)[0].item()].item())
            bad_action = int(flat_actions[bad_row].item())
            raise ValueError(f"illegal action {bad_action} for row {bad_row}")
        selected_non_empty = torch.segment_reduce(
            torch.where(matches, log_probs, torch.zeros_like(log_probs)),
            reduce="sum",
            lengths=non_empty_widths,
        )
        selected_logp[non_empty_rows] = selected_non_empty

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            raise ValueError("pass_action_id is required when legality contains empty rows")
        illegal_empty_rows = empty_rows & (flat_actions != int(pass_action_id))
        if bool(illegal_empty_rows.any().item()):
            row_index = int(torch.nonzero(illegal_empty_rows, as_tuple=False)[0].item())
            action = int(flat_actions[row_index].item())
            raise ValueError(
                f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
            )

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)


@dataclass(slots=True)
class ImpalaLearner:
    model: nn.Module | None = None
    compiled_model: nn.Module | None = None
    optimizer: Optimizer | None = None
    learning_rate: float = 2e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_norm_clip: float = 40.0
    mixed_precision: bool = False
    checkpoint_dir: Path | None = None
    fault_dir: Path | None = None
    checkpoint_interval_updates: int = 50000
    logs_dir: Path | None = None
    logging_interval_updates: int = 100
    vtrace_rho_bar: float = 2.4
    vtrace_c_bar: float = 1.0
    pass_action_id: int | None = None
    teacher_family_coef: float = 0.0
    teacher_slot_coef: float = 0.0
    teacher_attack_type_coef: float = 0.0
    profile_timers: bool = False
    structured_metrics_mode: str = "full"
    teacher_aux_mode: str = "always"

    update_count: int = field(default=0, init=False)
    policy_version: int = field(default=0, init=False)
    total_samples_processed: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.time, init=False)
    logger: TrainingLogger | None = field(default=None, init=False)
    last_log_time: float = field(default_factory=time.time, init=False)
    last_log_update: int = field(default=0, init=False)
    _amp_enabled: bool = field(default=False, init=False)
    _amp_device_type: str = field(default="cpu", init=False)
    _grad_scaler: torch.amp.GradScaler | None = field(default=None, init=False)
    _active_timing_metrics: dict[str, float] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.logs_dir:
            self.logger = TrainingLogger(self.logs_dir, start_time=self.start_time)
        self.structured_metrics_mode = str(self.structured_metrics_mode).strip().lower()
        self.teacher_aux_mode = str(self.teacher_aux_mode).strip().lower()
        if self.structured_metrics_mode not in {"off", "sampled", "full"}:
            raise ValueError("structured_metrics_mode must be one of: off, sampled, full")
        if self.teacher_aux_mode not in {"off", "warmstart_only", "always"}:
            raise ValueError("teacher_aux_mode must be one of: off, warmstart_only, always")
        self._refresh_acceleration_state()

    def set_entropy_coef(self, value: float) -> None:
        self.entropy_coef = float(value)

    def set_teacher_aux_coefs(
        self,
        *,
        family: float | None = None,
        slot: float | None = None,
        attack_type: float | None = None,
    ) -> None:
        if family is not None:
            self.teacher_family_coef = float(family)
        if slot is not None:
            self.teacher_slot_coef = float(slot)
        if attack_type is not None:
            self.teacher_attack_type_coef = float(attack_type)

    def _record_timing_ms(self, name: str, elapsed_seconds: float) -> None:
        if not self.profile_timers or self._active_timing_metrics is None:
            return
        key = f"timer_{name}_ms"
        self._active_timing_metrics[key] = self._active_timing_metrics.get(key, 0.0) + (float(elapsed_seconds) * 1000.0)

    def _teacher_aux_active(self, *, auxiliary_update: bool) -> bool:
        if self.teacher_aux_mode == "off":
            return False
        if self.teacher_aux_mode == "warmstart_only":
            return bool(auxiliary_update)
        return True

    def _should_emit_structured_metrics(self, *, auxiliary_update: bool) -> bool:
        if self.structured_metrics_mode == "off":
            return False
        if self.structured_metrics_mode == "sampled":
            return (not auxiliary_update) and (int(self.update_count) % 10 == 0)
        return True

    def _refresh_acceleration_state(self) -> None:
        if self.model is None:
            self._amp_enabled = False
            self._amp_device_type = "cpu"
            self._grad_scaler = None
            return
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            self._amp_enabled = False
            self._amp_device_type = "cpu"
            self._grad_scaler = None
            return
        self._amp_device_type = parameter.device.type
        self._amp_enabled = bool(self.mixed_precision and self._amp_device_type == "cuda")
        self._grad_scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if self._amp_enabled
            else None
        )

    def update(self, batch: Any) -> dict[str, float]:
        """Run one learner step when training tensors are present."""
        update_started = time.perf_counter()
        self.update_count += 1
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size

        elapsed = time.time() - self.start_time
        throughput_samples_per_sec = self.total_samples_processed / max(elapsed, 1e-6)
        throughput_updates_per_sec = self.update_count / max(elapsed, 1e-6)

        if self.checkpoint_dir and self.update_count % self.checkpoint_interval_updates == 0:
            self.policy_version += 1
            self._write_checkpoint_metadata()

        metrics: dict[str, float] = {
            "loss": 0.0,
            "throughput_samples_per_sec": throughput_samples_per_sec,
            "throughput_updates_per_sec": throughput_updates_per_sec,
            "entropy_coef": float(self.entropy_coef),
        }
        if self.profile_timers:
            self._active_timing_metrics = {}
        vtrace_result = _batch_value(batch, "vtrace_result")

        has_training_inputs = _batch_value(batch, "obs") is not None
        if has_training_inputs:
            missing = [key for key in ("obs", "actions") if _batch_value(batch, key) is None]
            if not self._has_legal_actions(batch):
                missing.append("legal_actions")
            has_vtrace_targets = isinstance(_batch_value(batch, "vtrace_result"), VTraceTargets)
            has_raw_vtrace_inputs = all(
                _batch_value(batch, key) is not None
                for key in ("rewards", "discounts", "behavior_logp", "behavior_values", "bootstrap_value")
            )
            if not has_vtrace_targets and not has_raw_vtrace_inputs:
                missing.append("vtrace_result_or_raw_inputs")
            if missing:
                missing_fields = ", ".join(missing)
                raise ValueError(
                    "batch must include obs, actions, legality, and either vtrace_result or raw vtrace inputs for learner updates; "
                    f"missing {missing_fields}"
                )
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model to run an optimizer step")

            self.model.train()
            if self.compiled_model is not None:
                self.compiled_model.train()
            loss_started = time.perf_counter()
            with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
                loss, loss_metrics, loss_context = self._loss_and_metrics_with_context(batch)
            self._record_timing_ms("learner_loss_and_metrics", time.perf_counter() - loss_started)
            optimizer = self._optimizer_for_step()
            backward_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_scale_before = None
            if self._grad_scaler is not None:
                loss_scale_before = float(self._grad_scaler.get_scale())
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.unscale_(optimizer)
            else:
                loss.backward()
            self._record_timing_ms("learner_backward", time.perf_counter() - backward_started)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            optimizer_started = time.perf_counter()
            if self._grad_scaler is not None:
                bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
                gradients_finite = not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item())
                if gradients_finite:
                    self._grad_scaler.step(optimizer)
                    self._grad_scaler.update()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    if loss_scale_before is not None:
                        try:
                            self._grad_scaler.update(loss_scale_before * 0.5)
                        except TypeError:
                            self._grad_scaler.update()
                    else:
                        self._grad_scaler.update()
                loss_scale_after = float(self._grad_scaler.get_scale())
                gradient_overflow = (not gradients_finite) or bool(
                    loss_scale_before is not None and loss_scale_after < loss_scale_before
                )
                if gradient_overflow:
                    metrics.update(loss_metrics)
                    metrics["amp_grad_overflow"] = 1.0
                    metrics["loss_scale"] = loss_scale_after
                    metrics["grad_norm"] = float(grad_norm_tensor)
                else:
                    metrics.update(loss_metrics)
                    metrics["amp_grad_overflow"] = 0.0
                    metrics["loss_scale"] = loss_scale_after
                    metrics["grad_norm"] = float(grad_norm_tensor)
            else:
                self._ensure_finite_gradients(batch=batch, context=loss_context, grad_norm=grad_norm)
                optimizer.step()
                metrics.update(loss_metrics)
                metrics["grad_norm"] = float(grad_norm)
            self._record_timing_ms("learner_optimizer", time.perf_counter() - optimizer_started)

        if isinstance(vtrace_result, VTraceTargets):
            rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
            c_bar_value = _batch_value(batch, "vtrace_c_bar")
            rho_bar = self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
            c_bar = self.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
            metrics.update(summarize_vtrace_diagnostics(vtrace_result, rho_bar=rho_bar, c_bar=c_bar))

        if self.logger and self.update_count % self.logging_interval_updates == 0:
            self._log_metrics(metrics, batch)
            self.last_log_time = time.time()
            self.last_log_update = self.update_count

        self._record_timing_ms("learner_total", time.perf_counter() - update_started)
        if self._active_timing_metrics is not None:
            metrics.update(self._active_timing_metrics)
            self._active_timing_metrics = None
        return metrics

    def auxiliary_update(self, batch: Any) -> dict[str, float]:
        """Run one optimizer step using only structured teacher supervision."""
        update_started = time.perf_counter()
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run an auxiliary optimizer step")
        batch_size = self._batch_size(batch)
        self.total_samples_processed += batch_size
        if self.profile_timers:
            self._active_timing_metrics = {}
        self.model.train()
        if self.compiled_model is not None:
            self.compiled_model.train()
        loss_started = time.perf_counter()
        with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
            loss, aux_metrics, aux_context = self._auxiliary_loss_and_metrics(batch)
        self._record_timing_ms("learner_auxiliary_loss_and_metrics", time.perf_counter() - loss_started)
        optimizer = self._optimizer_for_step()
        backward_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.unscale_(optimizer)
        else:
            loss.backward()
        self._record_timing_ms("learner_backward", time.perf_counter() - backward_started)
        grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        optimizer_started = time.perf_counter()
        if self._grad_scaler is not None:
            bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
            gradients_finite = not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item())
            if gradients_finite:
                self._grad_scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
            self._grad_scaler.update()
            metrics = dict(aux_metrics)
            metrics["grad_norm"] = float(grad_norm_tensor)
            metrics["amp_grad_overflow"] = 0.0 if gradients_finite else 1.0
            metrics["loss_scale"] = float(self._grad_scaler.get_scale())
        else:
            self._ensure_finite_gradients(batch=batch, context=aux_context, grad_norm=grad_norm)
            optimizer.step()
            metrics = dict(aux_metrics)
            metrics["grad_norm"] = float(grad_norm)
        self._record_timing_ms("learner_optimizer", time.perf_counter() - optimizer_started)
        self._record_timing_ms("learner_total", time.perf_counter() - update_started)
        if self._active_timing_metrics is not None:
            metrics.update(self._active_timing_metrics)
            self._active_timing_metrics = None
        return metrics

    def _loss_and_metrics(self, batch: Any) -> tuple[Tensor, dict[str, float]]:
        loss, metrics, _ = self._loss_and_metrics_with_context(batch)
        return loss, metrics

    def _auxiliary_loss_and_metrics(self, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute auxiliary losses")
        action_catalog = getattr(self.model, "action_catalog", None)
        if not isinstance(action_catalog, ActionCatalog):
            raise ValueError("structured auxiliary pretraining requires a structured action catalog")
        if not self._teacher_aux_active(auxiliary_update=True):
            zero = self._model_parameter().sum() * 0.0
            return zero, {"loss": 0.0, "policy_train_fraction": 0.0}, {}

        obs = self._require_obs(_batch_value(batch, "obs"))
        forward = self._forward_time_major(
            obs,
            initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
            to_play_seat=_batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
            legal_actions=_batch_value(batch, "legal_actions"),
        )
        logits = forward.logits
        packed_logits = forward.packed_logits
        values = forward.values
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=obs.shape[:2])
        legal_mask = None
        if packed_legal is None:
            if logits is None:
                raise ValueError("dense learner path requires dense logits")
            legal_mask = self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
        emit_structured_metrics = self._should_emit_structured_metrics(auxiliary_update=True)
        packed_view = None
        if packed_legal is not None:
            packed_view_started = time.perf_counter()
            packed_view = _packed_structured_legal_view(
                logits=packed_logits if packed_logits is not None else logits,
                packed_ids=packed_legal[0],
                packed_offsets=packed_legal[1],
                packed_meta=packed_legal[2],
            )
            self._record_timing_ms("learner_packed_view", time.perf_counter() - packed_view_started)
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=values.shape,
            like=values,
        )
        if loss_mask is None:
            loss_mask = torch.ones_like(values)

        teacher_aux_started = time.perf_counter()
        teacher_aux_loss, teacher_metrics, teacher_context = compute_structured_teacher_auxiliary_metrics(
            logits=logits,
            legal_mask=legal_mask,
            teacher_family=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_family"),
                field_name="teacher_family",
                expected_shape=values.shape,
            ),
            teacher_slot=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_slot"),
                field_name="teacher_slot",
                expected_shape=values.shape,
            ),
            teacher_attack_type=self._optional_time_major_index_field(
                _batch_value(batch, "teacher_attack_type"),
                field_name="teacher_attack_type",
                expected_shape=values.shape,
            ),
            teacher_valid=self._optional_time_major_bool_field(
                _batch_value(batch, "teacher_valid"),
                field_name="teacher_valid",
                expected_shape=values.shape,
            ),
            loss_mask=loss_mask,
            action_catalog=action_catalog,
            family_coef=float(self.teacher_family_coef),
            slot_coef=float(self.teacher_slot_coef),
            attack_type_coef=float(self.teacher_attack_type_coef),
            packed_ids=None if packed_legal is None else packed_legal[0],
            packed_offsets=None if packed_legal is None else packed_legal[1],
            packed_meta=None if packed_legal is None else packed_legal[2],
            packed_view=packed_view,
        )
        self._record_timing_ms("learner_teacher_aux", time.perf_counter() - teacher_aux_started)
        context: dict[str, Any] = {
            "auxiliary_loss": teacher_aux_loss.detach(),
            "logits": None if logits is None else logits.detach(),
            "packed_logits": None if packed_logits is None else packed_logits.detach(),
            "values": values.detach(),
            "policy_train_mask": loss_mask.detach(),
            **teacher_context,
        }
        self._ensure_finite_tensor("auxiliary_loss", teacher_aux_loss, batch=batch, context=context)
        metrics = {
            "loss": float(teacher_aux_loss.detach().item()),
            "policy_train_fraction": float(loss_mask.mean().detach().item()),
        }
        metrics.update(teacher_metrics)
        if emit_structured_metrics:
            summary_started = time.perf_counter()
            metrics.update(
                summarize_structured_policy_metrics(
                    logits,
                    legal_mask,
                    action_catalog=action_catalog,
                    packed_ids=None if packed_legal is None else packed_legal[0],
                    packed_offsets=None if packed_legal is None else packed_legal[1],
                    packed_meta=None if packed_legal is None else packed_legal[2],
                    packed_view=packed_view,
                )
            )
            self._record_timing_ms("learner_structured_summary", time.perf_counter() - summary_started)
        return teacher_aux_loss, metrics, context

    def _loss_and_metrics_with_context(self, batch: Any) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to compute losses")

        vtrace_result = _batch_value(batch, "vtrace_result")

        obs = self._require_obs(_batch_value(batch, "obs"))
        actions = self._require_actions(_batch_value(batch, "actions"), expected_shape=obs.shape[:2])
        forward = self._forward_time_major(
            obs,
            initial_hidden_state=_batch_value(batch, "initial_hidden_state"),
            to_play_seat=_batch_value(batch, "to_play_seat"),
            actor=_batch_value(batch, "actor"),
            legal_actions=_batch_value(batch, "legal_actions"),
        )
        logits = forward.logits
        packed_logits = forward.packed_logits
        values = forward.values
        packed_legal = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=obs.shape[:2])
        legal_mask = None
        if packed_legal is None:
            if logits is None:
                raise ValueError("dense learner path requires dense logits")
            legal_mask = self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
            if legal_mask.shape != logits.shape:
                raise ValueError("legal_mask must match learner logits on time, batch, and action dimensions")
        teacher_aux_active = isinstance(getattr(self.model, "action_catalog", None), ActionCatalog) and self._teacher_aux_active(auxiliary_update=False)
        emit_structured_metrics = self._should_emit_structured_metrics(auxiliary_update=False)
        packed_view = None
        if packed_legal is not None and (emit_structured_metrics or teacher_aux_active):
            packed_view_started = time.perf_counter()
            packed_view = _packed_structured_legal_view(
                logits=packed_logits if packed_logits is not None else logits,
                packed_ids=packed_legal[0],
                packed_offsets=packed_legal[1],
                packed_meta=packed_legal[2],
            )
            self._record_timing_ms("learner_packed_view", time.perf_counter() - packed_view_started)

        context: dict[str, Any] = {
            "logits": None if logits is None else logits.detach(),
            "packed_logits": None if packed_logits is None else packed_logits.detach(),
            "values": values.detach(),
        }
        if logits is not None:
            self._ensure_finite_tensor("forward_logits", logits, batch=batch, context=context)
        if packed_logits is not None:
            self._ensure_finite_tensor("forward_packed_logits", packed_logits, batch=batch, context=context)
        self._ensure_finite_tensor("forward_values", values, batch=batch, context=context)

        if packed_legal is not None:
            packed_reductions_started = time.perf_counter()
            packed_ids, packed_offsets, _packed_meta = packed_legal
            if packed_logits is not None:
                action_logp, entropy = _packed_scores_action_logp_and_entropy(
                    packed_logits,
                    packed_ids,
                    packed_offsets,
                    actions,
                    pass_action_id=self.pass_action_id,
                )
            else:
                assert logits is not None
                action_logp, entropy = _packed_action_logp_and_entropy(
                    logits,
                    packed_ids,
                    packed_offsets,
                    actions,
                    pass_action_id=self.pass_action_id,
                )
            self._record_timing_ms("learner_packed_reductions", time.perf_counter() - packed_reductions_started)
        else:
            assert legal_mask is not None
            assert logits is not None
            action_logp, entropy = _masked_action_logp_and_entropy(
                logits,
                legal_mask,
                actions,
                pass_action_id=self.pass_action_id,
            )
        context["action_logp"] = action_logp.detach()
        context["entropy"] = entropy.detach()
        self._ensure_finite_tensor("action_logp", action_logp, batch=batch, context=context)
        self._ensure_finite_tensor("entropy", entropy, batch=batch, context=context)

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        rho_bar = self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value)
        c_bar = self.vtrace_c_bar if c_bar_value is None else float(c_bar_value)
        if isinstance(vtrace_result, VTraceTargets):
            targets = self._float_target(vtrace_result.vs, expected_shape=values.shape, like=values)
            advantages = self._float_target(vtrace_result.pg_advantages, expected_shape=values.shape, like=values)
            rhos_for_metrics = self._float_target(vtrace_result.rhos, expected_shape=values.shape, like=values)
            raw_rewards = _batch_value(batch, "rewards")
            if raw_rewards is None:
                rewards_for_metrics = torch.zeros_like(values)
            else:
                rewards_for_metrics = self._float_target(raw_rewards, expected_shape=values.shape, like=values)
        else:
            rewards = self._float_target(_batch_value(batch, "rewards"), expected_shape=values.shape, like=values)
            discounts = self._float_target(_batch_value(batch, "discounts"), expected_shape=values.shape, like=values)
            behavior_logp = self._float_target(_batch_value(batch, "behavior_logp"), expected_shape=values.shape, like=values)
            behavior_values = self._float_target(_batch_value(batch, "behavior_values"), expected_shape=values.shape, like=values)
            bootstrap_value = self._float_input(_batch_value(batch, "bootstrap_value"))
            if bootstrap_value.ndim != 1 or bootstrap_value.shape[0] != values.shape[1]:
                raise ValueError(f"bootstrap_value must have shape ({values.shape[1]},), got {tuple(bootstrap_value.shape)}")
            full_values = torch.cat([behavior_values, bootstrap_value.unsqueeze(0)], dim=0)
            # Use the current learner policy log-prob for the V-trace target policy.
            # Passing behavior_logp twice silently forces rho=1 and disables off-policy correction.
            targets, advantages, rhos_for_metrics = _compute_vtrace_targets_torch(
                rewards,
                full_values,
                discounts,
                behavior_logp,
                action_logp,
                rho_bar=rho_bar,
                c_bar=c_bar,
            )
            rewards_for_metrics = rewards
        context["targets"] = targets.detach()
        context["advantages"] = advantages.detach()
        context["vtrace_rhos"] = rhos_for_metrics.detach()
        context["rewards"] = rewards_for_metrics.detach()
        loss_mask = self._optional_time_major_loss_mask(
            _batch_value(batch, "policy_train_mask"),
            expected_shape=values.shape,
            like=values,
        )
        if loss_mask is None:
            loss_mask = torch.ones_like(values)
        context["policy_train_mask"] = loss_mask.detach()
        loss_denominator = torch.clamp(loss_mask.sum(), min=1.0)

        policy_loss = -((action_logp * advantages) * loss_mask).sum() / loss_denominator
        value_loss = (((values - targets) ** 2) * loss_mask).sum() / loss_denominator
        entropy_mean = (entropy * loss_mask).sum() / loss_denominator
        total_loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy_mean)

        teacher_metrics: dict[str, float] = {}
        action_catalog = getattr(self.model, "action_catalog", None)
        if teacher_aux_active:
            structured_legal_mask = (
                legal_mask
                if legal_mask is not None
                else (
                    None
                    if packed_legal is not None and packed_legal[2] is not None
                    else self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
                )
            )
            teacher_aux_started = time.perf_counter()
            teacher_aux_loss, teacher_metrics, teacher_context = compute_structured_teacher_auxiliary_metrics(
                logits=logits,
                legal_mask=structured_legal_mask,
                teacher_family=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_family"),
                    field_name="teacher_family",
                    expected_shape=values.shape,
                ),
                teacher_slot=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_slot"),
                    field_name="teacher_slot",
                    expected_shape=values.shape,
                ),
                teacher_attack_type=self._optional_time_major_index_field(
                    _batch_value(batch, "teacher_attack_type"),
                    field_name="teacher_attack_type",
                    expected_shape=values.shape,
                ),
                teacher_valid=self._optional_time_major_bool_field(
                    _batch_value(batch, "teacher_valid"),
                    field_name="teacher_valid",
                    expected_shape=values.shape,
                ),
                loss_mask=loss_mask,
                action_catalog=action_catalog,
                family_coef=float(self.teacher_family_coef),
                slot_coef=float(self.teacher_slot_coef),
                attack_type_coef=float(self.teacher_attack_type_coef),
                packed_ids=None if packed_legal is None else packed_legal[0],
                packed_offsets=None if packed_legal is None else packed_legal[1],
                packed_meta=None if packed_legal is None else packed_legal[2],
                packed_view=packed_view,
            )
            self._record_timing_ms("learner_teacher_aux", time.perf_counter() - teacher_aux_started)
            total_loss = total_loss + teacher_aux_loss
            context.update(teacher_context)

        context["policy_loss"] = policy_loss.detach()
        context["value_loss"] = value_loss.detach()
        context["entropy_mean"] = entropy_mean.detach()
        context["total_loss"] = total_loss.detach()
        self._ensure_finite_tensor("policy_loss", policy_loss, batch=batch, context=context)
        self._ensure_finite_tensor("value_loss", value_loss, batch=batch, context=context)
        self._ensure_finite_tensor("entropy_mean", entropy_mean, batch=batch, context=context)
        self._ensure_finite_tensor("total_loss", total_loss, batch=batch, context=context)

        rho_metrics = rhos_for_metrics.detach().reshape(-1).to(dtype=torch.float32)
        metrics = {
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_mean.detach()),
            "policy_train_fraction": float(loss_mask.mean().detach()),
            "reward_mean": float(rewards_for_metrics.detach().mean().item()),
            "reward_abs_mean": float(rewards_for_metrics.detach().abs().mean().item()),
            "reward_nonzero_fraction": float((rewards_for_metrics.detach() != 0).float().mean().item()),
            "advantage_mean": float(advantages.detach().mean().item()),
            "advantage_abs_mean": float(advantages.detach().abs().mean().item()),
            "target_mean": float(targets.detach().mean().item()),
            "target_abs_mean": float(targets.detach().abs().mean().item()),
            "vtrace_rho_p50": float(torch.quantile(rho_metrics, 0.50).item()),
            "vtrace_rho_p90": float(torch.quantile(rho_metrics, 0.90).item()),
            "vtrace_rho_p95": float(torch.quantile(rho_metrics, 0.95).item()),
            "vtrace_rho_p99": float(torch.quantile(rho_metrics, 0.99).item()),
            "vtrace_rho_clip_rate": float((rhos_for_metrics.detach() > rho_bar).float().mean().item()),
            "vtrace_c_clip_rate": float((rhos_for_metrics.detach() > c_bar).float().mean().item()),
        }
        metrics.update(teacher_metrics)
        if isinstance(action_catalog, ActionCatalog) and emit_structured_metrics:
            structured_legal_mask = (
                legal_mask
                if legal_mask is not None
                else (
                    None
                    if packed_legal is not None and packed_legal[2] is not None
                    else self._resolve_legal_mask(batch, expected_shape=obs.shape[:2], action_dim=logits.shape[-1])
                )
            )
            summary_started = time.perf_counter()
            metrics.update(
                summarize_structured_policy_metrics(
                    logits,
                    structured_legal_mask,
                    action_catalog=action_catalog,
                    packed_ids=None if packed_legal is None else packed_legal[0],
                    packed_offsets=None if packed_legal is None else packed_legal[1],
                    packed_meta=None if packed_legal is None else packed_legal[2],
                    packed_view=packed_view,
                )
            )
            self._record_timing_ms("learner_structured_summary", time.perf_counter() - summary_started)
        return total_loss, metrics, context

    def _optimizer_for_step(self) -> Optimizer:
        if self.optimizer is None:
            if self.model is None:
                raise ValueError("ImpalaLearner requires a model before creating an optimizer")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.optimizer

    def _forward_time_major(
        self,
        obs: Tensor,
        *,
        initial_hidden_state: Any = None,
        to_play_seat: Any = None,
        actor: Any = None,
        legal_actions: LegalActionBatch | None = None,
    ) -> _ForwardTimeMajorResult:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model to run the forward pass")
        forward_model = self.compiled_model if self.compiled_model is not None else self.model
        if obs.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(obs.shape)}")

        expected_shape = obs.shape[:2]
        batch_size = int(obs.shape[1])
        structured_legal_actions = bool(getattr(forward_model, "supports_legal_candidate_scoring", False))
        acting_seat = self._prepare_acting_seat_batch(
            to_play_seat,
            actor=actor,
            expected_shape=expected_shape,
        )
        if structured_legal_actions and legal_actions is not None:
            if legal_actions.ids is None or legal_actions.offsets is None:
                raise ValueError("structured learner updates require packed legal_actions ids/offsets")
            if legal_actions.meta is None:
                raise ValueError("structured learner updates require packed legal_actions metadata")
        sequence_started = time.perf_counter()
        if (
            acting_seat is not None
            and structured_legal_actions
            and legal_actions is not None
            and hasattr(forward_model, "forward_trunk_sequence_seat_aware")
        ):
            trunk_started = time.perf_counter()
            recurrent_flat, state_repr, observation_context, values, _next_hidden = forward_model.forward_trunk_sequence_seat_aware(
                obs,
                acting_seat,
                self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs),
            )
            self._record_timing_ms("learner_trunk", time.perf_counter() - trunk_started)
            scorer_started = time.perf_counter()
            packed_logits = forward_model.score_packed_legal_candidates(
                recurrent_flat,
                obs.reshape(obs.shape[0] * obs.shape[1], obs.shape[2]),
                legal_actions,
                state_repr=state_repr,
                observation_context=observation_context,
            )
            self._record_timing_ms("learner_packed_scorer", time.perf_counter() - scorer_started)
            self._record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
            packed_rows = int(legal_actions.offsets.shape[0] - 1)
            packed_candidates = int(legal_actions.ids.shape[0]) if legal_actions.ids is not None else 0
            metrics = {
                "packed_candidate_count": float(packed_candidates),
                "packed_candidate_rows": float(packed_rows),
                "avg_legal_actions_per_row": float(packed_candidates / max(packed_rows, 1)),
            }
            if self._active_timing_metrics is not None:
                self._active_timing_metrics.update(metrics)
            return _ForwardTimeMajorResult(
                packed_logits=torch.as_tensor(packed_logits),
                values=torch.as_tensor(values),
            )
        if (
            acting_seat is not None
            and hasattr(forward_model, "forward_sequence_seat_aware")
        ):
            logits, values, _next_hidden = forward_model.forward_sequence_seat_aware(
                obs,
                acting_seat,
                self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs),
                legal_actions=legal_actions if structured_legal_actions else None,
            )
            self._record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
            metrics = {}
            if structured_legal_actions and legal_actions is not None and legal_actions.offsets is not None:
                packed_rows = int(legal_actions.offsets.shape[0] - 1)
                packed_candidates = int(legal_actions.ids.shape[0]) if legal_actions.ids is not None else 0
                metrics = {
                    "packed_candidate_count": float(packed_candidates),
                    "packed_candidate_rows": float(packed_rows),
                    "avg_legal_actions_per_row": float(packed_candidates / max(packed_rows, 1)),
                }
            if self._active_timing_metrics is not None:
                self._active_timing_metrics.update(metrics)
            return _ForwardTimeMajorResult(
                logits=torch.as_tensor(logits),
                values=torch.as_tensor(values),
            )
        logits_steps: list[Tensor] = []
        value_steps: list[Tensor] = []

        if acting_seat is None:
            hidden_state = self._prepare_legacy_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
            for step_index, step_obs in enumerate(obs.unbind(dim=0)):
                step_legal_actions = (
                    _time_step_legal_actions(legal_actions, step_index=step_index, batch_size=batch_size)
                    if structured_legal_actions
                    else None
                )
                if step_legal_actions is None:
                    step_logits, step_value, hidden_state = forward_model(step_obs, hidden_state)
                else:
                    step_logits, step_value, hidden_state = forward_model(
                        step_obs,
                        hidden_state,
                        legal_actions=step_legal_actions,
                )
                logits_steps.append(torch.as_tensor(step_logits))
                value_steps.append(torch.as_tensor(step_value))
                hidden_state = torch.as_tensor(hidden_state)
            return _ForwardTimeMajorResult(
                logits=torch.stack(logits_steps, dim=0),
                values=torch.stack(value_steps, dim=0),
            )

        seat_hidden_state = self._prepare_seat_hidden_state(initial_hidden_state, batch_size=batch_size, like=obs)
        for step_index, (step_obs, step_seat) in enumerate(zip(obs.unbind(dim=0), acting_seat.unbind(dim=0), strict=True)):
            step_legal_actions = (
                _time_step_legal_actions(legal_actions, step_index=step_index, batch_size=batch_size)
                if structured_legal_actions
                else None
            )
            if step_legal_actions is None:
                step_logits, step_value, seat_hidden_state = forward_model.forward_seat_aware(
                    step_obs,
                    step_seat,
                    seat_hidden_state,
                )
            else:
                step_logits, step_value, seat_hidden_state = forward_model.forward_seat_aware(
                    step_obs,
                    step_seat,
                    seat_hidden_state,
                    legal_actions=step_legal_actions,
                )
            logits_steps.append(torch.as_tensor(step_logits))
            value_steps.append(torch.as_tensor(step_value))
            seat_hidden_state = torch.as_tensor(seat_hidden_state)
        self._record_timing_ms("learner_forward_time_major", time.perf_counter() - sequence_started)
        return _ForwardTimeMajorResult(
            logits=torch.stack(logits_steps, dim=0),
            values=torch.stack(value_steps, dim=0),
        )

    def _require_obs(self, value: Any) -> Tensor:
        tensor = self._float_input(value)
        if tensor.ndim != 3:
            raise ValueError(f"obs must be 3D (time, batch, observation), got shape {tuple(tensor.shape)}")
        return tensor

    def _require_actions(self, value: Any, *, expected_shape: torch.Size) -> Tensor:
        tensor = self._long_input(value)
        if tensor.shape != expected_shape:
            raise ValueError(f"actions must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _require_legal_mask(self, value: Any, *, expected_shape: torch.Size) -> Tensor:
        tensor = self._bool_input(value)
        if tensor.ndim != 3 or tensor.shape[:2] != expected_shape:
            expected = (int(expected_shape[0]), int(expected_shape[1]), "action")
            raise ValueError(f"legal_mask must have shape {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _has_legal_actions(self, batch: Any) -> bool:
        if _batch_value(batch, "legal_actions") is not None:
            return True
        if _batch_value(batch, "legal_mask") is not None:
            return True
        return _batch_value(batch, "legal_ids") is not None and _batch_value(batch, "legal_offsets") is not None

    def _resolve_legal_mask(self, batch: Any, *, expected_shape: torch.Size, action_dim: int) -> Tensor:
        legal_actions = _batch_value(batch, "legal_actions")
        if isinstance(legal_actions, LegalActionBatch):
            mask = legal_actions.to_mask(
                expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
                action_space=action_dim,
            )
            return torch.as_tensor(mask, dtype=torch.bool, device=self._model_parameter().device)

        legal_mask = _batch_value(batch, "legal_mask")
        if legal_mask is not None:
            return self._require_legal_mask(legal_mask, expected_shape=expected_shape)

        legal_ids = _batch_value(batch, "legal_ids")
        legal_offsets = _batch_value(batch, "legal_offsets")
        if legal_ids is None or legal_offsets is None:
            raise ValueError("batch must include either legal_actions, legal_mask, or legal_ids/legal_offsets")
        mask = LegalActionBatch.from_packed(legal_ids, legal_offsets).to_mask(
            expected_shape=(int(expected_shape[0]), int(expected_shape[1])),
            action_space=action_dim,
        )
        return torch.as_tensor(mask, dtype=torch.bool, device=self._model_parameter().device)

    def _resolve_packed_legal_actions(self, batch: Any, *, expected_shape: torch.Size) -> tuple[Tensor, Tensor] | None:
        resolved = self._resolve_packed_legal_actions_with_meta(batch, expected_shape=expected_shape)
        if resolved is None:
            return None
        return resolved[0], resolved[1]

    def _resolve_packed_legal_actions_with_meta(
        self,
        batch: Any,
        *,
        expected_shape: torch.Size,
    ) -> tuple[Tensor, Tensor, Tensor | None] | None:
        legal_actions = _batch_value(batch, "legal_actions")
        if isinstance(legal_actions, LegalActionBatch) and legal_actions.ids is not None and legal_actions.offsets is not None:
            ids = torch.as_tensor(legal_actions.ids, device=self._model_parameter().device, dtype=torch.long)
            offsets = torch.as_tensor(legal_actions.offsets, device=self._model_parameter().device, dtype=torch.long)
            expected_rows = int(expected_shape[0] * expected_shape[1])
            if offsets.ndim != 1 or offsets.shape[0] != expected_rows + 1:
                raise ValueError(f"packed legal offsets must have shape ({expected_rows + 1},)")
            meta = (
                None
                if legal_actions.meta is None
                else torch.as_tensor(legal_actions.meta, device=self._model_parameter().device, dtype=torch.long)
            )
            if bool(getattr(self.model, "supports_legal_candidate_scoring", False)) and meta is None:
                raise ValueError("structured learner updates require packed legal action metadata")
            return ids, offsets, meta

        legal_ids = _batch_value(batch, "legal_ids")
        legal_offsets = _batch_value(batch, "legal_offsets")
        if legal_ids is None or legal_offsets is None:
            return None
        ids = torch.as_tensor(legal_ids, device=self._model_parameter().device, dtype=torch.long)
        offsets = torch.as_tensor(legal_offsets, device=self._model_parameter().device, dtype=torch.long)
        expected_rows = int(expected_shape[0] * expected_shape[1])
        if offsets.ndim != 1 or offsets.shape[0] != expected_rows + 1:
            raise ValueError(f"packed legal offsets must have shape ({expected_rows + 1},)")
        legal_action_meta = _batch_value(batch, "legal_action_meta")
        meta = (
            None
            if legal_action_meta is None
            else torch.as_tensor(legal_action_meta, device=self._model_parameter().device, dtype=torch.long)
        )
        if bool(getattr(self.model, "supports_legal_candidate_scoring", False)) and meta is None:
            raise ValueError("structured learner updates require packed legal action metadata")
        return ids, offsets, meta

    def _float_target(self, value: Any, *, expected_shape: torch.Size, like: Tensor) -> Tensor:
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.shape != expected_shape:
            raise ValueError(f"target must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _prepare_legacy_hidden_state(self, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.ndim != 2:
            raise ValueError(
                "initial_hidden_state must be 2D (batch, hidden_size) when to_play_seat/actor is absent, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size:
            raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        return tensor

    def _prepare_seat_hidden_state(self, value: Any, *, batch_size: int, like: Tensor) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.ndim != 3:
            raise ValueError(
                "initial_hidden_state must be 3D (batch, seat, hidden_size) when to_play_seat/actor is present, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != batch_size:
            raise ValueError(f"initial_hidden_state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        if tensor.shape[1] != 2:
            raise ValueError(f"initial_hidden_state seat mismatch: expected 2, got {tensor.shape[1]}")
        return tensor

    def _prepare_acting_seat_batch(
        self,
        to_play_seat: Any,
        *,
        actor: Any,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        seat_tensor = self._optional_time_major_seat_field(
            to_play_seat,
            field_name="to_play_seat",
            expected_shape=expected_shape,
        )
        actor_tensor = self._optional_time_major_seat_field(
            actor,
            field_name="actor",
            expected_shape=expected_shape,
        )

        if seat_tensor is None:
            return actor_tensor
        if actor_tensor is None:
            return seat_tensor
        if not torch.equal(seat_tensor, actor_tensor):
            raise ValueError("actor must match to_play_seat when both are provided")
        return seat_tensor

    def _optional_time_major_seat_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f"{field_name} must be integer-valued")
        tensor = tensor.to(dtype=torch.long)
        if tensor.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        if bool(((tensor != 0) & (tensor != 1)).any().item()):
            raise ValueError(f"{field_name} values must be 0 or 1")
        return tensor

    def _optional_time_major_loss_mask(
        self,
        value: Any,
        *,
        expected_shape: torch.Size,
        like: Tensor,
    ) -> Tensor | None:
        if value is None:
            return None
        tensor = self._tensor_on_model_device(value, dtype=like.dtype)
        if tensor.shape != expected_shape:
            raise ValueError(f"policy_train_mask must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor.clamp(min=0.0, max=1.0)

    def _optional_time_major_index_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        if tensor.is_floating_point() or tensor.is_complex():
            raise ValueError(f"{field_name} must be integer-valued")
        tensor = tensor.to(dtype=torch.long)
        if tensor.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _optional_time_major_bool_field(
        self,
        value: Any,
        *,
        field_name: str,
        expected_shape: torch.Size,
    ) -> Tensor | None:
        if value is None:
            return None
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device, dtype=torch.bool)
        if tensor.shape != expected_shape:
            raise ValueError(f"{field_name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _float_input(self, value: Any) -> Tensor:
        reference = self._model_parameter()
        return self._tensor_on_model_device(value, dtype=reference.dtype)

    def _long_input(self, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.long)

    def _bool_input(self, value: Any) -> Tensor:
        return self._tensor_on_model_device(value, dtype=torch.bool)

    def _tensor_on_model_device(self, value: Any, *, dtype: torch.dtype) -> Tensor:
        if value is None:
            raise ValueError("batch field is required")
        reference = self._model_parameter()
        tensor = torch.as_tensor(value, device=reference.device)
        return tensor.to(dtype=dtype)

    def _model_parameter(self) -> Tensor:
        if self.model is None:
            raise ValueError("ImpalaLearner requires a model")
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            raise ValueError("ImpalaLearner model must have at least one parameter")
        return parameter

    def _batch_size(self, batch: Any) -> int:
        for key in ("rewards", "actions", "logits", "obs"):
            value = _batch_value(batch, key)
            if value is not None:
                return int(np.asarray(value).size)
        return 1

    def _fault_dir_path(self) -> Path:
        if self.fault_dir is not None:
            return self.fault_dir
        if self.checkpoint_dir is not None:
            return self.checkpoint_dir / "faults"
        if self.logs_dir is not None:
            return self.logs_dir / "faults"
        return Path("faults")

    def _batch_fault_snapshot(self, batch: Any) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for key in (
            "obs",
            "actions",
            "legal_mask",
            "to_play_seat",
            "actor",
            "initial_hidden_state",
            "vtrace_rho_bar",
            "vtrace_c_bar",
        ):
            value = _batch_value(batch, key)
            if value is not None:
                snapshot[key] = value
        vtrace_result = _batch_value(batch, "vtrace_result")
        if vtrace_result is not None:
            snapshot["vtrace_result"] = vtrace_result
        return snapshot

    def _write_numeric_fault_bundle(self, *, stage: str, batch: Any, context: dict[str, Any]) -> Path:
        return write_fault_bundle(
            fault_dir=self._fault_dir_path(),
            prefix="learner_numeric_fault",
            payload={
                "format": "numeric_fault_bundle",
                "component": "impala_learner",
                "stage": stage,
                "update_count": self.update_count,
                "policy_version": self.policy_version,
                "batch_size": self._batch_size(batch),
                "pass_action_id": self.pass_action_id,
                "batch": self._batch_fault_snapshot(batch),
                "context": context,
            },
        )

    def _ensure_finite_tensor(
        self,
        name: str,
        tensor: Tensor,
        *,
        batch: Any,
        context: dict[str, Any],
    ) -> None:
        if bool(torch.isfinite(tensor).all().item()):
            return
        fault_context = dict(context)
        fault_context[name] = tensor.detach()
        fault_context[f"{name}_nonfinite_indices"] = _nonfinite_indices(tensor)
        fault_path = self._write_numeric_fault_bundle(stage=name, batch=batch, context=fault_context)
        raise RuntimeError(f"non-finite learner {name}; wrote fault bundle to {fault_path}")

    def _collect_nonfinite_gradients(self, grad_norm: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        model = self.model
        if model is None:
            raise ValueError("ImpalaLearner requires a model")

        bad_gradients = {
            name: parameter.grad.detach()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
        }
        grad_norm_tensor = torch.as_tensor(grad_norm)
        return bad_gradients, grad_norm_tensor

    def _ensure_finite_gradients(self, *, batch: Any, context: dict[str, Any], grad_norm: Tensor) -> None:
        bad_gradients, grad_norm_tensor = self._collect_nonfinite_gradients(grad_norm)
        if not bad_gradients and bool(torch.isfinite(grad_norm_tensor).all().item()):
            return

        self._raise_for_nonfinite_gradients(
            batch=batch,
            context=context,
            grad_norm_tensor=grad_norm_tensor,
            bad_gradients=bad_gradients,
        )

    def _raise_for_nonfinite_gradients(
        self,
        *,
        batch: Any,
        context: dict[str, Any],
        grad_norm_tensor: Tensor,
        bad_gradients: dict[str, Tensor],
    ) -> None:

        fault_context = dict(context)
        fault_context["grad_norm"] = grad_norm_tensor.detach()
        fault_context["grad_norm_nonfinite_indices"] = _nonfinite_indices(grad_norm_tensor)
        if bad_gradients:
            fault_context["bad_gradient_names"] = sorted(bad_gradients)
            fault_context["bad_gradients"] = bad_gradients
        fault_path = self._write_numeric_fault_bundle(stage="gradients", batch=batch, context=fault_context)
        raise RuntimeError(f"non-finite learner gradients; wrote fault bundle to {fault_path}")

    def _write_checkpoint_metadata(self) -> None:
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata_path = self.checkpoint_dir / f"checkpoint_metadata_{self.update_count}.json"
        checkpoint_metadata_path.write_text(
            json.dumps(
                {
                    "format": "checkpoint_metadata",
                    "parameters_included": False,
                    "update_count": self.update_count,
                    "policy_version": self.policy_version,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved checkpoint metadata: {checkpoint_metadata_path}")

    def _log_metrics(self, update_metrics: dict[str, float], batch: Any) -> None:
        if not self.logger:
            return

        rho_bar_value = _batch_value(batch, "vtrace_rho_bar")
        c_bar_value = _batch_value(batch, "vtrace_c_bar")
        vtrace_metrics = compute_vtrace_metrics(
            batch,
            rho_bar=self.vtrace_rho_bar if rho_bar_value is None else float(rho_bar_value),
            c_bar=self.vtrace_c_bar if c_bar_value is None else float(c_bar_value),
            pass_action_id=self.pass_action_id,
        )
        elapsed = time.time() - self.start_time
        metrics = TrainingMetrics(
            update_count=self.update_count,
            wall_clock_seconds=elapsed,
            wall_clock_ms=int(elapsed * 1000),
            policy_version=self.policy_version,
            loss=float(update_metrics.get("loss", 0.0)),
            throughput_samples_per_sec=float(update_metrics.get("throughput_samples_per_sec", 0.0)),
            throughput_updates_per_sec=float(update_metrics.get("throughput_updates_per_sec", 0.0)),
            vtrace_rho_mean=vtrace_metrics.rho_mean,
            vtrace_rho_p50=float(update_metrics.get("vtrace_rho_p50", vtrace_metrics.rho_p50)),
            vtrace_rho_p90=float(update_metrics.get("vtrace_rho_p90", vtrace_metrics.rho_p90)),
            vtrace_rho_p99=float(update_metrics.get("vtrace_rho_p99", vtrace_metrics.rho_p99)),
            vtrace_clip_rate=float(update_metrics.get("vtrace_rho_clip_rate", vtrace_metrics.clip_rate)),
            vtrace_c_clipped_rate=float(update_metrics.get("vtrace_c_clip_rate", vtrace_metrics.c_clipped_rate)),
            kl_divergence=vtrace_metrics.kl_divergence,
            value_loss=float(update_metrics.get("value_loss", 0.0)),
            actor_loss=float(update_metrics.get("policy_loss", 0.0)),
            entropy=float(update_metrics.get("entropy", vtrace_metrics.entropy)),
            custom_metrics=self._custom_log_metrics(update_metrics, vtrace_metrics),
        )
        self.logger.log(metrics)

    def _custom_log_metrics(
        self,
        update_metrics: dict[str, float],
        vtrace_metrics: VtraceMetrics,
    ) -> dict[str, float]:
        custom_metrics: dict[str, float] = {
            "vtrace_batch_metrics_available": float(np.isfinite(vtrace_metrics.rho_mean)),
        }
        if "vtrace_rho_p95" in update_metrics:
            custom_metrics["vtrace_rho_p95"] = float(update_metrics["vtrace_rho_p95"])
        if np.isfinite(vtrace_metrics.entropy):
            custom_metrics["vtrace_entropy"] = float(vtrace_metrics.entropy)
        return custom_metrics

    def get_policy_version(self) -> int:
        return self.policy_version
