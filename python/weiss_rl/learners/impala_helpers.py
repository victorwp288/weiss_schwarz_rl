"""Pure IMPALA loss, log-prob, and structured auxiliary helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch import Tensor

from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.learners.vtrace import VTraceTargets
from weiss_rl.legal_actions import LegalActionBatch
from weiss_rl.masking import masked_logp_from_legal_ids, masked_logp_from_mask

VTRACE_RHO_PERCENTILES = (50, 90, 95, 99)
_MAX_LOG_RHO_TORCH = float(np.log(np.finfo(np.float32).max))
_SUPPORTED_PUBLIC_HEURISTIC_PROFILES = frozenset({"base", "aggressive", "control"})
_SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES = frozenset({"mixture", "cycle"})


def _normalize_public_heuristic_profiles(profiles: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_name in profiles or ():
        name = str(raw_name).strip().lower()
        if not name or name in normalized:
            continue
        normalized.append(name)
    if not normalized:
        return ("base",)
    invalid = sorted(set(normalized) - _SUPPORTED_PUBLIC_HEURISTIC_PROFILES)
    if invalid:
        raise ValueError("teacher_public_heuristic_profiles contains unsupported profiles: " + ", ".join(invalid))
    return tuple(normalized)


def _normalize_public_heuristic_profile_mode(mode: str | None) -> str:
    normalized = str(mode or "mixture").strip().lower()
    if normalized not in _SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES:
        raise ValueError(
            "teacher_public_heuristic_profile_mode must be one of: "
            + ", ".join(sorted(_SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES))
        )
    return normalized


@dataclass(frozen=True, slots=True)
class _StructuredCatalogMetadata:
    family_names: tuple[str, ...]
    attack_type_names: tuple[str, ...]
    family_ids: tuple[int, ...]
    play_slots: tuple[int, ...]
    move_from_slots: tuple[int, ...]
    move_to_slots: tuple[int, ...]
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
    observation_context: Mapping[str, Tensor] | None = None

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
    move_from_slots = np.full((action_space,), -1, dtype=np.int64)
    move_to_slots = np.full((action_space,), -1, dtype=np.int64)
    attack_slots = np.full((action_space,), -1, dtype=np.int64)
    attack_types = np.full((action_space,), -1, dtype=np.int64)
    main_move_02_action_id: int | None = None
    attack_type_index = {name: index for index, name in enumerate(attack_type_names)}
    for action_id in range(action_space):
        decoded = action_catalog.decode(action_id)
        family_ids[action_id] = int(family_index.get(decoded.family, -1))
        if decoded.family == "main_play_character" and decoded.stage_slot is not None:
            play_slots[action_id] = int(decoded.stage_slot)
        if decoded.family == "main_move" and decoded.from_slot is not None:
            move_from_slots[action_id] = int(decoded.from_slot)
        if decoded.family == "main_move" and decoded.to_slot is not None:
            move_to_slots[action_id] = int(decoded.to_slot)
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
        move_from_slots=tuple(int(value) for value in move_from_slots.tolist()),
        move_to_slots=tuple(int(value) for value in move_to_slots.tolist()),
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
    shifted = torch.exp(values - gathered_max).to(dtype=values.dtype)
    sumexp = torch.zeros((num_segments,), dtype=values.dtype, device=values.device)
    sumexp.scatter_add_(0, keys.to(dtype=torch.long), shifted)
    valid = torch.isfinite(max_per) & (sumexp > 0)
    out = torch.full((num_segments,), -torch.inf, dtype=values.dtype, device=values.device)
    out[valid] = (torch.log(sumexp[valid]) + max_per[valid]).to(dtype=values.dtype)
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
    selected = (
        torch.ones_like(group_ids, dtype=torch.bool) if candidate_mask is None else candidate_mask.to(dtype=torch.bool)
    )
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
    flat_keys = packed_view.row_indices[valid].to(dtype=torch.long) * group_count + group_ids[valid].to(
        dtype=torch.long
    )
    grouped = _segment_logsumexp(packed_view.logits[valid], flat_keys, packed_view.row_count * group_count).view(
        packed_view.row_count,
        group_count,
    )
    finite_rows = torch.isfinite(row_log_z)
    if bool(finite_rows.any().item()):
        out[finite_rows] = grouped[finite_rows] - row_log_z[finite_rows].unsqueeze(1)
    return out


def _packed_soft_target_cross_entropy(
    packed_view: _PackedStructuredLegalView,
    *,
    target_logits: Tensor,
    temperature: float,
    row_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if temperature <= 0.0:
        raise ValueError("public heuristic temperature must be > 0")
    flat_target_logits = target_logits.reshape(-1).to(device=packed_view.logits.device, dtype=packed_view.logits.dtype)
    if int(flat_target_logits.shape[0]) != int(packed_view.logits.shape[0]):
        raise ValueError("public heuristic target logits must align 1:1 with packed logits")
    row_indices = packed_view.row_indices.to(dtype=torch.long)
    if row_mask is None:
        candidate_mask = torch.ones_like(row_indices, dtype=torch.bool, device=packed_view.logits.device)
    else:
        row_mask = row_mask.reshape(-1).to(device=packed_view.logits.device, dtype=torch.bool)
        if int(row_mask.shape[0]) != int(packed_view.row_count):
            raise ValueError("public heuristic row mask must align 1:1 with packed rows")
        candidate_mask = row_mask.index_select(0, row_indices)
    selected_rows = row_indices[candidate_mask]
    selected_target_logits = flat_target_logits[candidate_mask]
    selected_student_logits = packed_view.logits[candidate_mask]
    selected_student_log_z = packed_view.row_log_z.index_select(0, selected_rows)

    scaled_target_logits = selected_target_logits / float(temperature)
    target_row_log_z = _segment_logsumexp(scaled_target_logits, selected_rows, packed_view.row_count)
    target_log_probs = scaled_target_logits - target_row_log_z.index_select(0, selected_rows)
    target_probs = torch.exp(target_log_probs)
    student_log_probs = selected_student_logits - selected_student_log_z

    row_cross_entropy = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    if selected_rows.numel() > 0:
        row_cross_entropy.scatter_add_(0, selected_rows, -(target_probs * student_log_probs))

    row_target_entropy = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    if selected_rows.numel() > 0:
        row_target_entropy.scatter_add_(0, selected_rows, -(target_probs * target_log_probs))

    student_top_logits = _segment_max(selected_student_logits, selected_rows, packed_view.row_count)
    student_top_mask = selected_student_logits >= (student_top_logits.index_select(0, selected_rows) - 1.0e-6)
    row_student_top_mass = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    if bool(student_top_mask.any().item()):
        row_student_top_mass.scatter_add_(
            0,
            selected_rows[student_top_mask],
            target_probs[student_top_mask],
        )
    return row_cross_entropy, row_student_top_mass, row_target_entropy


def _packed_soft_target_group_probs(
    packed_view: _PackedStructuredLegalView,
    *,
    target_logits: Tensor,
    temperature: float,
    row_mask: Tensor,
    group_ids: Tensor,
    group_count: int,
    candidate_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if temperature <= 0.0:
        raise ValueError("public heuristic temperature must be > 0")
    group_count = int(group_count)
    flat_target_logits = target_logits.reshape(-1).to(device=packed_view.logits.device, dtype=packed_view.logits.dtype)
    if int(flat_target_logits.shape[0]) != int(packed_view.logits.shape[0]):
        raise ValueError("public heuristic target logits must align 1:1 with packed logits")
    row_mask = row_mask.reshape(-1).to(device=packed_view.logits.device, dtype=torch.bool)
    if int(row_mask.shape[0]) != int(packed_view.row_count):
        raise ValueError("public heuristic row mask must align 1:1 with packed rows")
    selected = row_mask.index_select(0, packed_view.row_indices.to(dtype=torch.long))
    if candidate_mask is not None:
        selected = selected & candidate_mask.to(device=packed_view.logits.device, dtype=torch.bool)
    selected = selected & (group_ids.to(device=packed_view.logits.device) >= 0)
    out = torch.zeros(
        (packed_view.row_count, max(group_count, 0)),
        dtype=packed_view.logits.dtype,
        device=packed_view.logits.device,
    )
    row_has = torch.zeros((packed_view.row_count,), dtype=torch.bool, device=packed_view.logits.device)
    if group_count <= 0 or not bool(selected.any().item()):
        return out, row_has

    selected_rows = packed_view.row_indices[selected].to(dtype=torch.long)
    selected_target_logits = flat_target_logits[selected] / float(temperature)
    target_row_log_z = _segment_logsumexp(selected_target_logits, selected_rows, packed_view.row_count)
    finite_rows = torch.isfinite(target_row_log_z)
    if not bool(finite_rows.any().item()):
        return out, row_has
    target_log_probs = selected_target_logits - target_row_log_z.index_select(0, selected_rows)
    target_probs = torch.exp(target_log_probs)
    out = _segment_group_sum(
        target_probs,
        selected_rows,
        group_ids[selected].to(device=packed_view.logits.device, dtype=torch.long),
        row_count=packed_view.row_count,
        group_count=group_count,
    )
    row_has = finite_rows
    return out, row_has


def _soft_group_cross_entropy(group_log_probs: Tensor, target_probs: Tensor) -> Tensor:
    target_probs = target_probs.to(device=group_log_probs.device, dtype=group_log_probs.dtype)
    safe_log_probs = torch.where(target_probs > 0.0, group_log_probs, torch.zeros_like(group_log_probs))
    return -(target_probs * safe_log_probs).sum(dim=1)


def _public_main_move_auxiliary_loss(
    *,
    packed_view: _PackedStructuredLegalView,
    public_heuristic_target_logits: Tensor,
    flat_loss_mask: Tensor,
    flat_teacher_valid: Tensor,
    active_rows: Tensor,
    family_names: tuple[str, ...],
    move_source_log_probs: Tensor | None,
    move_slot_log_probs: Tensor | None,
    temperature: float,
    zero: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    metrics = {
        "teacher_public_heuristic_target_main_play_character_mass": 0.0,
        "teacher_public_heuristic_target_main_move_mass": 0.0,
        "teacher_public_heuristic_target_attack_mass": 0.0,
        "teacher_public_heuristic_target_pass_mass": 0.0,
        "teacher_public_main_move_selected_fraction": 0.0,
        "teacher_public_main_move_source_accuracy": 0.0,
        "teacher_public_main_move_slot_accuracy": 0.0,
        "teacher_public_main_move_source_loss": 0.0,
        "teacher_public_main_move_slot_loss": 0.0,
        "teacher_public_main_move_supported_fraction": 0.0,
        "teacher_public_main_move_loss": 0.0,
    }
    family_index = {name: index for index, name in enumerate(family_names)}
    row_mask = packed_view.row_has_candidates & flat_teacher_valid & active_rows
    if not bool(row_mask.any().item()):
        return zero, metrics

    family_probs, _family_rows = _packed_soft_target_group_probs(
        packed_view,
        target_logits=public_heuristic_target_logits,
        temperature=float(temperature),
        row_mask=row_mask,
        group_ids=packed_view.family_ids,
        group_count=len(family_names),
    )
    row_weights = flat_loss_mask[row_mask]
    if float(row_weights.sum().item()) > 0.0:
        for family_name, metric_name in (
            ("main_play_character", "teacher_public_heuristic_target_main_play_character_mass"),
            ("main_move", "teacher_public_heuristic_target_main_move_mass"),
            ("attack", "teacher_public_heuristic_target_attack_mass"),
            ("pass", "teacher_public_heuristic_target_pass_mass"),
        ):
            family_id = int(family_index.get(family_name, -1))
            if family_id >= 0:
                metrics[metric_name] = float(_weighted_mean(family_probs[row_mask, family_id], row_weights).item())

    move_family_id = int(family_index.get("main_move", -1))
    if move_family_id < 0 or move_source_log_probs is None or move_slot_log_probs is None:
        return zero, metrics

    move_candidate_mask = packed_view.family_ids == move_family_id
    move_counts = _segment_group_sum(
        torch.ones_like(packed_view.logits),
        packed_view.row_indices,
        packed_view.family_ids,
        row_count=packed_view.row_count,
        group_count=len(family_names),
    )[:, move_family_id]
    main_move_mass = family_probs[:, move_family_id]
    move_rows = row_mask & (move_counts > 0.0) & (main_move_mass > 1.0e-6)
    active_total = float(active_rows.float().sum().item())
    if active_total > 0.0:
        metrics["teacher_public_main_move_selected_fraction"] = float(move_rows.float().sum().item() / active_total)
    if not bool(move_rows.any().item()):
        return zero, metrics

    source_group_count = int(move_source_log_probs.shape[-1])
    slot_group_count = int(move_slot_log_probs.shape[-1])
    source_targets, source_has = _packed_soft_target_group_probs(
        packed_view,
        target_logits=public_heuristic_target_logits,
        temperature=float(temperature),
        row_mask=move_rows,
        group_ids=packed_view.arg0,
        group_count=source_group_count,
        candidate_mask=move_candidate_mask,
    )
    slot_targets, slot_has = _packed_soft_target_group_probs(
        packed_view,
        target_logits=public_heuristic_target_logits,
        temperature=float(temperature),
        row_mask=move_rows,
        group_ids=packed_view.arg1,
        group_count=slot_group_count,
        candidate_mask=move_candidate_mask,
    )
    effective_weights = flat_loss_mask * main_move_mass.detach()
    loss_terms: list[Tensor] = []
    weight_terms: list[Tensor] = []
    supported_weight = 0.0
    candidate_weight = float(effective_weights[move_rows].sum().item())

    source_supported = move_rows & source_has
    source_loss = _soft_group_cross_entropy(move_source_log_probs, source_targets)
    source_supported = source_supported & torch.isfinite(source_loss)
    if bool(source_supported.any().item()):
        weights = effective_weights[source_supported]
        losses = source_loss[source_supported]
        loss_terms.append(losses)
        weight_terms.append(weights)
        supported_weight += float(weights.sum().item())
        source_metric_loss = _weighted_mean(losses, weights)
        metrics["teacher_public_main_move_source_loss"] = float(source_metric_loss.detach().item())
        metrics["teacher_public_main_move_source_accuracy"] = float(
            (
                (
                    move_source_log_probs[source_supported].argmax(dim=1)
                    == source_targets[source_supported].argmax(dim=1)
                ).float()
                * weights
            )
            .sum()
            .item()
            / max(float(weights.sum().item()), 1.0)
        )

    slot_supported = move_rows & slot_has
    slot_loss = _soft_group_cross_entropy(move_slot_log_probs, slot_targets)
    slot_supported = slot_supported & torch.isfinite(slot_loss)
    if bool(slot_supported.any().item()):
        weights = effective_weights[slot_supported]
        losses = slot_loss[slot_supported]
        loss_terms.append(losses)
        weight_terms.append(weights)
        supported_weight += float(weights.sum().item())
        slot_metric_loss = _weighted_mean(losses, weights)
        metrics["teacher_public_main_move_slot_loss"] = float(slot_metric_loss.detach().item())
        metrics["teacher_public_main_move_slot_accuracy"] = float(
            (
                (
                    move_slot_log_probs[slot_supported].argmax(dim=1) == slot_targets[slot_supported].argmax(dim=1)
                ).float()
                * weights
            )
            .sum()
            .item()
            / max(float(weights.sum().item()), 1.0)
        )

    if candidate_weight > 0.0:
        metrics["teacher_public_main_move_supported_fraction"] = float(
            supported_weight / max(candidate_weight * 2.0, 1.0e-8)
        )
    if not loss_terms:
        return zero, metrics
    loss = _weighted_mean(torch.cat(loss_terms, dim=0), torch.cat(weight_terms, dim=0)).to(dtype=zero.dtype)
    metrics["teacher_public_main_move_loss"] = float(loss.detach().item())
    return loss, metrics


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
    factorized_family_log_probs: Tensor | None = None,
) -> dict[str, float]:
    catalog_metadata = _structured_catalog_metadata(action_catalog)
    family_names = catalog_metadata.family_names
    family_index = {name: index for index, name in enumerate(family_names)}
    main_move_02_action_id = catalog_metadata.main_move_02_action_id
    if factorized_family_log_probs is not None:
        family_probs = torch.exp(factorized_family_log_probs.detach().to(dtype=torch.float32))
        move_family_id = family_index.get("main_move", -1)
        play_family_id = family_index.get("main_play_character", -1)
        pass_family_id = family_index.get("pass", -1)
        metrics = {
            "structured_exact_action_concentration": float(family_probs.max(dim=-1).values.mean().item()),
            "structured_main_play_character_mass": float(
                family_probs[..., play_family_id].mean().item() if play_family_id >= 0 else 0.0
            ),
            "structured_main_move_mass": float(
                family_probs[..., move_family_id].mean().item() if move_family_id >= 0 else 0.0
            ),
            "structured_pass_mass": float(
                family_probs[..., pass_family_id].mean().item() if pass_family_id >= 0 else 0.0
            ),
        }
        return metrics

    packed_view = (
        packed_view
        if packed_view is not None
        else _packed_structured_legal_view(
            logits=logits,
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
    )
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
            metrics["structured_main_move_share_when_play_available"] = float(move_mass[legal_play_rows].mean().item())
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
        metrics["structured_main_move_share_when_play_available"] = float(move_mass[legal_play_available].mean().item())
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
        "move_to_slots": torch.as_tensor(metadata.move_to_slots, dtype=torch.long, device=device),
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


def _resolve_public_heuristic_family_ids(
    *,
    family_names: tuple[str, ...],
    requested_families: tuple[str, ...],
) -> tuple[int, ...]:
    normalized = tuple(str(name).strip() for name in requested_families if str(name).strip())
    if not normalized:
        return ()
    family_index = {name: index for index, name in enumerate(family_names)}
    missing = sorted({name for name in normalized if name not in family_index})
    if missing:
        raise ValueError("teacher_public_heuristic_families contains unknown action families: " + ", ".join(missing))
    return tuple(int(family_index[name]) for name in normalized)


def compute_structured_teacher_auxiliary_metrics(
    *,
    logits: Tensor | None,
    legal_mask: Tensor | None,
    teacher_family: Tensor | None,
    teacher_slot: Tensor | None,
    teacher_attack_type: Tensor | None,
    teacher_action: Tensor | None,
    teacher_valid: Tensor | None,
    loss_mask: Tensor,
    action_catalog: ActionCatalog,
    family_coef: float,
    slot_coef: float,
    attack_type_coef: float,
    action_coef: float,
    same_family_action_coef: float,
    move_source_coef: float = 0.0,
    public_heuristic_coef: float = 0.0,
    public_main_move_coef: float = 0.0,
    development_pass_suppression_coef: float = 0.0,
    public_heuristic_temperature: float = 32.0,
    public_heuristic_families: tuple[str, ...] = (),
    public_heuristic_target_logits: Tensor | None = None,
    packed_ids: Tensor | None = None,
    packed_offsets: Tensor | None = None,
    packed_meta: Tensor | None = None,
    packed_view: _PackedStructuredLegalView | None = None,
    factorized_family_log_probs: Tensor | None = None,
    factorized_play_slot_log_probs: Tensor | None = None,
    factorized_move_source_log_probs: Tensor | None = None,
    factorized_move_slot_log_probs: Tensor | None = None,
    factorized_attack_slot_log_probs: Tensor | None = None,
    factorized_attack_type_log_probs: Tensor | None = None,
    factorized_top_action_ids: Tensor | None = None,
    factorized_same_family_action_logp: Tensor | None = None,
    factorized_same_family_top_action_ids: Tensor | None = None,
    teacher_move_source: Tensor | None = None,
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    zero_source = logits
    if zero_source is None and packed_view is not None:
        zero_source = packed_view.logits
    if zero_source is None:
        zero_source = loss_mask
    zero = zero_source.sum() * 0.0
    value_dtype = zero.dtype
    empty_metrics = {
        "teacher_active_fraction": 0.0,
        "teacher_valid_fraction": 0.0,
        "teacher_main_play_character_fraction": 0.0,
        "teacher_main_move_fraction": 0.0,
        "teacher_attack_fraction": 0.0,
        "teacher_family_accuracy": 0.0,
        "teacher_slot_accuracy": 0.0,
        "teacher_move_source_accuracy": 0.0,
        "teacher_attack_type_accuracy": 0.0,
        "teacher_action_accuracy": 0.0,
        "teacher_same_family_action_accuracy": 0.0,
        "teacher_same_family_main_play_character_accuracy": 0.0,
        "teacher_same_family_main_move_accuracy": 0.0,
        "teacher_family_loss": 0.0,
        "teacher_slot_loss": 0.0,
        "teacher_move_source_loss": 0.0,
        "teacher_move_source_supported_fraction": 0.0,
        "teacher_attack_type_loss": 0.0,
        "teacher_action_loss": 0.0,
        "teacher_action_supported_fraction": 0.0,
        "teacher_same_family_action_loss": 0.0,
        "teacher_same_family_action_supported_fraction": 0.0,
        "teacher_public_heuristic_loss": 0.0,
        "teacher_public_heuristic_supported_fraction": 0.0,
        "teacher_public_heuristic_selected_fraction": 0.0,
        "teacher_public_heuristic_teacher_valid_coverage": 0.0,
        "teacher_public_heuristic_top1_mass": 0.0,
        "teacher_public_heuristic_target_entropy": 0.0,
        "teacher_public_heuristic_target_main_play_character_mass": 0.0,
        "teacher_public_heuristic_target_main_move_mass": 0.0,
        "teacher_public_heuristic_target_attack_mass": 0.0,
        "teacher_public_heuristic_target_pass_mass": 0.0,
        "teacher_public_main_move_selected_fraction": 0.0,
        "teacher_public_main_move_source_accuracy": 0.0,
        "teacher_public_main_move_slot_accuracy": 0.0,
        "teacher_public_main_move_source_loss": 0.0,
        "teacher_public_main_move_slot_loss": 0.0,
        "teacher_public_main_move_supported_fraction": 0.0,
        "teacher_public_main_move_loss": 0.0,
        "teacher_development_pass_suppression_loss": 0.0,
        "teacher_development_pass_suppression_selected_fraction": 0.0,
        "teacher_development_pass_probability": 0.0,
        "teacher_aux_loss": 0.0,
    }

    def _record_teacher_family_coverage(
        metrics: dict[str, float],
        *,
        active_rows: Tensor,
        flat_teacher_family_local: Tensor,
        flat_teacher_valid_local: Tensor,
        play_family_id_local: int,
        move_family_id_local: int,
        attack_family_id_local: int,
    ) -> None:
        active_total = float(active_rows.float().sum().item())
        metrics["teacher_active_fraction"] = active_total / max(float(active_rows.numel()), 1.0)
        if active_total <= 0.0:
            return
        family_rows_local = active_rows & flat_teacher_valid_local & (flat_teacher_family_local >= 0)
        if play_family_id_local >= 0:
            metrics["teacher_main_play_character_fraction"] = float(
                ((family_rows_local & (flat_teacher_family_local == play_family_id_local)).float().sum().item())
                / active_total
            )
        if move_family_id_local >= 0:
            metrics["teacher_main_move_fraction"] = float(
                ((family_rows_local & (flat_teacher_family_local == move_family_id_local)).float().sum().item())
                / active_total
            )
        if attack_family_id_local >= 0:
            metrics["teacher_attack_fraction"] = float(
                ((family_rows_local & (flat_teacher_family_local == attack_family_id_local)).float().sum().item())
                / active_total
            )

    if teacher_family is None or teacher_slot is None or teacher_attack_type is None or teacher_valid is None:
        return zero, empty_metrics, {}

    if factorized_family_log_probs is not None:
        flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
        flat_teacher_family = teacher_family.reshape(-1).to(dtype=torch.long)
        flat_teacher_slot = teacher_slot.reshape(-1).to(dtype=torch.long)
        flat_teacher_move_source = (
            None if teacher_move_source is None else teacher_move_source.reshape(-1).to(dtype=torch.long)
        )
        flat_teacher_attack_type = teacher_attack_type.reshape(-1).to(dtype=torch.long)
        flat_teacher_action = None if teacher_action is None else teacher_action.reshape(-1).to(dtype=torch.long)
        flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
        family_log_probs = factorized_family_log_probs.reshape(-1, factorized_family_log_probs.shape[-1]).to(
            dtype=value_dtype
        )
        play_slot_log_probs = (
            None
            if factorized_play_slot_log_probs is None
            else factorized_play_slot_log_probs.reshape(-1, factorized_play_slot_log_probs.shape[-1]).to(
                dtype=value_dtype
            )
        )
        move_source_log_probs = (
            None
            if factorized_move_source_log_probs is None
            else factorized_move_source_log_probs.reshape(-1, factorized_move_source_log_probs.shape[-1]).to(
                dtype=value_dtype
            )
        )
        move_slot_log_probs = (
            None
            if factorized_move_slot_log_probs is None
            else factorized_move_slot_log_probs.reshape(-1, factorized_move_slot_log_probs.shape[-1]).to(
                dtype=value_dtype
            )
        )
        attack_slot_log_probs = (
            None
            if factorized_attack_slot_log_probs is None
            else factorized_attack_slot_log_probs.reshape(-1, factorized_attack_slot_log_probs.shape[-1]).to(
                dtype=value_dtype
            )
        )
        attack_type_log_probs = (
            None
            if factorized_attack_type_log_probs is None
            else factorized_attack_type_log_probs.reshape(-1, factorized_attack_type_log_probs.shape[-1]).to(
                dtype=value_dtype
            )
        )
        catalog_metadata = _structured_catalog_metadata(action_catalog)
        family_names = catalog_metadata.family_names
        family_index = {name: index for index, name in enumerate(family_names)}
        public_heuristic_family_ids = _resolve_public_heuristic_family_ids(
            family_names=family_names,
            requested_families=tuple(public_heuristic_families),
        )
        attack_type_names = catalog_metadata.attack_type_names
        metrics = dict(empty_metrics)
        metrics["teacher_valid_fraction"] = float(flat_teacher_valid.float().mean().item())
        context: dict[str, Tensor] = {}
        active_rows = flat_loss_mask > 0.0
        family_rows = active_rows & flat_teacher_valid & (flat_teacher_family >= 0)
        family_loss = zero
        if bool(family_rows.any().item()):
            valid_targets = flat_teacher_family[family_rows]
            row_weight = flat_loss_mask[family_rows]
            selected_family_log_probs = family_log_probs[family_rows]
            target_log_probs = selected_family_log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
            family_loss = _weighted_mean(-target_log_probs, row_weight).to(dtype=value_dtype)
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
        move_family_id = int(family_index.get("main_move", -1))
        attack_family_id = int(family_index.get("attack", -1))
        _record_teacher_family_coverage(
            metrics,
            active_rows=active_rows,
            flat_teacher_family_local=flat_teacher_family,
            flat_teacher_valid_local=flat_teacher_valid,
            play_family_id_local=play_family_id,
            move_family_id_local=move_family_id,
            attack_family_id_local=attack_family_id,
        )
        move_source_targets_by_action = None
        if flat_teacher_move_source is None:
            move_source_targets_by_action = torch.as_tensor(
                catalog_metadata.move_from_slots,
                device=family_log_probs.device,
                dtype=torch.long,
            )
        if play_slot_log_probs is not None and play_family_id >= 0:
            play_rows = family_rows & (flat_teacher_family == play_family_id) & (flat_teacher_slot >= 0)
            if bool(play_rows.any().item()):
                targets = flat_teacher_slot[play_rows]
                row_weight = flat_loss_mask[play_rows]
                selected_group_log_probs = play_slot_log_probs[play_rows]
                target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                slot_loss_terms.append(-target_log_probs)
                slot_weight_terms.append(row_weight)
                slot_predictions = selected_group_log_probs.argmax(dim=1)
                slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
                slot_total += max(float(row_weight.sum().item()), 0.0)
        if move_slot_log_probs is not None and move_family_id >= 0:
            move_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_slot >= 0)
            if bool(move_rows.any().item()):
                targets = flat_teacher_slot[move_rows]
                row_weight = flat_loss_mask[move_rows]
                selected_group_log_probs = move_slot_log_probs[move_rows]
                target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                slot_loss_terms.append(-target_log_probs)
                slot_weight_terms.append(row_weight)
                slot_predictions = selected_group_log_probs.argmax(dim=1)
                slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
                slot_total += max(float(row_weight.sum().item()), 0.0)
        move_source_loss = zero
        if move_source_log_probs is not None and move_family_id >= 0 and float(move_source_coef) != 0.0:
            if flat_teacher_move_source is not None:
                move_source_rows = (
                    active_rows
                    & flat_teacher_valid
                    & (flat_teacher_family == move_family_id)
                    & (flat_teacher_move_source >= 0)
                )
            elif flat_teacher_action is not None:
                move_source_rows = (
                    active_rows
                    & flat_teacher_valid
                    & (flat_teacher_family == move_family_id)
                    & (flat_teacher_action >= 0)
                )
            else:
                move_source_rows = None
            if move_source_rows is not None and bool(move_source_rows.any().item()):
                if flat_teacher_move_source is not None:
                    move_source_targets = flat_teacher_move_source[move_source_rows]
                else:
                    assert move_source_targets_by_action is not None
                    assert flat_teacher_action is not None
                    move_source_targets = move_source_targets_by_action.index_select(
                        0,
                        flat_teacher_action[move_source_rows],
                    )
                valid_targets = move_source_targets >= 0
                if bool(valid_targets.any().item()):
                    row_weight = flat_loss_mask[move_source_rows][valid_targets]
                    selected_group_log_probs = move_source_log_probs[move_source_rows][valid_targets]
                    move_source_targets = move_source_targets[valid_targets]
                    target_log_probs = selected_group_log_probs.gather(1, move_source_targets.unsqueeze(1)).squeeze(1)
                    supported = torch.isfinite(target_log_probs)
                    if float(row_weight.sum().item()) > 0.0:
                        metrics["teacher_move_source_supported_fraction"] = float(
                            (row_weight[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                        )
                    if bool(supported.any().item()):
                        row_weight = row_weight[supported]
                        move_source_targets = move_source_targets[supported]
                        selected_group_log_probs = selected_group_log_probs[supported]
                        target_log_probs = target_log_probs[supported]
                        move_source_loss = _weighted_mean(-target_log_probs, row_weight).to(dtype=value_dtype)
                        move_source_predictions = selected_group_log_probs.argmax(dim=1)
                        metrics["teacher_move_source_accuracy"] = float(
                            ((move_source_predictions == move_source_targets).float() * row_weight).sum().item()
                            / max(float(row_weight.sum().item()), 1.0)
                        )
                        metrics["teacher_move_source_loss"] = float(move_source_loss.detach().item())
        if attack_slot_log_probs is not None and attack_family_id >= 0:
            attack_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_slot >= 0)
            if bool(attack_rows.any().item()):
                targets = flat_teacher_slot[attack_rows]
                row_weight = flat_loss_mask[attack_rows]
                selected_group_log_probs = attack_slot_log_probs[attack_rows]
                target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                slot_loss_terms.append(-target_log_probs)
                slot_weight_terms.append(row_weight)
                slot_predictions = selected_group_log_probs.argmax(dim=1)
                slot_correct += float(((slot_predictions == targets).float() * row_weight).sum().item())
                slot_total += max(float(row_weight.sum().item()), 0.0)
        slot_loss = zero
        if slot_loss_terms:
            slot_loss = _weighted_mean(torch.cat(slot_loss_terms, dim=0), torch.cat(slot_weight_terms, dim=0)).to(
                dtype=value_dtype
            )
            metrics["teacher_slot_accuracy"] = float(slot_correct / max(slot_total, 1.0))
            metrics["teacher_slot_loss"] = float(slot_loss.detach().item())
        attack_type_loss = zero
        if attack_type_log_probs is not None and attack_family_id >= 0 and attack_type_names:
            attack_rows = family_rows & (flat_teacher_family == attack_family_id) & (flat_teacher_attack_type >= 0)
            if bool(attack_rows.any().item()):
                targets = flat_teacher_attack_type[attack_rows]
                row_weight = flat_loss_mask[attack_rows]
                selected_group_log_probs = attack_type_log_probs[attack_rows]
                target_log_probs = selected_group_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                attack_type_loss = _weighted_mean(-target_log_probs, row_weight).to(dtype=value_dtype)
                attack_type_predictions = selected_group_log_probs.argmax(dim=1)
                metrics["teacher_attack_type_accuracy"] = float(
                    ((attack_type_predictions == targets).float() * row_weight).sum().item()
                    / max(float(row_weight.sum().item()), 1.0)
                )
                metrics["teacher_attack_type_loss"] = float(attack_type_loss.detach().item())
                context["teacher_attack_type_log_probs"] = selected_group_log_probs.detach()
        action_loss = zero
        if (
            flat_teacher_action is not None
            and float(action_coef) != 0.0
            and factorized_same_family_action_logp is not None
        ):
            action_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
            if bool(action_rows.any().item()):
                teacher_family_log_probs = family_log_probs.gather(
                    1,
                    torch.clamp(flat_teacher_family, min=0).unsqueeze(1),
                ).squeeze(1)
                teacher_action_log_probs = teacher_family_log_probs + factorized_same_family_action_logp.reshape(-1).to(
                    dtype=value_dtype
                )
                row_weight = flat_loss_mask[action_rows]
                supported = action_rows & torch.isfinite(teacher_action_log_probs)
                if float(row_weight.sum().item()) > 0.0:
                    metrics["teacher_action_supported_fraction"] = float(
                        (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                    )
                if bool(supported.any().item()):
                    supported_log_probs = teacher_action_log_probs[supported]
                    supported_weights = flat_loss_mask[supported]
                    action_loss = _weighted_mean(-supported_log_probs, supported_weights).to(dtype=value_dtype)
                    if factorized_top_action_ids is not None:
                        supported_predictions = factorized_top_action_ids.reshape(-1).to(dtype=torch.long)[supported]
                        supported_targets = flat_teacher_action[supported]
                        metrics["teacher_action_accuracy"] = float(
                            ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                            / max(float(supported_weights.sum().item()), 1.0)
                        )
                    metrics["teacher_action_loss"] = float(action_loss.detach().item())
                    context["teacher_action_log_probs"] = supported_log_probs.detach()
        same_family_action_loss = zero
        if (
            flat_teacher_action is not None
            and float(same_family_action_coef) != 0.0
            and factorized_same_family_action_logp is not None
            and factorized_same_family_top_action_ids is not None
        ):
            same_family_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
            if bool(same_family_rows.any().item()):
                same_family_log_probs = factorized_same_family_action_logp.reshape(-1).to(dtype=value_dtype)
                same_family_top_actions = factorized_same_family_top_action_ids.reshape(-1).to(dtype=torch.long)
                row_weight = flat_loss_mask[same_family_rows]
                supported = same_family_rows & torch.isfinite(same_family_log_probs)
                if float(row_weight.sum().item()) > 0.0:
                    metrics["teacher_same_family_action_supported_fraction"] = float(
                        (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                    )
                if bool(supported.any().item()):
                    supported_log_probs = same_family_log_probs[supported]
                    supported_weights = flat_loss_mask[supported]
                    supported_predictions = same_family_top_actions[supported]
                    supported_targets = flat_teacher_action[supported]
                    same_family_action_loss = _weighted_mean(-supported_log_probs, supported_weights).to(
                        dtype=value_dtype
                    )
                    metrics["teacher_same_family_action_accuracy"] = float(
                        ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                        / max(float(supported_weights.sum().item()), 1.0)
                    )
                    metrics["teacher_same_family_action_loss"] = float(same_family_action_loss.detach().item())
                    context["teacher_same_family_action_log_probs"] = supported_log_probs.detach()
                    supported_families = flat_teacher_family[supported]
                    main_play_supported = supported_families == play_family_id
                    if bool(main_play_supported.any().item()):
                        play_weights = supported_weights[main_play_supported]
                        metrics["teacher_same_family_main_play_character_accuracy"] = float(
                            (
                                (
                                    supported_predictions[main_play_supported] == supported_targets[main_play_supported]
                                ).float()
                                * play_weights
                            )
                            .sum()
                            .item()
                            / max(float(play_weights.sum().item()), 1.0)
                        )
                    main_move_supported = supported_families == move_family_id
                    if bool(main_move_supported.any().item()):
                        move_weights = supported_weights[main_move_supported]
                        metrics["teacher_same_family_main_move_accuracy"] = float(
                            (
                                (
                                    supported_predictions[main_move_supported] == supported_targets[main_move_supported]
                                ).float()
                                * move_weights
                            )
                            .sum()
                            .item()
                            / max(float(move_weights.sum().item()), 1.0)
                        )
        public_heuristic_loss = zero
        if (
            packed_view is not None
            and public_heuristic_target_logits is not None
            and float(public_heuristic_coef) != 0.0
        ):
            public_candidate_rows = packed_view.row_has_candidates & flat_teacher_valid & active_rows
            public_rows = public_candidate_rows
            if public_heuristic_family_ids:
                public_rows = public_rows & torch.isin(
                    flat_teacher_family,
                    torch.as_tensor(
                        public_heuristic_family_ids,
                        device=flat_teacher_family.device,
                        dtype=flat_teacher_family.dtype,
                    ),
                )
            active_total = float(active_rows.float().sum().item())
            valid_total = float(public_candidate_rows.float().sum().item())
            if active_total > 0.0:
                metrics["teacher_public_heuristic_selected_fraction"] = float(
                    public_rows.float().sum().item() / active_total
                )
            if valid_total > 0.0:
                metrics["teacher_public_heuristic_teacher_valid_coverage"] = float(
                    public_rows.float().sum().item() / valid_total
                )
            if bool(public_rows.any().item()):
                row_cross_entropy, row_student_top_mass, row_target_entropy = _packed_soft_target_cross_entropy(
                    packed_view,
                    target_logits=public_heuristic_target_logits,
                    temperature=float(public_heuristic_temperature),
                    row_mask=public_rows,
                )
                public_weights = flat_loss_mask[public_rows]
                if float(public_weights.sum().item()) > 0.0:
                    metrics["teacher_public_heuristic_supported_fraction"] = 1.0
                    metrics["teacher_public_heuristic_top1_mass"] = float(
                        _weighted_mean(row_student_top_mass[public_rows], public_weights).item()
                    )
                    metrics["teacher_public_heuristic_target_entropy"] = float(
                        _weighted_mean(row_target_entropy[public_rows], public_weights).item()
                    )
                    public_heuristic_loss = _weighted_mean(
                        row_cross_entropy[public_rows],
                        public_weights,
                    ).to(dtype=value_dtype)
                    metrics["teacher_public_heuristic_loss"] = float(public_heuristic_loss.detach().item())

        public_main_move_loss = zero
        if packed_view is not None and public_heuristic_target_logits is not None:
            public_main_move_loss, public_main_move_metrics = _public_main_move_auxiliary_loss(
                packed_view=packed_view,
                public_heuristic_target_logits=public_heuristic_target_logits,
                flat_loss_mask=flat_loss_mask,
                flat_teacher_valid=flat_teacher_valid,
                active_rows=active_rows,
                family_names=family_names,
                move_source_log_probs=move_source_log_probs,
                move_slot_log_probs=move_slot_log_probs,
                temperature=float(public_heuristic_temperature),
                zero=zero,
            )
            metrics.update(public_main_move_metrics)

        development_pass_suppression_loss = zero
        if float(development_pass_suppression_coef) != 0.0:
            pass_family_id = int(family_index.get("pass", -1))
            development_rows = (
                active_rows
                & flat_teacher_valid
                & ((flat_teacher_family == play_family_id) | (flat_teacher_family == move_family_id))
            )
            if pass_family_id >= 0 and bool(development_rows.any().item()):
                pass_log_probs = family_log_probs[development_rows, pass_family_id]
                supported = torch.isfinite(pass_log_probs)
                if bool(supported.any().item()):
                    row_weight = flat_loss_mask[development_rows][supported]
                    pass_probs = torch.exp(pass_log_probs[supported]).clamp(max=1.0 - 1.0e-6)
                    development_pass_suppression_loss = _weighted_mean(
                        -torch.log1p(-pass_probs),
                        row_weight,
                    ).to(dtype=value_dtype)
                    active_total = float(active_rows.float().sum().item())
                    if active_total > 0.0:
                        metrics["teacher_development_pass_suppression_selected_fraction"] = float(
                            development_rows.float().sum().item() / active_total
                        )
                    metrics["teacher_development_pass_probability"] = float(
                        _weighted_mean(pass_probs.detach(), row_weight).item()
                    )
                    metrics["teacher_development_pass_suppression_loss"] = float(
                        development_pass_suppression_loss.detach().item()
                    )

        total_aux = (
            family_loss * float(family_coef)
            + slot_loss * float(slot_coef)
            + move_source_loss * float(move_source_coef)
            + attack_type_loss * float(attack_type_coef)
            + action_loss * float(action_coef)
            + same_family_action_loss * float(same_family_action_coef)
            + public_heuristic_loss * float(public_heuristic_coef)
            + public_main_move_loss * float(public_main_move_coef)
            + development_pass_suppression_loss * float(development_pass_suppression_coef)
        )
        metrics["teacher_aux_loss"] = float(total_aux.detach().item())
        return total_aux, metrics, context

    flat_loss_mask = loss_mask.reshape(-1).to(dtype=torch.float32)
    flat_teacher_family = teacher_family.reshape(-1).to(dtype=torch.long)
    flat_teacher_slot = teacher_slot.reshape(-1).to(dtype=torch.long)
    flat_teacher_move_source = (
        None if teacher_move_source is None else teacher_move_source.reshape(-1).to(dtype=torch.long)
    )
    flat_teacher_attack_type = teacher_attack_type.reshape(-1).to(dtype=torch.long)
    flat_teacher_action = None if teacher_action is None else teacher_action.reshape(-1).to(dtype=torch.long)
    flat_teacher_valid = teacher_valid.reshape(-1).to(dtype=torch.bool)
    packed_view = (
        packed_view
        if packed_view is not None
        else _packed_structured_legal_view(
            logits=logits,
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
    )
    if packed_view is not None:
        catalog_metadata = _structured_catalog_metadata(action_catalog)
        family_names = catalog_metadata.family_names
        family_index = {name: index for index, name in enumerate(family_names)}
        public_heuristic_family_ids = _resolve_public_heuristic_family_ids(
            family_names=family_names,
            requested_families=tuple(public_heuristic_families),
        )
        attack_type_names = catalog_metadata.attack_type_names
        metrics = dict(empty_metrics)
        metrics["teacher_valid_fraction"] = float(flat_teacher_valid.float().mean().item())
        context: dict[str, Tensor] = {}
        active_rows = flat_loss_mask > 0.0

        family_loss = zero
        family_rows = packed_view.row_has_candidates & active_rows & flat_teacher_valid & (flat_teacher_family >= 0)
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
        move_family_id = int(family_index.get("main_move", -1))
        attack_family_id = int(family_index.get("attack", -1))
        _record_teacher_family_coverage(
            metrics,
            active_rows=active_rows,
            flat_teacher_family_local=flat_teacher_family,
            flat_teacher_valid_local=flat_teacher_valid,
            play_family_id_local=play_family_id,
            move_family_id_local=move_family_id,
            attack_family_id_local=attack_family_id,
        )

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

        move_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_slot >= 0)
        if move_family_id >= 0 and bool(move_rows.any().item()):
            group_log_probs = _packed_group_log_probs(
                packed_view,
                group_ids=packed_view.arg1,
                group_count=max(int(action_catalog.max_stage), 1),
                candidate_mask=packed_view.family_ids == move_family_id,
            )
            targets = flat_teacher_slot[move_rows]
            row_weight = flat_loss_mask[move_rows]
            selected_group_log_probs = group_log_probs[move_rows]
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
        move_source_loss = zero
        if move_family_id >= 0 and float(move_source_coef) != 0.0:
            if flat_teacher_move_source is not None:
                move_source_rows = (
                    family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_move_source >= 0)
                )
            elif flat_teacher_action is not None:
                move_source_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_action >= 0)
            else:
                move_source_rows = None
            if move_source_rows is not None and not bool(move_source_rows.any().item()):
                move_source_rows = None
        else:
            move_source_rows = None
        if move_source_rows is not None:
            group_log_probs = _packed_group_log_probs(
                packed_view,
                group_ids=packed_view.arg0,
                group_count=max(int(action_catalog.max_stage), 1),
                candidate_mask=packed_view.family_ids == move_family_id,
            )
            if flat_teacher_move_source is not None:
                move_source_targets = flat_teacher_move_source[move_source_rows]
            else:
                move_source_targets = move_source_targets_by_action.index_select(
                    0, flat_teacher_action[move_source_rows]
                )
            valid_targets = move_source_targets >= 0
            if bool(valid_targets.any().item()):
                row_weight = flat_loss_mask[move_source_rows][valid_targets]
                selected_group_log_probs = group_log_probs[move_source_rows][valid_targets]
                move_source_targets = move_source_targets[valid_targets]
                target_log_probs = selected_group_log_probs.gather(1, move_source_targets.unsqueeze(1)).squeeze(1)
                supported = torch.isfinite(target_log_probs)
                if float(row_weight.sum().item()) > 0.0:
                    metrics["teacher_move_source_supported_fraction"] = float(
                        (row_weight[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                    )
                if bool(supported.any().item()):
                    row_weight = row_weight[supported]
                    move_source_targets = move_source_targets[supported]
                    selected_group_log_probs = selected_group_log_probs[supported]
                    target_log_probs = target_log_probs[supported]
                    move_source_loss = _weighted_mean(-target_log_probs, row_weight).to(dtype=value_dtype)
                    move_source_predictions = selected_group_log_probs.argmax(dim=1)
                    metrics["teacher_move_source_accuracy"] = float(
                        ((move_source_predictions == move_source_targets).float() * row_weight).sum().item()
                        / max(float(row_weight.sum().item()), 1.0)
                    )
                    metrics["teacher_move_source_loss"] = float(move_source_loss.detach().item())

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

        action_loss = zero
        if flat_teacher_action is not None and float(action_coef) != 0.0:
            action_rows = packed_view.row_has_candidates & flat_teacher_valid & (flat_teacher_action >= 0)
            if bool(action_rows.any().item()):
                teacher_action_log_probs = (
                    _packed_selected_action_logp(
                        packed_view.logits,
                        packed_view.action_ids,
                        packed_offsets
                        if packed_offsets is not None
                        else packed_ids.new_zeros((packed_view.row_count + 1,)),
                        flat_teacher_action,
                        pass_action_id=int(action_catalog.pass_action_id),
                        strict=False,
                    )
                    .reshape(-1)
                    .to(dtype=value_dtype)
                )
                supported = action_rows & torch.isfinite(teacher_action_log_probs)
                row_weight = flat_loss_mask[action_rows]
                if float(row_weight.sum().item()) > 0.0:
                    metrics["teacher_action_supported_fraction"] = float(
                        (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                    )
                if bool(supported.any().item()):
                    supported_log_probs = teacher_action_log_probs[supported]
                    supported_weights = flat_loss_mask[supported]
                    action_loss = _weighted_mean(-supported_log_probs, supported_weights).to(dtype=value_dtype)
                    top_logits = _segment_max(packed_view.logits, packed_view.row_indices, packed_view.row_count)
                    top_matches = packed_view.logits >= (top_logits.index_select(0, packed_view.row_indices) - 1.0e-6)
                    top_action_ids = torch.full(
                        (packed_view.row_count,),
                        -1,
                        dtype=torch.long,
                        device=packed_view.logits.device,
                    )
                    top_action_ids.scatter_reduce_(
                        0,
                        packed_view.row_indices.to(dtype=torch.long),
                        torch.where(
                            top_matches,
                            packed_view.action_ids.to(dtype=torch.long),
                            torch.full_like(packed_view.action_ids, -1),
                        ),
                        reduce="amax",
                        include_self=True,
                    )
                    metrics["teacher_action_accuracy"] = float(
                        ((top_action_ids[supported] == flat_teacher_action[supported]).float() * supported_weights)
                        .sum()
                        .item()
                        / max(float(supported_weights.sum().item()), 1.0)
                    )
                    metrics["teacher_action_loss"] = float(action_loss.detach().item())
                    context["teacher_action_log_probs"] = supported_log_probs.detach()

        same_family_action_loss = zero
        if flat_teacher_action is not None and float(same_family_action_coef) != 0.0:
            same_family_rows = (
                packed_view.row_has_candidates
                & flat_teacher_valid
                & (flat_teacher_action >= 0)
                & (flat_teacher_family >= 0)
            )
            if bool(same_family_rows.any().item()):
                candidate_mask = packed_view.family_ids == flat_teacher_family.index_select(
                    0,
                    packed_view.row_indices.to(dtype=torch.long),
                )
                same_family_log_probs, same_family_top_actions = _packed_subset_action_logp_and_top_action(
                    packed_view,
                    flat_teacher_action,
                    candidate_mask=candidate_mask,
                    strict=False,
                )
                same_family_log_probs = same_family_log_probs.reshape(-1).to(dtype=value_dtype)
                same_family_top_actions = same_family_top_actions.reshape(-1).to(dtype=torch.long)
                supported = same_family_rows & torch.isfinite(same_family_log_probs)
                row_weight = flat_loss_mask[same_family_rows]
                if float(row_weight.sum().item()) > 0.0:
                    metrics["teacher_same_family_action_supported_fraction"] = float(
                        (flat_loss_mask[supported].sum().item()) / max(float(row_weight.sum().item()), 1.0e-8)
                    )
                if bool(supported.any().item()):
                    supported_weights = flat_loss_mask[supported]
                    supported_targets = flat_teacher_action[supported]
                    same_family_action_loss = _weighted_mean(
                        -same_family_log_probs[supported],
                        supported_weights,
                    ).to(dtype=value_dtype)
                    metrics["teacher_same_family_action_accuracy"] = float(
                        ((same_family_top_actions[supported] == supported_targets).float() * supported_weights)
                        .sum()
                        .item()
                        / max(float(supported_weights.sum().item()), 1.0)
                    )
                    metrics["teacher_same_family_action_loss"] = float(same_family_action_loss.detach().item())
                    context["teacher_same_family_action_log_probs"] = same_family_log_probs[supported].detach()
                    main_play_supported = supported & (flat_teacher_family == play_family_id)
                    if bool(main_play_supported.any().item()):
                        main_play_weights = flat_loss_mask[main_play_supported]
                        metrics["teacher_same_family_main_play_character_accuracy"] = float(
                            (
                                (
                                    same_family_top_actions[main_play_supported]
                                    == flat_teacher_action[main_play_supported]
                                ).float()
                                * main_play_weights
                            )
                            .sum()
                            .item()
                            / max(float(main_play_weights.sum().item()), 1.0)
                        )
                    main_move_family_id = int(family_index.get("main_move", -1))
                    main_move_supported = supported & (flat_teacher_family == main_move_family_id)
                    if bool(main_move_supported.any().item()):
                        main_move_weights = flat_loss_mask[main_move_supported]
                        metrics["teacher_same_family_main_move_accuracy"] = float(
                            (
                                (
                                    same_family_top_actions[main_move_supported]
                                    == flat_teacher_action[main_move_supported]
                                ).float()
                                * main_move_weights
                            )
                            .sum()
                            .item()
                            / max(float(main_move_weights.sum().item()), 1.0)
                        )

        public_heuristic_loss = zero
        if public_heuristic_target_logits is not None and float(public_heuristic_coef) != 0.0:
            public_candidate_rows = packed_view.row_has_candidates & flat_teacher_valid & active_rows
            public_rows = public_candidate_rows
            if public_heuristic_family_ids:
                public_rows = public_rows & torch.isin(
                    flat_teacher_family,
                    torch.as_tensor(
                        public_heuristic_family_ids,
                        device=flat_teacher_family.device,
                        dtype=flat_teacher_family.dtype,
                    ),
                )
            active_total = float(active_rows.float().sum().item())
            valid_total = float(public_candidate_rows.float().sum().item())
            if active_total > 0.0:
                metrics["teacher_public_heuristic_selected_fraction"] = float(
                    public_rows.float().sum().item() / active_total
                )
            if valid_total > 0.0:
                metrics["teacher_public_heuristic_teacher_valid_coverage"] = float(
                    public_rows.float().sum().item() / valid_total
                )
            if bool(public_rows.any().item()):
                row_cross_entropy, row_student_top_mass, row_target_entropy = _packed_soft_target_cross_entropy(
                    packed_view,
                    target_logits=public_heuristic_target_logits,
                    temperature=float(public_heuristic_temperature),
                    row_mask=public_rows,
                )
                public_weights = flat_loss_mask[public_rows]
                if float(public_weights.sum().item()) > 0.0:
                    metrics["teacher_public_heuristic_supported_fraction"] = 1.0
                    metrics["teacher_public_heuristic_top1_mass"] = float(
                        _weighted_mean(row_student_top_mass[public_rows], public_weights).item()
                    )
                    metrics["teacher_public_heuristic_target_entropy"] = float(
                        _weighted_mean(row_target_entropy[public_rows], public_weights).item()
                    )
                    public_heuristic_loss = _weighted_mean(
                        row_cross_entropy[public_rows],
                        public_weights,
                    ).to(dtype=value_dtype)
                    metrics["teacher_public_heuristic_loss"] = float(public_heuristic_loss.detach().item())

        public_main_move_loss = zero
        if packed_view is not None and public_heuristic_target_logits is not None:
            packed_move_source_log_probs = None
            packed_move_slot_log_probs = None
            if move_family_id >= 0:
                packed_move_source_log_probs = _packed_group_log_probs(
                    packed_view,
                    group_ids=packed_view.arg0,
                    group_count=max(int(action_catalog.max_stage), 1),
                    candidate_mask=packed_view.family_ids == move_family_id,
                )
                packed_move_slot_log_probs = _packed_group_log_probs(
                    packed_view,
                    group_ids=packed_view.arg1,
                    group_count=max(int(action_catalog.max_stage), 1),
                    candidate_mask=packed_view.family_ids == move_family_id,
                )
            public_main_move_loss, public_main_move_metrics = _public_main_move_auxiliary_loss(
                packed_view=packed_view,
                public_heuristic_target_logits=public_heuristic_target_logits,
                flat_loss_mask=flat_loss_mask,
                flat_teacher_valid=flat_teacher_valid,
                active_rows=active_rows,
                family_names=family_names,
                move_source_log_probs=packed_move_source_log_probs,
                move_slot_log_probs=packed_move_slot_log_probs,
                temperature=float(public_heuristic_temperature),
                zero=zero,
            )
            metrics.update(public_main_move_metrics)

        development_pass_suppression_loss = zero
        if float(development_pass_suppression_coef) != 0.0:
            pass_family_id = int(family_index.get("pass", -1))
            development_rows = family_rows & (
                (flat_teacher_family == play_family_id) | (flat_teacher_family == move_family_id)
            )
            if pass_family_id >= 0 and bool(development_rows.any().item()):
                pass_log_probs = family_log_probs[development_rows, pass_family_id]
                supported = torch.isfinite(pass_log_probs)
                if bool(supported.any().item()):
                    row_weight = flat_loss_mask[development_rows][supported]
                    pass_probs = torch.exp(pass_log_probs[supported]).clamp(max=1.0 - 1.0e-6)
                    development_pass_suppression_loss = _weighted_mean(
                        -torch.log1p(-pass_probs),
                        row_weight,
                    ).to(dtype=value_dtype)
                    active_total = float(active_rows.float().sum().item())
                    if active_total > 0.0:
                        metrics["teacher_development_pass_suppression_selected_fraction"] = float(
                            development_rows.float().sum().item() / active_total
                        )
                    metrics["teacher_development_pass_probability"] = float(
                        _weighted_mean(pass_probs.detach(), row_weight).item()
                    )
                    metrics["teacher_development_pass_suppression_loss"] = float(
                        development_pass_suppression_loss.detach().item()
                    )

        total_aux = (
            family_loss * float(family_coef)
            + slot_loss * float(slot_coef)
            + move_source_loss * float(move_source_coef)
            + attack_type_loss * float(attack_type_coef)
            + action_loss * float(action_coef)
            + same_family_action_loss * float(same_family_action_coef)
            + public_heuristic_loss * float(public_heuristic_coef)
            + public_main_move_loss * float(public_main_move_coef)
            + development_pass_suppression_loss * float(development_pass_suppression_coef)
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
    move_to_slots = lookup["move_to_slots"]
    attack_slots = lookup["attack_slots"]
    attack_types = lookup["attack_types"]
    family_index = lookup["family_index"]
    family_names = lookup["family_names"]
    attack_type_names = lookup["attack_type_names"]
    public_heuristic_family_ids = _resolve_public_heuristic_family_ids(
        family_names=family_names,
        requested_families=tuple(public_heuristic_families),
    )

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
    move_family_id = int(family_index.get("main_move", -1))
    attack_family_id = int(family_index.get("attack", -1))
    _record_teacher_family_coverage(
        metrics,
        active_rows=flat_loss_mask > 0.0,
        flat_teacher_family_local=flat_teacher_family,
        flat_teacher_valid_local=flat_teacher_valid,
        play_family_id_local=play_family_id,
        move_family_id_local=move_family_id,
        attack_family_id_local=attack_family_id,
    )

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

    move_rows = family_rows & (flat_teacher_family == move_family_id) & (flat_teacher_slot >= 0)
    if move_family_id >= 0 and bool(move_rows.any().item()):
        family_logits = masked_logits[move_rows]
        family_mask = flat_legal_mask[move_rows] & (family_ids == move_family_id).unsqueeze(0)
        group_log_probs = _group_log_probs(
            masked_logits=torch.where(family_mask, family_logits, torch.full_like(family_logits, -1.0e9)),
            group_ids=move_to_slots,
            group_count=max(int(action_catalog.max_stage), 1),
        )
        targets = flat_teacher_slot[move_rows]
        row_weight = flat_loss_mask[move_rows]
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

    action_loss = zero
    if flat_teacher_action is not None and float(action_coef) != 0.0:
        action_rows = flat_teacher_valid & (flat_teacher_action >= 0)
        if bool(action_rows.any().item()):
            action_targets = flat_teacher_action[action_rows]
            action_weights = flat_loss_mask[action_rows]
            action_masks = flat_legal_mask[action_rows]
            action_logits = flat_logits[action_rows]
            action_log_probs = torch.full(
                action_targets.shape,
                float("-inf"),
                dtype=flat_logits.dtype,
                device=flat_logits.device,
            )
            predictions = torch.full_like(action_targets, -1)
            empty_rows = ~action_masks.any(dim=1)
            if bool((~empty_rows).any().item()):
                non_empty_targets = action_targets[~empty_rows]
                non_empty_masks = action_masks[~empty_rows]
                non_empty_logits = action_logits[~empty_rows]
                in_range = (non_empty_targets >= 0) & (non_empty_targets < non_empty_logits.shape[-1])
                if bool(in_range.any().item()):
                    selected_masks = non_empty_masks[in_range]
                    selected_targets = non_empty_targets[in_range]
                    supported = selected_masks.gather(1, selected_targets.unsqueeze(1)).squeeze(1)
                    if bool(supported.any().item()):
                        supported_logits = non_empty_logits[in_range][supported]
                        supported_masks = selected_masks[supported]
                        supported_targets = selected_targets[supported]
                        supported_log_probs, _supported_entropy = _masked_log_probs_and_entropy(
                            supported_logits,
                            supported_masks,
                        )
                        gather_log_probs = supported_log_probs.gather(1, supported_targets.unsqueeze(1)).squeeze(1)
                        action_log_probs[torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)[in_range][supported]] = (
                            gather_log_probs
                        )
                        predictions[torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)[in_range][supported]] = (
                            supported_log_probs.argmax(dim=1)
                        )
            if int(action_catalog.pass_action_id) >= 0 and bool(empty_rows.any().item()):
                pass_supported = action_targets[empty_rows] == int(action_catalog.pass_action_id)
                if bool(pass_supported.any().item()):
                    empty_indices = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)[pass_supported]
                    action_log_probs[empty_indices] = 0.0
                    predictions[empty_indices] = int(action_catalog.pass_action_id)
            supported_rows = torch.isfinite(action_log_probs)
            if float(action_weights.sum().item()) > 0.0:
                metrics["teacher_action_supported_fraction"] = float(
                    (action_weights[supported_rows].sum().item()) / max(float(action_weights.sum().item()), 1.0e-8)
                )
            if bool(supported_rows.any().item()):
                supported_weights = action_weights[supported_rows]
                supported_log_probs = action_log_probs[supported_rows]
                supported_predictions = predictions[supported_rows]
                supported_targets = action_targets[supported_rows]
                action_loss = _weighted_mean(-supported_log_probs, supported_weights).to(dtype=logits.dtype)
                metrics["teacher_action_accuracy"] = float(
                    ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                metrics["teacher_action_loss"] = float(action_loss.detach().item())
                context["teacher_action_log_probs"] = supported_log_probs.detach()

    same_family_action_loss = zero
    if flat_teacher_action is not None and float(same_family_action_coef) != 0.0:
        same_family_rows = flat_teacher_valid & (flat_teacher_action >= 0) & (flat_teacher_family >= 0)
        if bool(same_family_rows.any().item()):
            row_targets = flat_teacher_action[same_family_rows]
            row_weights = flat_loss_mask[same_family_rows]
            row_logits = flat_logits[same_family_rows]
            row_masks = flat_legal_mask[same_family_rows]
            row_teacher_families = flat_teacher_family[same_family_rows]
            same_family_masks = row_masks & (family_ids.unsqueeze(0) == row_teacher_families.unsqueeze(1))
            same_family_log_probs, _same_family_entropy = _masked_log_probs_and_entropy(
                row_logits,
                same_family_masks,
            )
            supported = same_family_masks.gather(1, row_targets.unsqueeze(1)).squeeze(1)
            if float(row_weights.sum().item()) > 0.0:
                metrics["teacher_same_family_action_supported_fraction"] = float(
                    (row_weights[supported].sum().item()) / max(float(row_weights.sum().item()), 1.0e-8)
                )
            if bool(supported.any().item()):
                supported_targets = row_targets[supported]
                supported_weights = row_weights[supported]
                supported_log_probs = (
                    same_family_log_probs[supported].gather(1, supported_targets.unsqueeze(1)).squeeze(1)
                )
                supported_predictions = same_family_log_probs[supported].argmax(dim=1)
                same_family_action_loss = _weighted_mean(-supported_log_probs, supported_weights).to(dtype=logits.dtype)
                metrics["teacher_same_family_action_accuracy"] = float(
                    ((supported_predictions == supported_targets).float() * supported_weights).sum().item()
                    / max(float(supported_weights.sum().item()), 1.0)
                )
                metrics["teacher_same_family_action_loss"] = float(same_family_action_loss.detach().item())
                context["teacher_same_family_action_log_probs"] = supported_log_probs.detach()
                supported_families = row_teacher_families[supported]
                main_play_supported = supported_families == play_family_id
                if bool(main_play_supported.any().item()):
                    play_weights = supported_weights[main_play_supported]
                    metrics["teacher_same_family_main_play_character_accuracy"] = float(
                        (
                            (
                                supported_predictions[main_play_supported] == supported_targets[main_play_supported]
                            ).float()
                            * play_weights
                        )
                        .sum()
                        .item()
                        / max(float(play_weights.sum().item()), 1.0)
                    )
                main_move_family_id = int(family_index.get("main_move", -1))
                main_move_supported = supported_families == main_move_family_id
                if bool(main_move_supported.any().item()):
                    move_weights = supported_weights[main_move_supported]
                    metrics["teacher_same_family_main_move_accuracy"] = float(
                        (
                            (
                                supported_predictions[main_move_supported] == supported_targets[main_move_supported]
                            ).float()
                            * move_weights
                        )
                        .sum()
                        .item()
                        / max(float(move_weights.sum().item()), 1.0)
                    )

    total_aux = (
        family_loss * float(family_coef)
        + slot_loss * float(slot_coef)
        + attack_type_loss * float(attack_type_coef)
        + action_loss * float(action_coef)
        + same_family_action_loss * float(same_family_action_coef)
    )
    metrics["teacher_aux_loss"] = float(total_aux.detach().item())
    context["teacher_aux_loss"] = total_aux.detach()
    return total_aux.to(dtype=logits.dtype), metrics, context


def _batch_value(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _time_step_legal_actions(
    legal_actions: LegalActionBatch | None, *, step_index: int, batch_size: int
) -> LegalActionBatch | None:
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


def _packed_selected_action_logp(
    packed_logits: Tensor,
    legal_ids: Tensor,
    legal_offsets: Tensor,
    actions: Tensor,
    *,
    pass_action_id: int | None,
    strict: bool = True,
) -> Tensor:
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

    selected_logp = torch.full(
        (row_count,),
        float("-inf") if not strict else 0.0,
        device=packed_logits.device,
        dtype=packed_logits.dtype,
    )
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

        repeated_actions = flat_actions[row_ids]
        matches = ids == repeated_actions
        match_counts = torch.segment_reduce(
            matches.to(dtype=packed_logits.dtype), reduce="sum", lengths=non_empty_widths
        )
        illegal_rows = match_counts != 1.0
        if strict and bool(illegal_rows.any().item()):
            bad_row = int(non_empty_rows[torch.nonzero(illegal_rows, as_tuple=False)[0].item()].item())
            bad_action = int(flat_actions[bad_row].item())
            raise ValueError(f"illegal action {bad_action} for row {bad_row}")
        selected_non_empty = torch.segment_reduce(
            torch.where(matches, log_probs, torch.zeros_like(log_probs)),
            reduce="sum",
            lengths=non_empty_widths,
        )
        if strict:
            selected_logp[non_empty_rows] = selected_non_empty.to(dtype=selected_logp.dtype)
        else:
            supported_rows = match_counts == 1.0
            if bool(supported_rows.any().item()):
                selected_logp[non_empty_rows[supported_rows]] = selected_non_empty[supported_rows].to(
                    dtype=selected_logp.dtype
                )

    if bool(empty_rows.any().item()):
        if pass_action_id is None:
            if strict:
                raise ValueError("pass_action_id is required when legality contains empty rows")
        else:
            support_empty_rows = flat_actions[empty_rows] == int(pass_action_id)
            if strict and bool((~support_empty_rows).any().item()):
                row_index = int(
                    torch.nonzero(empty_rows & (flat_actions != int(pass_action_id)), as_tuple=False)[0].item()
                )
                action = int(flat_actions[row_index].item())
                raise ValueError(
                    f"row {row_index} has no legal actions; expected pass action {pass_action_id}, got {action}"
                )
            if bool(support_empty_rows.any().item()):
                empty_row_ids = torch.nonzero(empty_rows, as_tuple=False).squeeze(1)
                selected_logp[empty_row_ids[support_empty_rows]] = 0.0

    return selected_logp.reshape(actions.shape)


def _packed_subset_action_logp_and_top_action(
    packed_view: _PackedStructuredLegalView,
    actions: Tensor,
    *,
    candidate_mask: Tensor,
    strict: bool = True,
) -> tuple[Tensor, Tensor]:
    flat_actions = actions.reshape(-1).to(dtype=torch.long, device=packed_view.logits.device)
    row_count = int(flat_actions.shape[0])
    if row_count != int(packed_view.row_count):
        raise ValueError("actions must align with the packed row count")
    if candidate_mask.shape != packed_view.action_ids.shape:
        raise ValueError("candidate_mask must align 1:1 with packed action ids")

    selected = candidate_mask.to(device=packed_view.logits.device, dtype=torch.bool)
    selected_logp = torch.full(
        (row_count,),
        float("-inf") if not strict else 0.0,
        device=packed_view.logits.device,
        dtype=packed_view.logits.dtype,
    )
    top_action_ids = torch.full(
        (row_count,),
        -1,
        device=packed_view.logits.device,
        dtype=torch.long,
    )
    if not bool(selected.any().item()):
        return selected_logp.reshape(actions.shape), top_action_ids.reshape(actions.shape)

    row_indices = packed_view.row_indices[selected].to(dtype=torch.long)
    subset_logits = packed_view.logits[selected]
    subset_action_ids = packed_view.action_ids[selected].to(dtype=torch.long)
    row_log_z = _segment_logsumexp(subset_logits, row_indices, row_count)
    log_probs = subset_logits - row_log_z.index_select(0, row_indices)

    repeated_actions = flat_actions.index_select(0, row_indices)
    matches = subset_action_ids == repeated_actions
    match_counts = torch.zeros((row_count,), device=packed_view.logits.device, dtype=packed_view.logits.dtype)
    match_counts.scatter_add_(0, row_indices, matches.to(dtype=packed_view.logits.dtype))
    illegal_rows = match_counts != 1.0
    if strict and bool(illegal_rows.any().item()):
        bad_row = int(torch.nonzero(illegal_rows, as_tuple=False)[0].item())
        bad_action = int(flat_actions[bad_row].item())
        raise ValueError(f"illegal action {bad_action} for subset row {bad_row}")

    selected_non_empty = torch.zeros((row_count,), device=packed_view.logits.device, dtype=packed_view.logits.dtype)
    selected_non_empty.scatter_add_(0, row_indices, torch.where(matches, log_probs, torch.zeros_like(log_probs)))
    if strict:
        row_has_candidates = torch.zeros((row_count,), device=packed_view.logits.device, dtype=torch.bool)
        row_has_candidates[row_indices] = True
        selected_logp[row_has_candidates] = selected_non_empty[row_has_candidates]
    else:
        supported_rows = match_counts == 1.0
        if bool(supported_rows.any().item()):
            selected_logp[supported_rows] = selected_non_empty[supported_rows]

    top_logits = _segment_max(subset_logits, row_indices, row_count)
    top_matches = subset_logits >= (top_logits.index_select(0, row_indices) - 1.0e-6)
    top_action_ids.scatter_reduce_(
        0,
        row_indices,
        torch.where(top_matches, subset_action_ids, torch.full_like(subset_action_ids, -1)),
        reduce="amax",
        include_self=True,
    )
    return selected_logp.reshape(actions.shape), top_action_ids.reshape(actions.shape)


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
    selected_logp = _packed_selected_action_logp(
        packed_logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=pass_action_id,
        strict=True,
    )
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

    entropy = torch.zeros((row_count,), device=packed_logits.device, dtype=packed_logits.dtype)
    empty_rows = widths == 0
    non_empty_rows = torch.nonzero(~empty_rows, as_tuple=False).squeeze(1)
    if non_empty_rows.numel() > 0:
        non_empty_widths = widths[non_empty_rows]
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

    return selected_logp.reshape(actions.shape), entropy.reshape(actions.shape)
