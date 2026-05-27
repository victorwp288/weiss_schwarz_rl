"""Structured auxiliary learner metadata and configuration helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch import Tensor

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.tensor_ops import segment_logsumexp, segment_max

SUPPORTED_PUBLIC_HEURISTIC_PROFILES = frozenset({"base", "aggressive", "control"})
SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES = frozenset({"mixture", "cycle"})


def normalize_public_heuristic_profiles(profiles: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Normalize public-heuristic teacher profile names."""
    normalized: list[str] = []
    for raw_name in profiles or ():
        name = str(raw_name).strip().lower()
        if not name or name in normalized:
            continue
        normalized.append(name)
    if not normalized:
        return ("base",)
    invalid = sorted(set(normalized) - SUPPORTED_PUBLIC_HEURISTIC_PROFILES)
    if invalid:
        raise ValueError("teacher_public_heuristic_profiles contains unsupported profiles: " + ", ".join(invalid))
    return tuple(normalized)


def normalize_public_heuristic_profile_mode(mode: str | None) -> str:
    """Normalize the public-heuristic profile selection mode."""
    normalized = str(mode or "mixture").strip().lower()
    if normalized not in SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES:
        raise ValueError(
            "teacher_public_heuristic_profile_mode must be one of: "
            + ", ".join(sorted(SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES))
        )
    return normalized


def active_public_heuristic_profiles(
    profiles: tuple[str, ...],
    *,
    update_count: int,
    end_updates: int,
) -> tuple[str, ...]:
    if not profiles:
        return ("base",)
    if int(end_updates) >= 0 and int(update_count) > int(end_updates):
        return (profiles[0],)
    return profiles


def selected_public_heuristic_profiles(
    profiles: tuple[str, ...],
    *,
    profile_mode: str,
    update_count: int,
    end_updates: int,
) -> tuple[str, ...]:
    active_profiles = active_public_heuristic_profiles(
        profiles,
        update_count=update_count,
        end_updates=end_updates,
    )
    if len(active_profiles) > 1 and str(profile_mode) == "cycle":
        return (active_profiles[int(update_count) % len(active_profiles)],)
    return active_profiles


def mix_public_heuristic_profile_logits(
    profile_logits: list[Tensor],
    *,
    offsets: Tensor,
    temperature: float,
    device: torch.device,
) -> Tensor:
    if not profile_logits:
        return torch.zeros((0,), device=device)
    if len(profile_logits) == 1:
        return profile_logits[0]
    offsets = torch.as_tensor(offsets, device=device, dtype=torch.long)
    row_count = max(int(offsets.shape[0]) - 1, 0)
    total_candidates = int(offsets[-1].item()) if offsets.numel() > 0 else 0
    if row_count == 0 or total_candidates == 0:
        return profile_logits[0]
    widths = (offsets[1:] - offsets[:-1]).to(dtype=torch.long)
    row_indices = torch.repeat_interleave(
        torch.arange(row_count, device=device, dtype=torch.long),
        widths,
    )
    scaled_profile_log_probs: list[Tensor] = []
    temperature_value = float(temperature)
    for logits in profile_logits:
        scaled_logits = logits.to(device=device) / temperature_value
        row_log_z = segment_logsumexp(scaled_logits, row_indices, row_count)
        scaled_profile_log_probs.append(scaled_logits - row_log_z.index_select(0, row_indices))
    mixture_log_probs = torch.logsumexp(
        torch.stack(scaled_profile_log_probs, dim=0),
        dim=0,
    ) - math.log(float(len(scaled_profile_log_probs)))
    return mixture_log_probs * temperature_value


def score_public_heuristic_target_logits(
    *,
    forward_model: Any,
    obs_rows: Tensor,
    legal_actions: Any,
    observation_context: Mapping[str, Tensor] | None,
    profiles: tuple[str, ...],
    profile_mode: str,
    update_count: int,
    end_updates: int,
    temperature: float,
    device: torch.device,
) -> Tensor:
    profile_names = selected_public_heuristic_profiles(
        profiles,
        profile_mode=profile_mode,
        update_count=update_count,
        end_updates=end_updates,
    )
    profile_logits: list[Tensor] = []
    for profile_name in profile_names:
        profile_logits.append(
            torch.as_tensor(
                forward_model.score_packed_public_heuristic_candidates(
                    obs_rows,
                    legal_actions,
                    observation_context=observation_context,
                    scoring_profile=profile_name,
                ),
                device=device,
            ).reshape(-1)
        )
    return mix_public_heuristic_profile_logits(
        profile_logits,
        offsets=legal_actions.offsets,
        temperature=temperature,
        device=device,
    )


@dataclass(frozen=True, slots=True)
class StructuredCatalogMetadata:
    family_names: tuple[str, ...]
    attack_type_names: tuple[str, ...]
    family_ids: tuple[int, ...]
    hand_indices: tuple[int, ...]
    play_slots: tuple[int, ...]
    move_from_slots: tuple[int, ...]
    move_to_slots: tuple[int, ...]
    attack_slots: tuple[int, ...]
    attack_types: tuple[int, ...]
    main_move_02_action_id: int | None


@dataclass(frozen=True, slots=True)
class PackedStructuredLegalView:
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


@lru_cache(maxsize=8)
def structured_catalog_metadata(action_catalog: ActionCatalog) -> StructuredCatalogMetadata:
    """Build stable per-action structured metadata for auxiliary losses."""
    family_names = tuple(family.name for family in action_catalog.families)
    attack_type_names = tuple(action_catalog.attack_type_names)
    family_index = {name: index for index, name in enumerate(family_names)}
    action_space = int(action_catalog.action_space_size)
    family_ids = np.full((action_space,), -1, dtype=np.int64)
    hand_indices = np.full((action_space,), -1, dtype=np.int64)
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
        if decoded.hand_index is not None:
            hand_indices[action_id] = int(decoded.hand_index)
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
    return StructuredCatalogMetadata(
        family_names=family_names,
        attack_type_names=attack_type_names,
        family_ids=tuple(int(value) for value in family_ids.tolist()),
        hand_indices=tuple(int(value) for value in hand_indices.tolist()),
        play_slots=tuple(int(value) for value in play_slots.tolist()),
        move_from_slots=tuple(int(value) for value in move_from_slots.tolist()),
        move_to_slots=tuple(int(value) for value in move_to_slots.tolist()),
        attack_slots=tuple(int(value) for value in attack_slots.tolist()),
        attack_types=tuple(int(value) for value in attack_types.tolist()),
        main_move_02_action_id=main_move_02_action_id,
    )


def packed_structured_legal_view(
    *,
    logits: Tensor | None,
    packed_ids: Tensor | None,
    packed_offsets: Tensor | None,
    packed_meta: Tensor | None,
) -> PackedStructuredLegalView | None:
    """Build a row-indexed view over packed legal actions."""
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

    return PackedStructuredLegalView(
        row_count=row_count,
        row_indices=row_indices,
        action_ids=packed_ids.to(device=flat_device, dtype=torch.long),
        logits=selected_logits,
        row_log_z=segment_logsumexp(selected_logits, row_indices, row_count),
        row_has_candidates=lengths.to(device=flat_device, dtype=torch.bool),
        family_ids=_normalize_meta_column(meta_long[:, 0]),
        arg0=_normalize_meta_column(meta_long[:, 1]),
        arg1=_normalize_meta_column(meta_long[:, 2]),
        arg2=_normalize_meta_column(meta_long[:, 3]),
    )


def structured_group_lookup(action_catalog: ActionCatalog, *, device: torch.device) -> dict[str, Any]:
    metadata = structured_catalog_metadata(action_catalog)
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


def dense_group_log_probs(
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


def packed_group_log_probs(
    packed_view: PackedStructuredLegalView,
    *,
    group_ids: Tensor,
    group_count: int,
    candidate_mask: Tensor | None = None,
) -> Tensor:
    """Compute per-row log probability mass for packed action groups."""
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
        else segment_logsumexp(
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
    grouped = segment_logsumexp(packed_view.logits[valid], flat_keys, packed_view.row_count * group_count).view(
        packed_view.row_count,
        group_count,
    )
    finite_rows = torch.isfinite(row_log_z)
    if bool(finite_rows.any().item()):
        out[finite_rows] = grouped[finite_rows] - row_log_z[finite_rows].unsqueeze(1)
    return out


def packed_soft_target_cross_entropy(
    packed_view: PackedStructuredLegalView,
    *,
    target_logits: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-row soft-target cross entropy, top mass, and entropy."""
    if temperature <= 0.0:
        raise ValueError("public heuristic temperature must be > 0")
    flat_target_logits = target_logits.reshape(-1).to(device=packed_view.logits.device, dtype=packed_view.logits.dtype)
    if int(flat_target_logits.shape[0]) != int(packed_view.logits.shape[0]):
        raise ValueError("public heuristic target logits must align 1:1 with packed logits")
    scaled_target_logits = flat_target_logits / float(temperature)
    target_row_log_z = segment_logsumexp(scaled_target_logits, packed_view.row_indices, packed_view.row_count)
    target_log_probs = scaled_target_logits - target_row_log_z.index_select(
        0, packed_view.row_indices.to(dtype=torch.long)
    )
    target_probs = torch.exp(target_log_probs)
    student_log_probs = packed_view.logits - packed_view.row_log_z.index_select(
        0, packed_view.row_indices.to(dtype=torch.long)
    )

    row_cross_entropy = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    row_cross_entropy.scatter_add_(
        0,
        packed_view.row_indices.to(dtype=torch.long),
        -(target_probs * student_log_probs),
    )

    row_target_entropy = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    row_target_entropy.scatter_add_(
        0,
        packed_view.row_indices.to(dtype=torch.long),
        -(target_probs * target_log_probs),
    )

    student_top_logits = segment_max(packed_view.logits, packed_view.row_indices, packed_view.row_count)
    student_top_mask = packed_view.logits >= (
        student_top_logits.index_select(0, packed_view.row_indices.to(dtype=torch.long)) - 1.0e-6
    )
    row_student_top_mass = torch.zeros(
        (packed_view.row_count,), dtype=packed_view.logits.dtype, device=packed_view.logits.device
    )
    if bool(student_top_mask.any().item()):
        row_student_top_mass.scatter_add_(
            0,
            packed_view.row_indices[student_top_mask].to(dtype=torch.long),
            target_probs[student_top_mask],
        )
    return row_cross_entropy, row_student_top_mass, row_target_entropy


def resolve_public_heuristic_family_ids(
    *,
    family_names: tuple[str, ...],
    requested_families: tuple[str, ...],
) -> tuple[int, ...]:
    """Resolve configured public-heuristic family names to catalog ids."""
    normalized = tuple(str(name).strip() for name in requested_families if str(name).strip())
    if not normalized:
        return ()
    family_index = {name: index for index, name in enumerate(family_names)}
    missing = sorted({name for name in normalized if name not in family_index})
    if missing:
        raise ValueError("teacher_public_heuristic_families contains unknown action families: " + ", ".join(missing))
    return tuple(int(family_index[name]) for name in normalized)
