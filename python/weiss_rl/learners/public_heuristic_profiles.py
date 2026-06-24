"""Public-heuristic teacher profile selection and target scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from weiss_rl.learners.tensor_ops import segment_logsumexp
from weiss_rl.public_heuristic.profiles import SUPPORTED_PUBLIC_HEURISTIC_PROFILES

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


def score_public_teacher_target_logits(
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


score_public_heuristic_target_logits = score_public_teacher_target_logits


__all__ = [
    "SUPPORTED_PUBLIC_HEURISTIC_PROFILE_MODES",
    "SUPPORTED_PUBLIC_HEURISTIC_PROFILES",
    "active_public_heuristic_profiles",
    "mix_public_heuristic_profile_logits",
    "normalize_public_heuristic_profile_mode",
    "normalize_public_heuristic_profiles",
    "score_public_heuristic_target_logits",
    "score_public_teacher_target_logits",
    "selected_public_heuristic_profiles",
]
