"""Public-heuristic teacher config parsing shared by aux and warmstart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.config.loading.parsing_utils import require_choice, require_int
from weiss_rl.config.sections.sections_training_schema import TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES

from .sections_training_aux_helpers import public_heuristic_profiles, require_positive_float


@dataclass(frozen=True, slots=True)
class PublicHeuristicTeacherProfileSettings:
    temperature: float
    profiles: tuple[str, ...]
    profile_mode: str
    profiles_end_updates: int


def parse_public_heuristic_teacher_profile_settings(
    body: dict[str, Any],
    *,
    context: str,
) -> PublicHeuristicTeacherProfileSettings:
    return PublicHeuristicTeacherProfileSettings(
        temperature=require_positive_float(
            body,
            "teacher_public_heuristic_temperature",
            32.0,
            field_name=f"{context}.teacher_public_heuristic_temperature",
        ),
        profiles=public_heuristic_profiles(
            body,
            key="teacher_public_heuristic_profiles",
            field_name=f"{context}.teacher_public_heuristic_profiles",
        ),
        profile_mode=require_choice(
            body.get("teacher_public_heuristic_profile_mode", "mixture"),
            field_name=f"{context}.teacher_public_heuristic_profile_mode",
            allowed=TRAINING_PUBLIC_HEURISTIC_PROFILE_MODES,
        ),
        profiles_end_updates=require_int(
            body.get("teacher_public_heuristic_profiles_end_updates", -1),
            field_name=f"{context}.teacher_public_heuristic_profiles_end_updates",
            minimum=-1,
        ),
    )


__all__ = [
    "PublicHeuristicTeacherProfileSettings",
    "parse_public_heuristic_teacher_profile_settings",
]
