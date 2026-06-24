"""Shared field parsers for structured training auxiliary config."""

from __future__ import annotations

from typing import Any

from weiss_rl.config.loading.parsing_utils import require_float, require_int, require_str_list
from weiss_rl.config.sections.sections_training_schema import (
    TRAINING_PUBLIC_HEURISTIC_PROFILES,
    TRAINING_TRAJECTORY_RETENTION_SOURCES,
)


def public_heuristic_profiles(payload: dict[str, Any], *, key: str, field_name: str) -> tuple[str, ...]:
    profiles = tuple(
        name.strip().lower()
        for name in require_str_list(
            payload.get(key, []),
            field_name=field_name,
        )
        if name.strip()
    )
    invalid_profiles = sorted(set(profiles) - TRAINING_PUBLIC_HEURISTIC_PROFILES)
    if invalid_profiles:
        raise ValueError(f"{field_name} contains unsupported profiles: " + ", ".join(invalid_profiles))
    return profiles


def require_nonnegative_float(payload: dict[str, Any], key: str, default: float, *, field_name: str) -> float:
    value = require_float(payload.get(key, default), field_name=field_name)
    if value < 0.0:
        raise ValueError(f"{field_name} must be >= 0.0")
    return value


def require_positive_float(payload: dict[str, Any], key: str, default: float, *, field_name: str) -> float:
    value = require_float(payload.get(key, default), field_name=field_name)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def require_update_window(
    payload: dict[str, Any],
    *,
    start_key: str,
    end_key: str,
    start_default: int = 0,
    end_default: int = -1,
    context: str,
) -> tuple[int, int]:
    start_updates = require_int(
        payload.get(start_key, start_default),
        field_name=f"{context}.{start_key}",
        minimum=0,
    )
    end_updates = require_int(
        payload.get(end_key, end_default),
        field_name=f"{context}.{end_key}",
        minimum=-1,
    )
    if end_updates >= 0 and end_updates < start_updates:
        raise ValueError(f"{context}.{end_key} must be >= {context}.{start_key}")
    return start_updates, end_updates


def trajectory_retention_sources(payload: dict[str, Any]) -> tuple[str, ...]:
    sources = tuple(
        source.strip().lower()
        for source in require_str_list(
            payload.get("trajectory_retention_sources", ["champions"]),
            field_name="training.structured_aux.trajectory_retention_sources",
        )
        if source.strip()
    )
    invalid_sources = sorted(set(sources) - TRAINING_TRAJECTORY_RETENTION_SOURCES)
    if invalid_sources:
        raise ValueError(
            "training.structured_aux.trajectory_retention_sources contains unsupported sources: "
            + ", ".join(invalid_sources)
        )
    return sources


__all__ = [
    "public_heuristic_profiles",
    "require_nonnegative_float",
    "require_positive_float",
    "require_update_window",
    "trajectory_retention_sources",
]
