"""Structured auxiliary focus-group parsing helpers."""

from __future__ import annotations

from typing import Any

from weiss_rl.config.models import TrainingTrajectoryBcFocusGroupConfig
from weiss_rl.config.parsing_utils import (
    reject_unknown_keys,
    require_float,
    require_mapping,
    require_str_list,
    require_text,
)


def trajectory_bc_focus_fraction(structured_aux: dict[str, Any]) -> float:
    fraction = require_float(
        structured_aux.get("trajectory_bc_focus_fraction", 0.0),
        field_name="training.structured_aux.trajectory_bc_focus_fraction",
    )
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("training.structured_aux.trajectory_bc_focus_fraction must be between 0.0 and 1.0")
    return fraction


def trajectory_bc_focus_groups(structured_aux: dict[str, Any]) -> tuple[TrainingTrajectoryBcFocusGroupConfig, ...]:
    return _focus_groups(
        structured_aux,
        key="trajectory_bc_focus_groups",
        context_prefix="training.structured_aux.trajectory_bc_focus_groups",
    )


def trajectory_bc_focus_source_labels(structured_aux: dict[str, Any]) -> tuple[str, ...]:
    return require_str_list(
        structured_aux.get("trajectory_bc_focus_source_labels", []),
        field_name="training.structured_aux.trajectory_bc_focus_source_labels",
    )


def validate_trajectory_bc_focus_contract(
    *,
    source_labels: tuple[str, ...],
    fraction: float,
    groups: tuple[TrainingTrajectoryBcFocusGroupConfig, ...],
) -> None:
    if groups and (source_labels or fraction > 0.0):
        raise ValueError(
            "training.structured_aux.trajectory_bc_focus_groups cannot be combined with "
            "trajectory_bc_focus_source_labels or trajectory_bc_focus_fraction"
        )


def paired_swing_focus_fraction(structured_aux: dict[str, Any]) -> float:
    fraction = require_float(
        structured_aux.get("paired_swing_focus_fraction", 0.0),
        field_name="training.structured_aux.paired_swing_focus_fraction",
    )
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("training.structured_aux.paired_swing_focus_fraction must be between 0.0 and 1.0")
    return fraction


def paired_swing_focus_groups(structured_aux: dict[str, Any]) -> tuple[TrainingTrajectoryBcFocusGroupConfig, ...]:
    return _focus_groups(
        structured_aux,
        key="paired_swing_focus_groups",
        context_prefix="training.structured_aux.paired_swing_focus_groups",
    )


def paired_swing_focus_source_labels(structured_aux: dict[str, Any]) -> tuple[str, ...]:
    return require_str_list(
        structured_aux.get("paired_swing_focus_source_labels", []),
        field_name="training.structured_aux.paired_swing_focus_source_labels",
    )


def validate_paired_swing_focus_contract(
    *,
    source_labels: tuple[str, ...],
    fraction: float,
    groups: tuple[TrainingTrajectoryBcFocusGroupConfig, ...],
) -> None:
    if groups and (source_labels or fraction > 0.0):
        raise ValueError(
            "training.structured_aux.paired_swing_focus_groups cannot be combined with "
            "paired_swing_focus_source_labels or paired_swing_focus_fraction"
        )


def paired_swing_action_source(structured_aux: dict[str, Any], *, key: str, default: str) -> str:
    value = require_text(
        structured_aux.get(key, default),
        field_name=f"training.structured_aux.{key}",
    ).strip()
    if value not in {"actions", "teacher_action"}:
        raise ValueError(f"training.structured_aux.{key} must be one of: actions, teacher_action")
    return value


def _focus_groups(
    structured_aux: dict[str, Any],
    *,
    key: str,
    context_prefix: str,
) -> tuple[TrainingTrajectoryBcFocusGroupConfig, ...]:
    raw_groups = structured_aux.get(key, [])
    if not isinstance(raw_groups, list):
        raise ValueError(f"{context_prefix} must be a list")
    groups: list[TrainingTrajectoryBcFocusGroupConfig] = []
    total_fraction = 0.0
    seen_names: set[str] = set()
    seen_labels: set[str] = set()
    for index, raw_group in enumerate(raw_groups):
        context = f"{context_prefix}[{index}]"
        group = require_mapping(raw_group, context=context)
        reject_unknown_keys(group, allowed={"name", "source_labels", "fraction"}, context=context)
        name = str(group.get("name", f"group_{index}")).strip()
        if not name:
            raise ValueError(f"{context}.name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"{context_prefix} contains duplicate name: {name}")
        labels = tuple(
            label.strip()
            for label in require_str_list(
                group.get("source_labels", []),
                field_name=f"{context}.source_labels",
            )
            if label.strip()
        )
        if not labels:
            raise ValueError(f"{context}.source_labels must contain at least one label")
        duplicate_labels = sorted(label for label in labels if label in seen_labels)
        if duplicate_labels:
            raise ValueError(f"{context_prefix} contains labels in multiple groups: " + ", ".join(duplicate_labels))
        fraction = require_float(group.get("fraction", 0.0), field_name=f"{context}.fraction")
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"{context}.fraction must be between 0.0 and 1.0")
        total_fraction += fraction
        if total_fraction > 1.0 + 1e-9:
            raise ValueError(f"{context_prefix} fractions must sum to <= 1.0")
        seen_names.add(name)
        seen_labels.update(labels)
        groups.append(TrainingTrajectoryBcFocusGroupConfig(name=name, source_labels=labels, fraction=fraction))
    return tuple(groups)


__all__ = [
    "paired_swing_action_source",
    "paired_swing_focus_fraction",
    "paired_swing_focus_groups",
    "paired_swing_focus_source_labels",
    "trajectory_bc_focus_fraction",
    "trajectory_bc_focus_groups",
    "trajectory_bc_focus_source_labels",
    "validate_paired_swing_focus_contract",
    "validate_trajectory_bc_focus_contract",
]
