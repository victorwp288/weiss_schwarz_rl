"""Subsection extraction and key validation for training config parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.config.loading.parsing_utils import reject_unknown_keys, require_mapping
from weiss_rl.config.sections.sections_training_schema import (
    TRAINING_ACTION_SURFACE_KEYS,
    TRAINING_CHECKPOINTING_KEYS,
    TRAINING_EXPLORATION_KEYS,
    TRAINING_KEYS,
    TRAINING_OPTIMIZER_KEYS,
    TRAINING_PPO_KEYS,
    TRAINING_PRECISION_KEYS,
    TRAINING_ROLLOUT_KEYS,
    TRAINING_STRUCTURED_AUX_KEYS,
    TRAINING_STRUCTURED_METRICS_KEYS,
    TRAINING_STRUCTURED_WARMSTART_KEYS,
    TRAINING_TEACHER_AUX_KEYS,
    TRAINING_VTRACE_KEYS,
)


@dataclass(frozen=True, slots=True)
class TrainingSectionMappings:
    rollout: dict[str, Any]
    optimizer: dict[str, Any]
    exploration: dict[str, Any]
    precision: dict[str, Any]
    checkpointing: dict[str, Any]
    vtrace: dict[str, Any]
    ppo: dict[str, Any]
    structured_aux: dict[str, Any]
    structured_warmstart: dict[str, Any]
    structured_metrics: dict[str, Any]
    teacher_aux: dict[str, Any]
    action_surface: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingConfigSectionMapItem:
    section: str
    purpose: str
    key_count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "section": self.section,
            "purpose": self.purpose,
            "key_count": self.key_count,
        }


TRAINING_CONFIG_SECTION_MAP: tuple[TrainingConfigSectionMapItem, ...] = (
    TrainingConfigSectionMapItem("rollout", "runtime batch shape and actor collection cadence", len(TRAINING_ROLLOUT_KEYS)),
    TrainingConfigSectionMapItem("optimizer", "learner step size, coefficients, and gradient clipping", len(TRAINING_OPTIMIZER_KEYS)),
    TrainingConfigSectionMapItem("exploration", "actor sampling temperature and heuristic schedule", len(TRAINING_EXPLORATION_KEYS)),
    TrainingConfigSectionMapItem("precision", "mixed precision and runtime profiling toggles", len(TRAINING_PRECISION_KEYS)),
    TrainingConfigSectionMapItem("checkpointing", "checkpoint cadence and logging intervals", len(TRAINING_CHECKPOINTING_KEYS)),
    TrainingConfigSectionMapItem("vtrace", "IMPALA off-policy clipping constants", len(TRAINING_VTRACE_KEYS)),
    TrainingConfigSectionMapItem("ppo", "PPO-lite ablation settings", len(TRAINING_PPO_KEYS)),
    TrainingConfigSectionMapItem("structured_aux", "structured auxiliary and replay-retention settings", len(TRAINING_STRUCTURED_AUX_KEYS)),
    TrainingConfigSectionMapItem("structured_warmstart", "structured-head warmstart controls", len(TRAINING_STRUCTURED_WARMSTART_KEYS)),
    TrainingConfigSectionMapItem("structured_metrics", "structured diagnostics emission controls", len(TRAINING_STRUCTURED_METRICS_KEYS)),
    TrainingConfigSectionMapItem("teacher_aux", "teacher-label and public-heuristic auxiliary loss controls", len(TRAINING_TEACHER_AUX_KEYS)),
    TrainingConfigSectionMapItem("action_surface", "actor/learner legal-action scoring surfaces", len(TRAINING_ACTION_SURFACE_KEYS)),
)


def training_config_section_map_payload() -> list[dict[str, object]]:
    return [section.as_payload() for section in TRAINING_CONFIG_SECTION_MAP]


def resolve_training_section_mappings(body: dict[str, Any]) -> TrainingSectionMappings:
    """Return validated nested mappings for the training config section."""

    reject_unknown_keys(body, allowed=TRAINING_KEYS, context="training")
    sections = TrainingSectionMappings(
        rollout=require_mapping(body["rollout"], context="training.rollout"),
        optimizer=require_mapping(body["optimizer"], context="training.optimizer"),
        exploration=require_mapping(body["exploration"], context="training.exploration"),
        precision=require_mapping(body["precision"], context="training.precision"),
        checkpointing=require_mapping(body["checkpointing"], context="training.checkpointing"),
        vtrace=require_mapping(body["vtrace"], context="training.vtrace"),
        ppo=require_mapping(body.get("ppo", {}), context="training.ppo"),
        structured_aux=require_mapping(body.get("structured_aux", {}), context="training.structured_aux"),
        structured_warmstart=require_mapping(
            body.get("structured_warmstart", {}),
            context="training.structured_warmstart",
        ),
        structured_metrics=require_mapping(
            body.get("structured_metrics", {}),
            context="training.structured_metrics",
        ),
        teacher_aux=require_mapping(body.get("teacher_aux", {}), context="training.teacher_aux"),
        action_surface=require_mapping(body.get("action_surface", {}), context="training.action_surface"),
    )
    reject_unknown_keys(sections.rollout, allowed=TRAINING_ROLLOUT_KEYS, context="training.rollout")
    reject_unknown_keys(sections.optimizer, allowed=TRAINING_OPTIMIZER_KEYS, context="training.optimizer")
    reject_unknown_keys(sections.exploration, allowed=TRAINING_EXPLORATION_KEYS, context="training.exploration")
    reject_unknown_keys(sections.precision, allowed=TRAINING_PRECISION_KEYS, context="training.precision")
    reject_unknown_keys(
        sections.checkpointing,
        allowed=TRAINING_CHECKPOINTING_KEYS,
        context="training.checkpointing",
    )
    reject_unknown_keys(sections.vtrace, allowed=TRAINING_VTRACE_KEYS, context="training.vtrace")
    reject_unknown_keys(sections.ppo, allowed=TRAINING_PPO_KEYS, context="training.ppo")
    reject_unknown_keys(
        sections.structured_aux,
        allowed=TRAINING_STRUCTURED_AUX_KEYS,
        context="training.structured_aux",
    )
    reject_unknown_keys(
        sections.structured_warmstart,
        allowed=TRAINING_STRUCTURED_WARMSTART_KEYS,
        context="training.structured_warmstart",
    )
    reject_unknown_keys(
        sections.structured_metrics,
        allowed=TRAINING_STRUCTURED_METRICS_KEYS,
        context="training.structured_metrics",
    )
    reject_unknown_keys(sections.teacher_aux, allowed=TRAINING_TEACHER_AUX_KEYS, context="training.teacher_aux")
    reject_unknown_keys(
        sections.action_surface,
        allowed=TRAINING_ACTION_SURFACE_KEYS,
        context="training.action_surface",
    )
    return sections


__all__ = [
    "TRAINING_CONFIG_SECTION_MAP",
    "TrainingConfigSectionMapItem",
    "TrainingSectionMappings",
    "resolve_training_section_mappings",
    "training_config_section_map_payload",
]
