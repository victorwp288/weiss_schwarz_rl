"""Curriculum stack config section parser."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import CurriculumCheckpointGuardConfig, CurriculumConfig, CurriculumStallMonitorConfig
from .parsing_utils import reject_unknown_keys, require_bool, require_float, require_int, require_mapping, require_text


def normalize_curriculum_payload(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [normalize_curriculum_payload(item, field_name=f"{field_name}[]") for item in value]
    if isinstance(value, Mapping):
        return {
            require_text(key, field_name=f"{field_name}.<key>"): normalize_curriculum_payload(
                item,
                field_name=f"{field_name}.{key}",
            )
            for key, item in value.items()
        }
    raise ValueError(f"{field_name} contains unsupported value type: {type(value).__name__}")


def parse_curriculum_config(body: dict[str, Any] | None) -> CurriculumConfig:
    if body is None:
        return CurriculumConfig()
    reject_unknown_keys(body, allowed={"simulator", "stall_monitor", "checkpoint_guard"}, context="curriculum")
    simulator = require_mapping(body.get("simulator", {}), context="curriculum.simulator")
    stall_monitor = require_mapping(body.get("stall_monitor", {}), context="curriculum.stall_monitor")
    checkpoint_guard = require_mapping(body.get("checkpoint_guard", {}), context="curriculum.checkpoint_guard")
    reject_unknown_keys(
        stall_monitor,
        allowed={"enabled", "truncation_rate_threshold", "consecutive_evals"},
        context="curriculum.stall_monitor",
    )
    reject_unknown_keys(
        checkpoint_guard,
        allowed={
            "enabled",
            "rollback_score_margin",
            "rollback_truncation_rate_threshold",
            "rollback_max_prob_lt_half",
            "min_best_score",
            "promote_min_prob_gt_half",
            "promote_max_ci_half_width",
            "cooldown_updates",
            "stop_after_rollback",
        },
        context="curriculum.checkpoint_guard",
    )
    return CurriculumConfig(
        simulator={
            key: normalize_curriculum_payload(value, field_name=f"curriculum.simulator.{key}")
            for key, value in simulator.items()
        },
        stall_monitor=CurriculumStallMonitorConfig(
            enabled=require_bool(stall_monitor.get("enabled", False), field_name="curriculum.stall_monitor.enabled"),
            truncation_rate_threshold=require_float(
                stall_monitor.get("truncation_rate_threshold", 1.0),
                field_name="curriculum.stall_monitor.truncation_rate_threshold",
            ),
            consecutive_evals=require_int(
                stall_monitor.get("consecutive_evals", 2),
                field_name="curriculum.stall_monitor.consecutive_evals",
                minimum=1,
            ),
        ),
        checkpoint_guard=CurriculumCheckpointGuardConfig(
            enabled=require_bool(
                checkpoint_guard.get("enabled", False),
                field_name="curriculum.checkpoint_guard.enabled",
            ),
            rollback_score_margin=require_float(
                checkpoint_guard.get("rollback_score_margin", 1.0),
                field_name="curriculum.checkpoint_guard.rollback_score_margin",
            ),
            rollback_truncation_rate_threshold=require_float(
                checkpoint_guard.get("rollback_truncation_rate_threshold", 1.0),
                field_name="curriculum.checkpoint_guard.rollback_truncation_rate_threshold",
            ),
            rollback_max_prob_lt_half=require_float(
                checkpoint_guard.get("rollback_max_prob_lt_half", 1.0),
                field_name="curriculum.checkpoint_guard.rollback_max_prob_lt_half",
            ),
            min_best_score=require_float(
                checkpoint_guard.get("min_best_score", 1.0),
                field_name="curriculum.checkpoint_guard.min_best_score",
            ),
            promote_min_prob_gt_half=require_float(
                checkpoint_guard.get("promote_min_prob_gt_half", 0.0),
                field_name="curriculum.checkpoint_guard.promote_min_prob_gt_half",
            ),
            promote_max_ci_half_width=require_float(
                checkpoint_guard.get("promote_max_ci_half_width", 1.0),
                field_name="curriculum.checkpoint_guard.promote_max_ci_half_width",
            ),
            cooldown_updates=require_int(
                checkpoint_guard.get("cooldown_updates", 0),
                field_name="curriculum.checkpoint_guard.cooldown_updates",
                minimum=0,
            ),
            stop_after_rollback=require_bool(
                checkpoint_guard.get("stop_after_rollback", False),
                field_name="curriculum.checkpoint_guard.stop_after_rollback",
            ),
        ),
    )
