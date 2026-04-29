"""Curriculum guard state updates for periodic dev-eval summaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from weiss_rl.config import StackConfig
from weiss_rl.training.dev_eval_metrics import (
    dev_eval_aggregate_score,
    dev_eval_is_authoritative,
    dev_eval_worst_stall_rate,
    summary_rate,
)


class CurriculumGuardPaths(Protocol):
    logs_dir: Path


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stall_monitor_state_path(training_paths: CurriculumGuardPaths) -> Path:
    return training_paths.logs_dir / "stall_monitor.json"


def early_cutoff_state_path(training_paths: CurriculumGuardPaths) -> Path:
    return training_paths.logs_dir / "early_cutoff.json"


def early_cutoff_events_path(training_paths: CurriculumGuardPaths) -> Path:
    return training_paths.logs_dir / "early_cutoff_events.jsonl"


def append_early_cutoff_event(training_paths: CurriculumGuardPaths, payload: Mapping[str, Any]) -> None:
    path = early_cutoff_events_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def update_stall_monitor(
    *,
    stack: StackConfig,
    training_paths: CurriculumGuardPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.stall_monitor.enabled:
        return None
    threshold = float(curriculum.stall_monitor.truncation_rate_threshold)
    required_consecutive = int(curriculum.stall_monitor.consecutive_evals)
    anchors_raw = summary_payload.get("anchors", {})
    if not isinstance(anchors_raw, Mapping):
        return None

    anchor_truncation_rates: dict[str, float] = {}
    anchor_no_progress_rates: dict[str, float] = {}
    anchor_natural_timeout_rates: dict[str, float] = {}
    anchor_stall_rates: dict[str, float] = {}
    for anchor_name, anchor_payload in anchors_raw.items():
        if not isinstance(anchor_payload, Mapping):
            continue
        matchup_summary = anchor_payload.get("summary", {})
        if not isinstance(matchup_summary, Mapping):
            continue
        truncation_rate = summary_rate(matchup_summary, "truncations")
        no_progress_rate = summary_rate(matchup_summary, "no_progress_timeouts")
        natural_timeout_rate = summary_rate(matchup_summary, "natural_timeouts")
        if truncation_rate is None and no_progress_rate is None and natural_timeout_rate is None:
            continue
        anchor_truncation_rates[anchor_name] = 0.0 if truncation_rate is None else truncation_rate
        anchor_no_progress_rates[anchor_name] = 0.0 if no_progress_rate is None else no_progress_rate
        anchor_natural_timeout_rates[anchor_name] = 0.0 if natural_timeout_rate is None else natural_timeout_rate
        anchor_stall_rates[anchor_name] = (
            anchor_no_progress_rates[anchor_name]
            if no_progress_rate is not None
            else anchor_truncation_rates[anchor_name]
        )
    if not anchor_stall_rates:
        return None

    state_path = stall_monitor_state_path(training_paths)
    state = load_json_object(state_path, label="stall monitor state") if state_path.is_file() else {}
    previous_consecutive = int(state.get("consecutive_trigger_count", 0))
    worst_anchor = max(anchor_stall_rates, key=anchor_stall_rates.get)
    worst_rate = float(anchor_stall_rates[worst_anchor])
    consecutive = previous_consecutive + 1 if worst_rate >= threshold else 0
    stall_risk = consecutive >= required_consecutive
    payload = {
        "enabled": True,
        "update_count": int(update_count),
        "threshold": threshold,
        "required_consecutive_evals": required_consecutive,
        "consecutive_trigger_count": consecutive,
        "stall_risk": stall_risk,
        "worst_anchor": worst_anchor,
        "stall_indicator_kind": (
            "no_progress_timeout" if anchor_no_progress_rates.get(worst_anchor, 0.0) > 0.0 else "truncation_fallback"
        ),
        "worst_stall_rate": worst_rate,
        "worst_truncation_rate": float(anchor_truncation_rates.get(worst_anchor, 0.0)),
        "worst_no_progress_timeout_rate": float(anchor_no_progress_rates.get(worst_anchor, 0.0)),
        "worst_natural_timeout_rate": float(anchor_natural_timeout_rates.get(worst_anchor, 0.0)),
        "anchor_truncation_rates": anchor_truncation_rates,
        "anchor_no_progress_timeout_rates": anchor_no_progress_rates,
        "anchor_natural_timeout_rates": anchor_natural_timeout_rates,
    }
    write_json(state_path, payload)
    return payload


def format_stall_monitor_warning(stall_monitor: Mapping[str, Any], *, update_count: int) -> str:
    return (
        "Stall monitor warning: "
        f"update={int(update_count)} worst_anchor={stall_monitor['worst_anchor']} "
        f"stall_rate={float(stall_monitor['worst_stall_rate']):.3f} "
        f"no_progress_rate={float(stall_monitor['worst_no_progress_timeout_rate']):.3f} "
        f"truncation_rate={float(stall_monitor['worst_truncation_rate']):.3f} "
        f"consecutive={int(stall_monitor['consecutive_trigger_count'])}"
    )


def early_cutoff_metric_updates(early_cutoff_payload: Mapping[str, Any]) -> dict[str, float]:
    return {
        "early_cutoff_triggered": 1.0,
        "early_cutoff_best_score": float(early_cutoff_payload["best_score"]),
        "early_cutoff_current_score": float(early_cutoff_payload["current_score"]),
        "early_cutoff_no_improvement_updates": float(early_cutoff_payload["no_improvement_updates"]),
        "early_cutoff_consecutive_stall_evals": float(early_cutoff_payload["consecutive_stall_evals"]),
    }


def format_early_cutoff_triggered_message(
    early_cutoff_payload: Mapping[str, Any],
    *,
    update_count: int,
) -> str:
    reasons = early_cutoff_payload.get("reasons", ())
    if not isinstance(reasons, list | tuple):
        reasons = ()
    return (
        "Early cutoff triggered: "
        f"update={int(update_count)} "
        f"best_update={int(early_cutoff_payload['best_update_count'])} "
        f"best_score={float(early_cutoff_payload['best_score']):.4f} "
        f"current_score={float(early_cutoff_payload['current_score']):.4f} "
        f"reasons={','.join(str(reason) for reason in reasons)}"
    )


def format_training_stopped_by_early_cutoff_message(metrics: Mapping[str, Any]) -> str:
    return (
        "Training stopped by early cutoff: "
        f"best_score={float(metrics.get('early_cutoff_best_score', 0.0)):.4f} "
        f"current_score={float(metrics.get('early_cutoff_current_score', 0.0)):.4f} "
        f"no_improvement_updates={int(float(metrics.get('early_cutoff_no_improvement_updates', 0.0)))} "
        f"consecutive_stall_evals={int(float(metrics.get('early_cutoff_consecutive_stall_evals', 0.0)))}"
    )


def apply_stall_monitor_to_dev_eval_summary(
    *,
    stack: StackConfig,
    training_paths: CurriculumGuardPaths,
    summary_payload: dict[str, Any],
    summary_path: Path | None = None,
) -> dict[str, Any] | None:
    if not dev_eval_is_authoritative(summary_payload):
        return None
    stall_monitor = update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=int(summary_payload["update_count"]),
        summary_payload=summary_payload,
    )
    if stall_monitor is None:
        return None
    summary_payload["stall_monitor"] = stall_monitor
    if summary_path is not None:
        write_json(summary_path, summary_payload)
    return stall_monitor


def update_early_cutoff(
    *,
    stack: StackConfig,
    training_paths: CurriculumGuardPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.early_cutoff.enabled:
        return None
    current_score = dev_eval_aggregate_score(summary_payload)
    if current_score is None:
        return None

    early_cutoff = curriculum.early_cutoff
    state_path = early_cutoff_state_path(training_paths)
    state = load_json_object(state_path, label="early cutoff state") if state_path.is_file() else {}
    previous_best_score = state.get("best_score")
    previous_best_update = state.get("best_update_count")
    previous_consecutive_stall = int(state.get("consecutive_stall_evals", 0))

    improved = False
    if isinstance(previous_best_score, (int, float)) and np.isfinite(float(previous_best_score)):
        best_score = float(previous_best_score)
        best_update_count = int(previous_best_update) if isinstance(previous_best_update, int) else int(update_count)
        if float(current_score) > best_score + float(early_cutoff.min_improvement):
            best_score = float(current_score)
            best_update_count = int(update_count)
            improved = True
    else:
        best_score = float(current_score)
        best_update_count = int(update_count)
        improved = True

    patience_reference_update = max(int(best_update_count), int(early_cutoff.warmup_updates))
    no_improvement_updates = max(0, int(update_count) - patience_reference_update)
    worst_stall_rate = dev_eval_worst_stall_rate(summary_payload)
    if worst_stall_rate is not None and worst_stall_rate >= float(early_cutoff.stall_rate_threshold):
        consecutive_stall_evals = previous_consecutive_stall + 1
    else:
        consecutive_stall_evals = 0

    reasons: list[str] = []
    if (
        int(early_cutoff.patience_updates) > 0
        and int(update_count) >= int(early_cutoff.warmup_updates)
        and no_improvement_updates >= int(early_cutoff.patience_updates)
    ):
        reasons.append("no_improvement")
    if int(early_cutoff.stall_patience_evals) > 0 and consecutive_stall_evals >= int(early_cutoff.stall_patience_evals):
        reasons.append("stall")

    payload = {
        "enabled": True,
        "update_count": int(update_count),
        "current_score": float(current_score),
        "best_score": float(best_score),
        "best_update_count": int(best_update_count),
        "improved": bool(improved),
        "min_improvement": float(early_cutoff.min_improvement),
        "warmup_updates": int(early_cutoff.warmup_updates),
        "patience_updates": int(early_cutoff.patience_updates),
        "no_improvement_updates": int(no_improvement_updates),
        "stall_patience_evals": int(early_cutoff.stall_patience_evals),
        "stall_rate_threshold": float(early_cutoff.stall_rate_threshold),
        "worst_stall_rate": None if worst_stall_rate is None else float(worst_stall_rate),
        "consecutive_stall_evals": int(consecutive_stall_evals),
        "should_stop": bool(reasons),
        "reasons": reasons,
    }
    write_json(state_path, payload)
    if reasons:
        append_early_cutoff_event(
            training_paths,
            {
                "format": "early_cutoff_event_v1",
                **payload,
            },
        )
    return payload
