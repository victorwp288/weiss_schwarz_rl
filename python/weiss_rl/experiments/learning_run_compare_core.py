from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DIAGNOSTIC_METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "actor_lag_max_update": ("actor_model_sync", "max_learner_to_actor_update_lag"),
    "chosen_attack_fraction_last_window": (
        "chosen_action_learning",
        "chosen_attack_train_fraction",
        "last_window_mean",
    ),
    "chosen_main_play_character_fraction_last_window": (
        "chosen_action_learning",
        "chosen_main_play_character_train_fraction",
        "last_window_mean",
    ),
    "chosen_nonpass_advantage_last_window": (
        "chosen_action_learning",
        "chosen_nonpass_train_advantage_mean",
        "last_window_mean",
    ),
    "chosen_pass_advantage_last_window": (
        "chosen_action_learning",
        "chosen_pass_train_advantage_mean",
        "last_window_mean",
    ),
    "latest_minus_best": ("periodic_dev_eval", "latest_minus_best"),
    "off_policy_max_target_behavior_train_logp_delta_abs_p99": (
        "off_policy",
        "max_target_behavior_train_logp_delta_abs_p99",
    ),
    "off_policy_max_vtrace_clip_rate": ("off_policy", "max_vtrace_clip_rate"),
    "off_policy_max_vtrace_rho_p99": ("off_policy", "max_vtrace_rho_p99"),
    "off_policy_max_vtrace_train_rho_p95": ("off_policy", "max_vtrace_train_rho_p95"),
    "reward_advantage_abs_mean_last_window": (
        "reward_scale",
        "advantage_abs_mean",
        "last_window_mean",
    ),
    "reward_abs_mean_max": ("reward_scale", "max_reward_abs_mean"),
    "target_abs_mean_max": ("reward_scale", "max_target_abs_mean"),
    "teacher_action_accuracy_last_window": ("teacher_action_accuracy", "last_window_mean"),
    "teacher_family_accuracy_last_window": ("teacher_family_accuracy", "last_window_mean"),
    "teacher_public_heuristic_coef_max": (
        "teacher_guidance",
        "max_teacher_public_heuristic_coef_active",
    ),
    "teacher_public_heuristic_loss_last_window": (
        "teacher_guidance",
        "teacher_public_heuristic_loss",
        "last_window_mean",
    ),
}

POLICY_ALIGNMENT_ANCHOR_ALIASES = {
    "B2 HeuristicPublic": "b2",
    "B3 HeuristicPublicAggro": "b3",
    "B4 HeuristicPublicControl": "b4",
}
POLICY_ALIGNMENT_FAMILIES = (
    "attack",
    "clock_from_hand",
    "climax_play",
    "level_up",
    "main_play_character",
)


def json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def dev_eval_records_from_training_log(run_dir: Path) -> list[dict[str, Any]]:
    payload = json_or_none(run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json")
    if payload is None:
        return []
    records: list[dict[str, Any]] = []
    for policy_id, summary in payload.items():
        if not isinstance(summary, dict):
            continue
        update_count = summary.get("update_count")
        aggregate_score = summary.get("aggregate_score")
        if not isinstance(update_count, int) or not isinstance(aggregate_score, int | float):
            continue
        anchor_scores = summary.get("anchor_scores")
        records.append(
            {
                "policy_id": str(policy_id),
                "update_count": int(update_count),
                "aggregate_score": float(aggregate_score),
                "anchor_scores": {
                    str(key): float(value)
                    for key, value in (anchor_scores if isinstance(anchor_scores, dict) else {}).items()
                    if isinstance(value, int | float)
                },
            }
        )
    return sorted(records, key=lambda record: int(record["update_count"]))


def dev_eval_records_from_eval_dirs(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dev_eval_dir = run_dir / "eval" / "dev_eval"
    if not dev_eval_dir.exists():
        return records
    for summary_path in sorted(dev_eval_dir.glob("update_*/summary.json")):
        summary = json_or_none(summary_path)
        if summary is None:
            continue
        update_count = summary.get("update_count")
        aggregate_score = summary.get("aggregate_score")
        if not isinstance(update_count, int) or not isinstance(aggregate_score, int | float):
            continue
        anchor_scores = summary.get("anchor_scores")
        records.append(
            {
                "policy_id": str(
                    summary.get("policy_id") or summary.get("focal_policy_id") or summary_path.parent.name
                ),
                "update_count": int(update_count),
                "aggregate_score": float(aggregate_score),
                "anchor_scores": {
                    str(key): float(value)
                    for key, value in (anchor_scores if isinstance(anchor_scores, dict) else {}).items()
                    if isinstance(value, int | float)
                },
            }
        )
    return sorted(records, key=lambda record: int(record["update_count"]))


def load_dev_eval_records(run_dir: Path) -> list[dict[str, Any]]:
    records = dev_eval_records_from_training_log(run_dir)
    return records if records else dev_eval_records_from_eval_dirs(run_dir)


def best_and_latest(records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not records:
        return None, None
    best = max(records, key=lambda record: float(record["aggregate_score"]))
    return best, records[-1]


def score_range(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "mean": sum(values) / len(values),
    }


def numeric_at_path(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if isinstance(cursor, bool) or not isinstance(cursor, int | float):
        return None
    return float(cursor)


def load_learning_progress_metrics(run_dir: Path) -> dict[str, float]:
    payload = json_or_none(run_dir / "diagnostics" / "learning_progress_summary.json")
    if payload is None:
        return {}
    metrics: dict[str, float] = {}
    for metric_name, path in DIAGNOSTIC_METRIC_PATHS.items():
        value = numeric_at_path(payload, path)
        if value is not None:
            metrics[metric_name] = value
    return metrics


def family_alignment_by_name(scope_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    family_summaries = scope_payload.get("reference_top_family_summaries")
    if not isinstance(family_summaries, list):
        return {}
    by_name: dict[str, dict[str, Any]] = {}
    for family_summary in family_summaries:
        if not isinstance(family_summary, dict):
            continue
        family_name = family_summary.get("family")
        if isinstance(family_name, str):
            by_name[family_name] = family_summary
    return by_name


def policy_alignment_metric_prefix(anchor_name: str) -> str | None:
    alias = POLICY_ALIGNMENT_ANCHOR_ALIASES.get(anchor_name)
    return None if alias is None else f"{alias}_focal"


def load_policy_alignment_metrics(run_dir: Path, *, update_count: int | None) -> dict[str, float]:
    if update_count is None:
        return {}
    summary = json_or_none(run_dir / "eval" / "dev_eval" / f"update_{int(update_count)}" / "summary.json")
    if summary is None:
        return {}
    anchors = summary.get("anchors")
    if not isinstance(anchors, dict):
        return {}

    metrics: dict[str, float] = {}
    for anchor_name, anchor_payload in anchors.items():
        if not isinstance(anchor_name, str) or not isinstance(anchor_payload, dict):
            continue
        prefix = policy_alignment_metric_prefix(anchor_name)
        if prefix is None:
            continue
        diagnostics = anchor_payload.get("policy_alignment_diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        scope_payload = diagnostics.get("focal_policy_turns")
        if not isinstance(scope_payload, dict):
            scope_payload = diagnostics.get("all_decisions")
        if not isinstance(scope_payload, dict):
            continue
        for source_name, metric_name in (
            ("model_matches_reference_top_action_rate", "overall_top_action_rate"),
            ("model_matches_reference_top_action_family_rate", "overall_top_family_rate"),
            ("model_mean_probability_on_reference_top_action", "overall_probability_on_reference_top_action"),
            (
                "model_mean_probability_on_reference_top_action_family",
                "overall_probability_on_reference_top_family",
            ),
        ):
            value = numeric_at_path(scope_payload, (source_name,))
            if value is not None:
                metrics[f"{prefix}_{metric_name}"] = value

        family_summaries = family_alignment_by_name(scope_payload)
        for family_name in POLICY_ALIGNMENT_FAMILIES:
            family_payload = family_summaries.get(family_name)
            if not isinstance(family_payload, dict):
                continue
            for source_name, metric_name in (
                ("model_matches_reference_top_action_rate", "top_action_rate"),
                ("model_mean_probability_on_reference_top_action", "probability_on_reference_top_action"),
            ):
                value = numeric_at_path(family_payload, (source_name,))
                if value is not None:
                    metrics[f"{prefix}_{family_name}_{metric_name}"] = value
            margin_mean = numeric_at_path(
                family_payload,
                ("model_reference_top_action_same_family_logit_margin_percentiles", "mean"),
            )
            if margin_mean is not None:
                metrics[f"{prefix}_{family_name}_same_family_margin_mean"] = margin_mean
    return metrics


def named_score_range(values: list[tuple[str, float]]) -> dict[str, float | int | str]:
    numeric_values = [value for _run_name, value in values]
    minimum = min(values, key=lambda item: item[1])
    maximum = max(values, key=lambda item: item[1])
    summary: dict[str, float | int | str] = {key: value for key, value in score_range(numeric_values).items()}
    summary.update({"min_run": minimum[0], "max_run": maximum[0]})
    return summary


def build_run_learning_comparison(
    run_dirs: list[Path],
    *,
    fragility_threshold: float = 0.25,
    anchor_fragility_threshold: float = 0.25,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        records = load_dev_eval_records(run_dir)
        best, latest = best_and_latest(records)
        runs.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "records": records,
                "best": best,
                "latest": latest,
                "latest_minus_best": (
                    None
                    if best is None or latest is None
                    else float(latest["aggregate_score"]) - float(best["aggregate_score"])
                ),
                "diagnostic_metrics": load_learning_progress_metrics(run_dir),
                "policy_alignment_update": None if best is None else int(best["update_count"]),
                "policy_alignment_metrics": load_policy_alignment_metrics(
                    run_dir,
                    update_count=None if best is None else int(best["update_count"]),
                ),
            }
        )

    update_values: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for run in runs:
        for record in run["records"]:
            update_values.setdefault(int(record["update_count"]), []).append((str(run["run_name"]), record))

    by_update: dict[str, Any] = {}
    warnings: list[str] = []
    for update_count, entries in sorted(update_values.items()):
        if len(entries) < 2:
            continue
        aggregate_values = [float(record["aggregate_score"]) for _run_name, record in entries]
        aggregate_summary = score_range(aggregate_values)
        anchor_names = sorted(
            {anchor_name for _run_name, record in entries for anchor_name in record.get("anchor_scores", {}).keys()}
        )
        anchor_summaries: dict[str, dict[str, float | int]] = {}
        for anchor_name in anchor_names:
            values = [
                float(record["anchor_scores"][anchor_name])
                for _run_name, record in entries
                if anchor_name in record.get("anchor_scores", {})
            ]
            if len(values) >= 2:
                anchor_summaries[anchor_name] = score_range(values)
        by_update[str(update_count)] = {
            "aggregate": aggregate_summary,
            "anchors": anchor_summaries,
            "runs": [
                {
                    "run_name": run_name,
                    "policy_id": record["policy_id"],
                    "aggregate_score": record["aggregate_score"],
                    "anchor_scores": record["anchor_scores"],
                }
                for run_name, record in entries
            ],
        }
        if float(aggregate_summary["range"]) >= float(fragility_threshold):
            warnings.append(
                f"aggregate seed/run fragility at update {update_count}: "
                f"range {aggregate_summary['range']:.4f} >= {fragility_threshold:.4f}"
            )
        for anchor_name, summary in anchor_summaries.items():
            if float(summary["range"]) >= float(anchor_fragility_threshold):
                warnings.append(
                    f"{anchor_name} seed/run fragility at update {update_count}: "
                    f"range {summary['range']:.4f} >= {anchor_fragility_threshold:.4f}"
                )

    diagnostic_metric_values: dict[str, list[tuple[str, float]]] = {}
    for run in runs:
        for metric_name, value in run["diagnostic_metrics"].items():
            diagnostic_metric_values.setdefault(metric_name, []).append((str(run["run_name"]), float(value)))
    diagnostic_metric_ranges = {
        metric_name: named_score_range(values)
        for metric_name, values in sorted(diagnostic_metric_values.items())
        if len(values) >= 2
    }
    policy_alignment_metric_values: dict[str, list[tuple[str, float]]] = {}
    for run in runs:
        for metric_name, value in run["policy_alignment_metrics"].items():
            policy_alignment_metric_values.setdefault(metric_name, []).append((str(run["run_name"]), float(value)))
    policy_alignment_metric_ranges = {
        metric_name: named_score_range(values)
        for metric_name, values in sorted(policy_alignment_metric_values.items())
        if len(values) >= 2
    }

    return {
        "run_count": len(runs),
        "runs": runs,
        "by_update": by_update,
        "diagnostic_metric_ranges": diagnostic_metric_ranges,
        "policy_alignment_metric_ranges": policy_alignment_metric_ranges,
        "thresholds": {
            "aggregate_fragility": float(fragility_threshold),
            "anchor_fragility": float(anchor_fragility_threshold),
        },
        "warnings": warnings,
    }


def write_learning_comparison_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DIAGNOSTIC_METRIC_PATHS",
    "POLICY_ALIGNMENT_ANCHOR_ALIASES",
    "POLICY_ALIGNMENT_FAMILIES",
    "best_and_latest",
    "build_run_learning_comparison",
    "dev_eval_records_from_eval_dirs",
    "dev_eval_records_from_training_log",
    "family_alignment_by_name",
    "json_or_none",
    "load_dev_eval_records",
    "load_learning_progress_metrics",
    "load_policy_alignment_metrics",
    "named_score_range",
    "numeric_at_path",
    "policy_alignment_metric_prefix",
    "score_range",
    "write_learning_comparison_json",
]
