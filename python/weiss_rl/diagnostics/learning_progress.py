from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from weiss_rl.diagnostics.learning_progress_artifacts import (
    _checkpoint_alias_integrity,
    _file_sha256_or_none,
    _final_eval_matrix_summaries,
    _final_eval_matrix_summary,
    _periodic_dev_eval_trend,
    _policy_id_from_checkpoint_record,
    _policy_update_map,
    _promotion_gate_summary,
    _read_mean_matrix,
    _read_numeric_matrix_payload,
    _row_mean_excluding_self,
    _run_relative_path,
    _update_from_promotion_gate_path,
)
from weiss_rl.diagnostics.learning_progress_guard import (
    DEFAULT_LEAGUE_GUARD_ANCHORS,
    _finite_float,
    _latest_periodic_anchor_scores,
    evaluate_league_guard,
)
from weiss_rl.diagnostics.learning_progress_metrics import (
    _fraction_values,
    _last_window_mean,
    _mean,
    _numeric_by_update,
    _numeric_value,
    _numeric_values,
    _paired_update_values,
    _pearson_correlation,
    _ratio_values,
    _sum_fraction_values,
    _window_summary,
    build_training_log_summary_sections,
)

_FINAL_EVAL_BEST_ROW_WARN_MARGIN = 0.0
__all__ = [
    "DEFAULT_LEAGUE_GUARD_ANCHORS",
    "_checkpoint_alias_integrity",
    "_file_sha256_or_none",
    "_finite_float",
    "_final_eval_matrix_summaries",
    "_final_eval_matrix_summary",
    "_periodic_dev_eval_trend",
    "_policy_id_from_checkpoint_record",
    "_policy_update_map",
    "_promotion_gate_summary",
    "_latest_periodic_anchor_scores",
    "_fraction_values",
    "_last_window_mean",
    "_mean",
    "_numeric_by_update",
    "_numeric_value",
    "_numeric_values",
    "_paired_update_values",
    "_pearson_correlation",
    "_ratio_values",
    "_read_mean_matrix",
    "_read_numeric_matrix_payload",
    "_row_mean_excluding_self",
    "_run_relative_path",
    "_sum_fraction_values",
    "_update_from_promotion_gate_path",
    "_window_summary",
    "build_learning_progress_summary",
    "build_training_log_summary_sections",
    "evaluate_league_guard",
    "main",
]


def _json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def build_learning_progress_summary(run_dir: Path) -> dict[str, Any]:
    metrics = _jsonl_records(run_dir / "training" / "logs" / "training_metrics.jsonl")
    scalars = _jsonl_records(run_dir / "training" / "logs" / "scalars.jsonl")
    performance = _jsonl_records(run_dir / "training" / "logs" / "performance.jsonl")
    checkpoint_tracker = _json_or_none(run_dir / "training" / "checkpoints" / "checkpoint_tracker.json") or {}
    checkpoint_alias_integrity = _checkpoint_alias_integrity(run_dir, checkpoint_tracker)
    best_checkpoint = cast(
        Mapping[str, Any],
        checkpoint_tracker.get("best") if isinstance(checkpoint_tracker.get("best"), dict) else {},
    )
    final_eval_matrix = _final_eval_matrix_summary(run_dir, checkpoint_best=best_checkpoint)
    final_eval_matrices = _final_eval_matrix_summaries(run_dir, checkpoint_best=best_checkpoint)
    periodic_dev_eval = run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json"
    dev_eval_trend = _periodic_dev_eval_trend(periodic_dev_eval)
    promotion_gate = _promotion_gate_summary(run_dir)
    matrix = _read_mean_matrix(run_dir / "eval" / "final_eval" / "matrices" / "mean.csv")
    policy_updates = _policy_update_map(run_dir)
    training_log_sections = build_training_log_summary_sections(
        metrics=metrics,
        scalars=scalars,
        performance=performance,
        promotion_gate=promotion_gate,
    )

    warnings = list(training_log_sections.warnings)
    if isinstance(best_checkpoint, dict) and best_checkpoint.get("metric_kind") == "training_loss":
        warnings.append("checkpoint best was selected by scalar training_loss, not dev-eval quality")
    if checkpoint_alias_integrity["latest_alias_matches_source"] is False:
        warnings.append("latest checkpoint alias file does not match its tracker source checkpoint")
    if checkpoint_alias_integrity["observed_best_alias_matches_source"] is False:
        warnings.append("observed_best checkpoint alias file does not match its tracker source checkpoint")
    if not periodic_dev_eval.exists():
        warnings.append("periodic dev-eval summaries are absent; learning quality was not monitored during training")
    elif dev_eval_trend["latest_minus_best"] is not None and float(dev_eval_trend["latest_minus_best"]) < -0.05:
        warnings.append("latest periodic dev-eval aggregate is more than 0.05 below an earlier checkpoint")

    best_row_mean = final_eval_matrix["best_row_mean_excluding_self"]
    checkpoint_best_row_mean = final_eval_matrix["checkpoint_best_row_mean_excluding_self"]
    if (
        isinstance(best_row_mean, int | float)
        and isinstance(checkpoint_best_row_mean, int | float)
        and float(best_row_mean) - float(checkpoint_best_row_mean) > _FINAL_EVAL_BEST_ROW_WARN_MARGIN
    ):
        warnings.append(
            "periodic-dev selected best checkpoint is not the strongest row in final-eval confirmation matrix"
        )

    comparisons: dict[str, float] = {}
    for focal, opponent in (
        ("policy_000004", "policy_000006"),
        ("policy_000004", "B1 NoLeague baseline"),
        ("B1 NoLeague baseline", "policy_000006"),
        ("policy_000004", "B2 HeuristicPublic"),
        ("policy_000006", "B2 HeuristicPublic"),
        ("B1 NoLeague baseline", "B2 HeuristicPublic"),
    ):
        if focal in matrix and opponent in matrix[focal]:
            comparisons[f"{focal}__vs__{opponent}"] = matrix[focal][opponent]

    update_counts = _numeric_values(metrics, "update_count")
    return {
        "run_dir": run_dir.resolve().as_posix(),
        "training_record_count": len(metrics),
        "update_min": None if not update_counts else int(min(update_counts)),
        "update_max": None if not update_counts else int(max(update_counts)),
        **training_log_sections.sections,
        "periodic_dev_eval": dev_eval_trend,
        "promotion_gate": promotion_gate,
        "checkpoint_best": best_checkpoint,
        "checkpoint_alias_integrity": checkpoint_alias_integrity,
        "policy_updates": policy_updates,
        "final_eval_matrix": final_eval_matrix,
        "final_eval_matrices": final_eval_matrices,
        "final_eval_mean_matrix_subset": comparisons,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize whether an existing thesis run is visibly learning")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--league-guard",
        action="store_true",
        help="Exit nonzero when a guarded league probe violates promotion/anchor health thresholds.",
    )
    parser.add_argument("--guard-required-anchor", action="append", default=None)
    parser.add_argument("--guard-min-latest-anchor-score", type=float, default=0.45)
    parser.add_argument("--guard-max-latest-drop", type=float, default=0.05)
    parser.add_argument("--guard-require-promotion-pass-after-attempts", type=int, default=3)
    parser.add_argument("--guard-max-consecutive-promotion-failures", type=int, default=3)
    parser.add_argument("--guard-max-vtrace-rho-p99", type=float, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary = build_learning_progress_summary(run_dir)
    league_guard = None
    if args.league_guard:
        league_guard = evaluate_league_guard(
            summary,
            required_anchors=tuple(args.guard_required_anchor or DEFAULT_LEAGUE_GUARD_ANCHORS),
            min_latest_anchor_score=float(args.guard_min_latest_anchor_score),
            max_latest_drop=float(args.guard_max_latest_drop),
            require_promotion_pass_after_attempts=int(args.guard_require_promotion_pass_after_attempts),
            max_consecutive_promotion_failures=int(args.guard_max_consecutive_promotion_failures),
            max_vtrace_rho_p99=args.guard_max_vtrace_rho_p99,
        )
        summary["league_guard"] = league_guard
    output_path = args.output_json
    if output_path is None:
        output_path = run_dir / "diagnostics" / "learning_progress_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    if league_guard is not None and not bool(league_guard["passed"]):
        failure_codes = ",".join(str(failure.get("code", "unknown")) for failure in league_guard["failures"])
        raise SystemExit(f"league guard failed: {failure_codes}")


if __name__ == "__main__":
    main()
