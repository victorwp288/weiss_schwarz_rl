from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_OFF_POLICY_RHO_WARN_THRESHOLD = 10.0
_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD = 10.0
_VTRACE_CLIP_RATE_WARN_THRESHOLD = 0.5
_LEARNER_ACTOR_LAG_WARN_THRESHOLD = 25.0
_FINAL_EVAL_BEST_ROW_WARN_MARGIN = 0.0
_MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD = 1.0
_TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD = 1.0
_MULLIGAN_SELECT_SHARE_WARN_THRESHOLD = 0.8
_TEACHER_SUPPORTED_WARN_THRESHOLD = 0.05
DEFAULT_LEAGUE_GUARD_ANCHORS = (
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
)


def _json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _run_relative_path(run_dir: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _file_sha256_or_none(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _numeric_values(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _numeric_value(record, key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _numeric_value(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if not isinstance(value, int | float):
        custom_metrics = record.get("custom_metrics")
        if isinstance(custom_metrics, dict):
            value = custom_metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def _fraction_values(records: Iterable[dict[str, Any]], numerator_key: str, denominator_key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = _numeric_value(record, numerator_key)
        denominator = _numeric_value(record, denominator_key)
        if numerator is None or denominator is None or denominator <= 0.0:
            continue
        values.append(float(numerator) / float(denominator))
    return values


def _ratio_values(
    records: Iterable[dict[str, Any]],
    numerator_key: str,
    denominator_keys: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = _numeric_value(record, numerator_key)
        if numerator is None:
            continue
        denominator = 0.0
        complete = True
        for key in denominator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            denominator += float(value)
        if not complete or denominator <= 0.0:
            continue
        values.append(float(numerator) / denominator)
    return values


def _sum_fraction_values(
    records: Iterable[dict[str, Any]],
    numerator_keys: tuple[str, ...],
    denominator_keys: tuple[str, ...],
) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = 0.0
        denominator = 0.0
        complete = True
        for key in numerator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            numerator += float(value)
        if not complete:
            continue
        for key in denominator_keys:
            value = _numeric_value(record, key)
            if value is None:
                complete = False
                break
            denominator += float(value)
        if complete and denominator > 0.0:
            values.append(numerator / denominator)
    return values


def _numeric_by_update(records: Iterable[dict[str, Any]], key: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for record in records:
        update_count = record.get("update_count")
        value = _numeric_value(record, key)
        if isinstance(update_count, int) and value is not None:
            values[int(update_count)] = float(value)
    return values


def _paired_update_values(
    left_records: Iterable[dict[str, Any]],
    left_key: str,
    right_records: Iterable[dict[str, Any]],
    right_key: str,
) -> list[tuple[float, float]]:
    left = _numeric_by_update(left_records, left_key)
    right = _numeric_by_update(right_records, right_key)
    return [(left[update], right[update]) for update in sorted(left.keys() & right.keys())]


def _pearson_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left_values = [left for left, _right in pairs]
    right_values = [right for _left, right in pairs]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    covariance = sum(left * right for left, right in zip(left_centered, right_centered, strict=True))
    return float(covariance / ((left_ss * right_ss) ** 0.5))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _window_summary(values: list[float], *, window: int) -> dict[str, float | None]:
    if not values:
        return {"first": None, "last": None, "first_window_mean": None, "last_window_mean": None}
    first_window = values[:window]
    last_window = values[-window:]
    return {
        "first": values[0],
        "last": values[-1],
        "first_window_mean": _mean(first_window),
        "last_window_mean": _mean(last_window),
    }


def _read_mean_matrix(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matrix: dict[str, dict[str, float]] = {}
    for row in rows:
        focal = str(row.get("focal_policy_id", "")).strip()
        if not focal:
            continue
        matrix[focal] = {}
        for key, value in row.items():
            if key == "focal_policy_id" or value is None or value == "":
                continue
            matrix[focal][key] = float(value)
    return matrix


def _read_numeric_matrix_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"row_policy_ids": [], "column_policy_ids": [], "values": []}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        column_policy_ids = [name for name in (reader.fieldnames or []) if name != "focal_policy_id"]
        row_policy_ids: list[str] = []
        values: list[list[float | None]] = []
        for row in reader:
            focal = str(row.get("focal_policy_id", "")).strip()
            if not focal:
                continue
            row_policy_ids.append(focal)
            row_values: list[float | None] = []
            for policy_id in column_policy_ids:
                raw_value = row.get(policy_id)
                row_values.append(None if raw_value is None or raw_value == "" else float(raw_value))
            values.append(row_values)
    return {"row_policy_ids": row_policy_ids, "column_policy_ids": column_policy_ids, "values": values}


def _row_mean_excluding_self(matrix_payload: Mapping[str, Any]) -> dict[str, float]:
    rows = matrix_payload.get("row_policy_ids")
    columns = matrix_payload.get("column_policy_ids")
    values = matrix_payload.get("values")
    if not isinstance(rows, list) or not isinstance(columns, list) or not isinstance(values, list):
        return {}
    result: dict[str, float] = {}
    for row_index, policy_id in enumerate(rows):
        if not isinstance(policy_id, str) or row_index >= len(values):
            continue
        row_values = values[row_index]
        if not isinstance(row_values, list):
            continue
        usable: list[float] = []
        for column_index, raw_value in enumerate(row_values):
            if column_index >= len(columns) or columns[column_index] == policy_id:
                continue
            if isinstance(raw_value, int | float):
                usable.append(float(raw_value))
        if usable:
            result[policy_id] = float(sum(usable) / len(usable))
    return result


def _policy_id_from_checkpoint_record(record: Mapping[str, Any]) -> str | None:
    policy_id = record.get("policy_id")
    if isinstance(policy_id, str) and policy_id.strip():
        return policy_id.strip()
    policy_version = record.get("policy_version")
    if isinstance(policy_version, int):
        return f"policy_{policy_version:06d}"
    return None


def _final_eval_matrix_summary(
    run_dir: Path,
    *,
    checkpoint_best: Mapping[str, Any],
    eval_subdir: str = "final_eval",
) -> dict[str, Any]:
    matrix_dir = run_dir / "eval" / eval_subdir / "matrices"
    mean_payload = _read_numeric_matrix_payload(matrix_dir / "mean.csv")
    row_strength = _row_mean_excluding_self(mean_payload)
    best_row_policy_id = None
    best_row_mean = None
    if row_strength:
        best_row_policy_id, best_row_mean = max(row_strength.items(), key=lambda item: item[1])
    checkpoint_best_policy_id = _policy_id_from_checkpoint_record(checkpoint_best)
    checkpoint_best_row_mean = None
    if checkpoint_best_policy_id is not None:
        checkpoint_best_row_mean = row_strength.get(checkpoint_best_policy_id)
    policy_updates = _policy_update_map(run_dir)
    return {
        "eval_subdir": eval_subdir,
        "mean": mean_payload,
        "wins": _read_numeric_matrix_payload(matrix_dir / "wins.csv"),
        "games": _read_numeric_matrix_payload(matrix_dir / "games.csv"),
        "prob_gt_half": _read_numeric_matrix_payload(matrix_dir / "prob_gt_half.csv"),
        "row_mean_excluding_self": row_strength,
        "best_row_policy_id": best_row_policy_id,
        "best_row_update": None if best_row_policy_id is None else policy_updates.get(best_row_policy_id),
        "best_row_mean_excluding_self": best_row_mean,
        "checkpoint_best_policy_id": checkpoint_best_policy_id,
        "checkpoint_best_row_mean_excluding_self": checkpoint_best_row_mean,
    }


def _final_eval_matrix_summaries(run_dir: Path, *, checkpoint_best: Mapping[str, Any]) -> dict[str, Any]:
    eval_root = run_dir / "eval"
    if not eval_root.exists():
        return {}
    summaries: dict[str, Any] = {}
    for eval_dir in sorted(
        path for path in eval_root.iterdir() if path.is_dir() and path.name.startswith("final_eval")
    ):
        if not (eval_dir / "matrices" / "mean.csv").exists():
            continue
        summaries[eval_dir.name] = _final_eval_matrix_summary(
            run_dir,
            checkpoint_best=checkpoint_best,
            eval_subdir=eval_dir.name,
        )
    return summaries


def _policy_update_map(run_dir: Path) -> dict[str, int]:
    registry = _json_or_none(run_dir / "training" / "snapshots" / "registry.json") or {}
    snapshots = registry.get("snapshots")
    if not isinstance(snapshots, list):
        return {}
    result: dict[str, int] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        policy_id = str(snapshot.get("policy_id", "")).strip()
        update = snapshot.get("update_count", snapshot.get("update"))
        if policy_id and isinstance(update, int):
            result[policy_id] = int(update)
    return result


def _checkpoint_alias_integrity(run_dir: Path, checkpoint_tracker: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for alias_name in ("latest", "best", "observed_best"):
        record = checkpoint_tracker.get(alias_name)
        if not isinstance(record, Mapping):
            result[f"{alias_name}_alias_path"] = None
            result[f"{alias_name}_source_checkpoint_path"] = None
            result[f"{alias_name}_alias_matches_source"] = None
            continue
        alias_path = _run_relative_path(run_dir, record.get("alias_path"))
        source_path = _run_relative_path(run_dir, record.get("source_checkpoint_path"))
        alias_hash = _file_sha256_or_none(alias_path)
        source_hash = _file_sha256_or_none(source_path)
        result[f"{alias_name}_alias_path"] = record.get("alias_path")
        result[f"{alias_name}_source_checkpoint_path"] = record.get("source_checkpoint_path")
        result[f"{alias_name}_alias_matches_source"] = (
            None if alias_hash is None or source_hash is None else alias_hash == source_hash
        )
    return result


def _periodic_dev_eval_trend(path: Path) -> dict[str, Any]:
    payload = _json_or_none(path)
    if payload is None:
        return {
            "records": [],
            "best_update": None,
            "best_aggregate_score": None,
            "last_update": None,
            "last_aggregate_score": None,
            "latest_minus_best": None,
            "non_monotonic_drop_count": 0,
        }
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
                "anchor_scores": anchor_scores if isinstance(anchor_scores, dict) else {},
            }
        )
    records.sort(key=lambda record: int(record["update_count"]))
    if not records:
        return {
            "records": [],
            "best_update": None,
            "best_aggregate_score": None,
            "last_update": None,
            "last_aggregate_score": None,
            "latest_minus_best": None,
            "non_monotonic_drop_count": 0,
        }
    best = max(records, key=lambda record: float(record["aggregate_score"]))
    last = records[-1]
    drop_count = 0
    best_so_far = float(records[0]["aggregate_score"])
    for record in records[1:]:
        current = float(record["aggregate_score"])
        if current < best_so_far:
            drop_count += 1
        best_so_far = max(best_so_far, current)
    return {
        "records": records,
        "best_update": int(best["update_count"]),
        "best_aggregate_score": float(best["aggregate_score"]),
        "last_update": int(last["update_count"]),
        "last_aggregate_score": float(last["aggregate_score"]),
        "latest_minus_best": float(last["aggregate_score"]) - float(best["aggregate_score"]),
        "non_monotonic_drop_count": int(drop_count),
    }


def _update_from_promotion_gate_path(path: Path) -> int:
    parent_name = path.parent.name
    if parent_name.startswith("update_"):
        suffix = parent_name.removeprefix("update_")
        if suffix.isdigit():
            return int(suffix)
    return -1


def _promotion_gate_summary(run_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    gate_paths = sorted(
        (run_dir / "eval" / "promotion_gate").glob("update_*/promotion_gate.json"),
        key=_update_from_promotion_gate_path,
    )
    for gate_path in gate_paths:
        payload = _json_or_none(gate_path)
        if payload is None:
            continue
        decision = payload.get("decision")
        passed = bool(decision.get("passed")) if isinstance(decision, Mapping) else False
        reasons = decision.get("reasons") if isinstance(decision, Mapping) else []
        reason_codes = [
            str(reason.get("code"))
            for reason in reasons
            if isinstance(reason, Mapping) and isinstance(reason.get("code"), str)
        ]
        anchor_means: dict[str, float] = {}
        anchors = payload.get("anchors")
        if isinstance(anchors, list):
            for anchor in anchors:
                if not isinstance(anchor, Mapping):
                    continue
                anchor_name = anchor.get("anchor_name")
                posterior = anchor.get("posterior")
                mean = posterior.get("mean") if isinstance(posterior, Mapping) else None
                if isinstance(anchor_name, str) and isinstance(mean, int | float):
                    anchor_means[anchor_name] = float(mean)
        overall = payload.get("overall_posterior")
        records.append(
            {
                "update_count": _update_from_promotion_gate_path(gate_path),
                "focal_policy_id": payload.get("focal_policy_id"),
                "passed": passed,
                "reason_codes": reason_codes,
                "overall_mean": overall.get("mean") if isinstance(overall, Mapping) else None,
                "overall_prob_gt_target": overall.get("prob_gt_target") if isinstance(overall, Mapping) else None,
                "anchor_means": anchor_means,
                "summary_path": gate_path.as_posix(),
            }
        )
    passed_records = [record for record in records if bool(record["passed"])]
    failed_records = [record for record in records if not bool(record["passed"])]
    consecutive_failures = 0
    for record in reversed(records):
        if bool(record["passed"]):
            break
        consecutive_failures += 1
    latest = records[-1] if records else None
    return {
        "records": records,
        "attempt_count": len(records),
        "passed_count": len(passed_records),
        "failed_count": len(failed_records),
        "first_pass_update": None if not passed_records else int(passed_records[0]["update_count"]),
        "latest_update": None if latest is None else int(latest["update_count"]),
        "latest_passed": None if latest is None else bool(latest["passed"]),
        "latest_reason_codes": [] if latest is None else list(latest["reason_codes"]),
        "consecutive_failure_count": consecutive_failures,
    }


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _latest_periodic_anchor_scores(summary: Mapping[str, Any]) -> dict[str, float]:
    periodic = summary.get("periodic_dev_eval")
    if not isinstance(periodic, Mapping):
        return {}
    records = periodic.get("records")
    if not isinstance(records, list) or not records:
        return {}
    latest_record = records[-1]
    if not isinstance(latest_record, Mapping):
        return {}
    anchor_scores = latest_record.get("anchor_scores")
    if not isinstance(anchor_scores, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in anchor_scores.items():
        score = _finite_float(value)
        if score is not None:
            result[str(key)] = score
    return result


def evaluate_league_guard(
    summary: Mapping[str, Any],
    *,
    required_anchors: Iterable[str] = DEFAULT_LEAGUE_GUARD_ANCHORS,
    min_latest_anchor_score: float | None = 0.45,
    max_latest_drop: float | None = 0.05,
    require_promotion_pass_after_attempts: int | None = 3,
    max_consecutive_promotion_failures: int | None = 3,
    max_vtrace_rho_p99: float | None = None,
) -> dict[str, Any]:
    """Evaluate machine-actionable gates for guarded league probes."""

    failures: list[dict[str, Any]] = []
    required_anchor_tuple = tuple(str(anchor) for anchor in required_anchors)
    anchor_scores = _latest_periodic_anchor_scores(summary)
    if min_latest_anchor_score is not None:
        for anchor in required_anchor_tuple:
            score = anchor_scores.get(anchor)
            if score is None:
                failures.append({"code": "missing_latest_anchor_score", "anchor": anchor})
            elif score < float(min_latest_anchor_score):
                failures.append(
                    {
                        "code": "latest_anchor_below_threshold",
                        "anchor": anchor,
                        "observed": score,
                        "threshold": float(min_latest_anchor_score),
                    }
                )
    periodic = summary.get("periodic_dev_eval")
    latest_minus_best = periodic.get("latest_minus_best") if isinstance(periodic, Mapping) else None
    latest_drop = _finite_float(latest_minus_best)
    if max_latest_drop is not None and latest_drop is not None and latest_drop < -float(max_latest_drop):
        failures.append(
            {
                "code": "latest_periodic_drop_exceeded",
                "observed": latest_drop,
                "threshold": -float(max_latest_drop),
            }
        )
    promotion_gate = summary.get("promotion_gate")
    if isinstance(promotion_gate, Mapping):
        attempt_count = promotion_gate.get("attempt_count")
        passed_count = promotion_gate.get("passed_count")
        consecutive_failure_count = promotion_gate.get("consecutive_failure_count")
        if (
            require_promotion_pass_after_attempts is not None
            and isinstance(attempt_count, int)
            and isinstance(passed_count, int)
            and attempt_count >= int(require_promotion_pass_after_attempts)
            and passed_count <= 0
        ):
            failures.append(
                {
                    "code": "promotion_gate_no_pass_after_attempts",
                    "attempt_count": attempt_count,
                    "passed_count": passed_count,
                    "threshold": int(require_promotion_pass_after_attempts),
                }
            )
        if (
            max_consecutive_promotion_failures is not None
            and isinstance(consecutive_failure_count, int)
            and consecutive_failure_count >= int(max_consecutive_promotion_failures)
        ):
            failures.append(
                {
                    "code": "promotion_gate_consecutive_failures_exceeded",
                    "observed": consecutive_failure_count,
                    "threshold": int(max_consecutive_promotion_failures),
                }
            )
    off_policy = summary.get("off_policy")
    max_rho_p99 = off_policy.get("max_vtrace_rho_p99") if isinstance(off_policy, Mapping) else None
    max_train_rho_p99 = off_policy.get("max_vtrace_train_rho_p99") if isinstance(off_policy, Mapping) else None
    max_train_rho_p95 = off_policy.get("max_vtrace_train_rho_p95") if isinstance(off_policy, Mapping) else None
    max_rho_p99_float = _finite_float(max_rho_p99)
    max_train_tail_float = _finite_float(max_train_rho_p99)
    if max_train_tail_float is None:
        max_train_tail_float = _finite_float(max_train_rho_p95)
    guard_tail_float = max_train_tail_float if max_train_tail_float is not None else max_rho_p99_float
    if max_vtrace_rho_p99 is not None and guard_tail_float is not None and guard_tail_float > float(max_vtrace_rho_p99):
        failures.append(
            {
                "code": "vtrace_train_rho_tail_exceeded"
                if max_train_tail_float is not None
                else "vtrace_rho_p99_exceeded",
                "observed": guard_tail_float,
                "raw_vtrace_rho_p99": max_rho_p99_float,
                "train_vtrace_rho_tail": max_train_tail_float,
                "threshold": float(max_vtrace_rho_p99),
            }
        )
    return {
        "kind": "league_guard_v1",
        "passed": not failures,
        "failures": failures,
        "required_anchors": list(required_anchor_tuple),
        "latest_anchor_scores": anchor_scores,
        "min_latest_anchor_score": min_latest_anchor_score,
        "max_latest_drop": max_latest_drop,
        "require_promotion_pass_after_attempts": require_promotion_pass_after_attempts,
        "max_consecutive_promotion_failures": max_consecutive_promotion_failures,
        "max_vtrace_rho_p99": max_vtrace_rho_p99,
        "vtrace_guard_tail_source": "train" if max_train_tail_float is not None else "raw",
    }


def build_learning_progress_summary(run_dir: Path) -> dict[str, Any]:
    metrics = _jsonl_records(run_dir / "training" / "logs" / "training_metrics.jsonl")
    scalars = _jsonl_records(run_dir / "training" / "logs" / "scalars.jsonl")
    performance = _jsonl_records(run_dir / "training" / "logs" / "performance.jsonl")
    checkpoint_tracker = _json_or_none(run_dir / "training" / "checkpoints" / "checkpoint_tracker.json") or {}
    checkpoint_alias_integrity = _checkpoint_alias_integrity(run_dir, checkpoint_tracker)
    best_checkpoint = checkpoint_tracker.get("best") if isinstance(checkpoint_tracker.get("best"), dict) else {}
    final_eval_matrix = _final_eval_matrix_summary(run_dir, checkpoint_best=best_checkpoint)
    final_eval_matrices = _final_eval_matrix_summaries(run_dir, checkpoint_best=best_checkpoint)
    periodic_dev_eval = run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json"
    dev_eval_trend = _periodic_dev_eval_trend(periodic_dev_eval)
    promotion_gate = _promotion_gate_summary(run_dir)
    matrix = _read_mean_matrix(run_dir / "eval" / "final_eval" / "matrices" / "mean.csv")
    policy_updates = _policy_update_map(run_dir)
    records_for_route = scalars + performance
    records_for_learning = metrics + scalars
    actor_heuristic_values = _numeric_values(records_for_route, "actor_heuristic_fraction_active")
    heuristic_mix_values = _numeric_values(records_for_route, "heuristic_public_mix_fraction_active")
    pfsp_pool_size_values = _numeric_values(records_for_route, "pfsp_pool_size")
    pfsp_champion_pool_size_values = _numeric_values(records_for_route, "pfsp_champion_pool_size")
    pfsp_recent_pool_size_values = _numeric_values(records_for_route, "pfsp_recent_pool_size")
    pfsp_hard_negative_pool_size_values = _numeric_values(records_for_route, "pfsp_hard_negative_pool_size")
    pfsp_quarantined_opponent_values = _numeric_values(records_for_route, "pfsp_quarantined_opponents")
    pfsp_snapshot_env_fraction_values = _sum_fraction_values(
        records_for_route,
        (
            "pfsp_champion_envs",
            "pfsp_recent_envs",
            "pfsp_hard_negative_envs",
            "pfsp_warmup_snapshot_envs",
        ),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    pfsp_recent_env_fraction_values = _sum_fraction_values(
        records_for_route,
        ("pfsp_recent_envs",),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    pfsp_champion_env_fraction_values = _sum_fraction_values(
        records_for_route,
        ("pfsp_champion_envs",),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    pfsp_hard_negative_env_fraction_values = _sum_fraction_values(
        records_for_route,
        ("pfsp_hard_negative_envs",),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    pfsp_warmup_snapshot_env_fraction_values = _sum_fraction_values(
        records_for_route,
        ("pfsp_warmup_snapshot_envs",),
        ("pfsp_sampled_envs", "pfsp_mirror_envs"),
    )
    policy_version_lag_p50_values = _numeric_values(records_for_route, "policy_version_lag_p50")
    policy_version_lag_p90_values = _numeric_values(records_for_route, "policy_version_lag_p90")
    learner_actor_update_lag_p50_values = _numeric_values(records_for_route, "learner_actor_update_lag_p50")
    learner_actor_update_lag_p90_values = _numeric_values(records_for_route, "learner_actor_update_lag_p90")
    league_update_lag_values = _numeric_values(records_for_route, "league_update_lag")
    actor_lag_warning_values = (
        learner_actor_update_lag_p90_values or league_update_lag_values or policy_version_lag_p90_values
    )
    if learner_actor_update_lag_p90_values:
        actor_lag_warning_source = "learner_actor_update_lag_p90"
    elif league_update_lag_values:
        actor_lag_warning_source = "league_update_lag"
    else:
        actor_lag_warning_source = "policy_version_lag_p90"
    stale_policy_pairs = {
        "vtrace_rho_mean": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_rho_mean",
        ),
        "vtrace_rho_p99": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_rho_p99",
        ),
        "vtrace_train_rho_p95": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_train_rho_p95",
        ),
        "vtrace_train_rho_p99": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_train_rho_p99",
        ),
        "vtrace_clip_rate": _paired_update_values(
            records_for_route,
            actor_lag_warning_source,
            metrics,
            "vtrace_clip_rate",
        ),
    }
    vtrace_rho_values = _numeric_values(metrics, "vtrace_rho_mean")
    vtrace_rho_p99_values = _numeric_values(metrics, "vtrace_rho_p99")
    vtrace_train_rho_values = _numeric_values(metrics, "vtrace_train_rho_mean")
    vtrace_train_rho_p95_values = _numeric_values(metrics, "vtrace_train_rho_p95")
    vtrace_train_rho_p99_values = _numeric_values(metrics, "vtrace_train_rho_p99")
    vtrace_clip_rate_values = _numeric_values(metrics, "vtrace_clip_rate")
    logp_delta_abs_values = _numeric_values(metrics, "target_behavior_logp_delta_abs_mean")
    logp_delta_abs_p99_values = _numeric_values(metrics, "target_behavior_logp_delta_abs_p99")
    train_logp_delta_abs_values = _numeric_values(metrics, "target_behavior_train_logp_delta_abs_mean")
    train_logp_delta_abs_p99_values = _numeric_values(metrics, "target_behavior_train_logp_delta_abs_p99")
    reward_mean_values = _numeric_values(metrics, "reward_mean")
    reward_abs_values = _numeric_values(metrics, "reward_abs_mean")
    reward_std_values = _numeric_values(metrics, "reward_std")
    reward_nonzero_values = _numeric_values(metrics, "reward_nonzero_fraction")
    reward_positive_values = _numeric_values(metrics, "reward_positive_fraction")
    reward_negative_values = _numeric_values(metrics, "reward_negative_fraction")
    advantage_abs_values = _numeric_values(metrics, "advantage_abs_mean")
    target_abs_values = _numeric_values(metrics, "target_abs_mean")
    chosen_pass_train_fraction_values = _numeric_values(metrics, "chosen_pass_train_fraction")
    chosen_pass_train_advantage_values = _numeric_values(metrics, "chosen_pass_train_advantage_mean")
    chosen_nonpass_train_advantage_values = _numeric_values(metrics, "chosen_nonpass_train_advantage_mean")
    chosen_mulligan_confirm_train_fraction_values = _numeric_values(
        metrics,
        "chosen_mulligan_confirm_train_fraction",
    )
    chosen_mulligan_select_train_fraction_values = _numeric_values(metrics, "chosen_mulligan_select_train_fraction")
    chosen_mulligan_confirm_train_advantage_values = _numeric_values(
        metrics,
        "chosen_mulligan_confirm_train_advantage_mean",
    )
    chosen_mulligan_select_train_advantage_values = _numeric_values(
        metrics,
        "chosen_mulligan_select_train_advantage_mean",
    )
    chosen_mulligan_select_share_values = _ratio_values(
        metrics,
        "chosen_mulligan_select_train_fraction",
        ("chosen_mulligan_select_train_fraction", "chosen_mulligan_confirm_train_fraction"),
    )
    chosen_play_train_fraction_values = _numeric_values(metrics, "chosen_main_play_character_train_fraction")
    chosen_attack_train_fraction_values = _numeric_values(metrics, "chosen_attack_train_fraction")
    teacher_public_heuristic_coef_active_values = _numeric_values(
        records_for_learning,
        "teacher_public_heuristic_coef_active",
    )
    teacher_hand_coef_active_values = _numeric_values(records_for_learning, "teacher_hand_coef_active")
    teacher_aux_loss_values = _numeric_values(records_for_learning, "teacher_aux_loss")
    teacher_main_play_slot_accuracy_values = _numeric_values(
        records_for_learning,
        "teacher_main_play_character_slot_accuracy",
    )
    teacher_hand_accuracy_values = _numeric_values(records_for_learning, "teacher_hand_accuracy")
    teacher_main_play_hand_accuracy_values = _numeric_values(
        records_for_learning,
        "teacher_main_play_character_hand_accuracy",
    )
    teacher_clock_hand_accuracy_values = _numeric_values(records_for_learning, "teacher_clock_from_hand_accuracy")
    teacher_hand_loss_values = _numeric_values(records_for_learning, "teacher_hand_loss")
    teacher_hand_supported_values = _numeric_values(records_for_learning, "teacher_hand_supported_fraction")
    teacher_same_family_action_accuracy_values = _numeric_values(
        records_for_learning,
        "teacher_same_family_action_accuracy",
    )
    teacher_same_family_main_play_accuracy_values = _numeric_values(
        records_for_learning,
        "teacher_same_family_main_play_character_accuracy",
    )
    teacher_action_margin_mean_values = _numeric_values(records_for_learning, "teacher_action_margin_mean")
    teacher_action_margin_satisfied_values = _numeric_values(
        records_for_learning,
        "teacher_action_margin_satisfied_fraction",
    )
    teacher_same_family_action_margin_mean_values = _numeric_values(
        records_for_learning,
        "teacher_same_family_action_margin_mean",
    )
    teacher_same_family_action_margin_satisfied_values = _numeric_values(
        records_for_learning,
        "teacher_same_family_action_margin_satisfied_fraction",
    )
    teacher_public_heuristic_loss_values = _numeric_values(records_for_learning, "teacher_public_heuristic_loss")
    teacher_public_heuristic_supported_values = _numeric_values(
        records_for_learning,
        "teacher_public_heuristic_supported_fraction",
    )
    teacher_public_heuristic_top1_mass_values = _numeric_values(
        records_for_learning,
        "teacher_public_heuristic_top1_mass",
    )
    teacher_public_heuristic_target_entropy_values = _numeric_values(
        records_for_learning,
        "teacher_public_heuristic_target_entropy",
    )
    policy_anchor_coef_active_values = _numeric_values(records_for_learning, "policy_anchor_coef_active")
    policy_anchor_top_action_coef_active_values = _numeric_values(
        records_for_learning,
        "policy_anchor_top_action_coef_active",
    )
    policy_anchor_loss_values = _numeric_values(records_for_learning, "policy_anchor_loss")
    policy_anchor_weighted_loss_values = _numeric_values(records_for_learning, "policy_anchor_weighted_loss")
    policy_anchor_kl_mean_values = _numeric_values(records_for_learning, "policy_anchor_kl_mean")
    policy_anchor_kl_p95_values = _numeric_values(records_for_learning, "policy_anchor_kl_p95")
    policy_anchor_top_action_loss_values = _numeric_values(records_for_learning, "policy_anchor_top_action_loss")
    policy_anchor_top_action_loss_p95_values = _numeric_values(
        records_for_learning,
        "policy_anchor_top_action_loss_p95",
    )
    policy_anchor_top_action_agreement_values = _numeric_values(
        records_for_learning,
        "policy_anchor_top_action_agreement",
    )
    main_move_fraction_values = _fraction_values(scalars, "collector_main_move_actions", "collector_total_actions")
    teacher_tactical_row_fraction_values = _fraction_values(
        scalars,
        "collector_teacher_tactical_row_count",
        "collector_total_actions",
    )
    pass_fraction_values = _fraction_values(scalars, "collector_pass_actions", "collector_total_actions")
    pass_with_nonpass_total_fraction_values = _fraction_values(
        scalars,
        "collector_pass_with_nonpass_available",
        "collector_total_actions",
    )
    pass_with_nonpass_pass_fraction_values = _fraction_values(
        scalars,
        "collector_pass_with_nonpass_available",
        "collector_pass_actions",
    )
    pass_penalty_total_fraction_values = _fraction_values(
        scalars,
        "collector_pass_with_nonpass_penalty_count",
        "collector_total_actions",
    )
    pass_penalty_pass_fraction_values = _fraction_values(
        scalars,
        "collector_pass_with_nonpass_penalty_count",
        "collector_pass_actions",
    )
    mulligan_penalty_total_fraction_values = _fraction_values(
        scalars,
        "collector_mulligan_select_with_confirm_penalty_count",
        "collector_total_actions",
    )
    mulligan_guard_rows_total_fraction_values = _fraction_values(
        scalars,
        "collector_mulligan_force_confirm_after_select_rows",
        "collector_total_actions",
    )
    mulligan_guard_actions_total_fraction_values = _fraction_values(
        scalars,
        "collector_mulligan_force_confirm_after_select_actions",
        "collector_total_actions",
    )
    main_move_guard_rows_total_fraction_values = _fraction_values(
        scalars,
        "collector_main_move_only_force_pass_rows",
        "collector_total_actions",
    )
    main_move_guard_actions_total_fraction_values = _fraction_values(
        scalars,
        "collector_main_move_only_force_pass_actions",
        "collector_total_actions",
    )
    max_consecutive_main_move_values = _numeric_values(scalars, "collector_max_consecutive_main_moves")

    warnings: list[str] = []
    if actor_heuristic_values and max(actor_heuristic_values) > 0.0:
        warnings.append("actor_heuristic_fraction_active was nonzero; focal actions were not pure model-policy rows")
    if heuristic_mix_values and max(heuristic_mix_values) > 0.0:
        warnings.append("heuristic_public_mix_fraction_active was nonzero; eval/train pressure includes B2 heuristic")
    if isinstance(best_checkpoint, dict) and best_checkpoint.get("metric_kind") == "training_loss":
        warnings.append("checkpoint best was selected by scalar training_loss, not dev-eval quality")
    if checkpoint_alias_integrity["latest_alias_matches_source"] is False:
        warnings.append("latest checkpoint alias file does not match its tracker source checkpoint")
    if checkpoint_alias_integrity["observed_best_alias_matches_source"] is False:
        warnings.append("observed_best checkpoint alias file does not match its tracker source checkpoint")
    if actor_lag_warning_values and max(actor_lag_warning_values) > _LEARNER_ACTOR_LAG_WARN_THRESHOLD:
        warnings.append(
            f"{actor_lag_warning_source} exceeded {_LEARNER_ACTOR_LAG_WARN_THRESHOLD:g}; actor policy may be stale"
        )
    if not periodic_dev_eval.exists():
        warnings.append("periodic dev-eval summaries are absent; learning quality was not monitored during training")
    elif dev_eval_trend["latest_minus_best"] is not None and float(dev_eval_trend["latest_minus_best"]) < -0.05:
        warnings.append("latest periodic dev-eval aggregate is more than 0.05 below an earlier checkpoint")
    latest_champion_pool_size = None if not pfsp_champion_pool_size_values else pfsp_champion_pool_size_values[-1]
    latest_snapshot_env_fraction = (
        None if not pfsp_snapshot_env_fraction_values else pfsp_snapshot_env_fraction_values[-1]
    )
    latest_recent_env_fraction = None if not pfsp_recent_env_fraction_values else pfsp_recent_env_fraction_values[-1]
    if promotion_gate["attempt_count"] > 0 and promotion_gate["passed_count"] == 0:
        if latest_champion_pool_size is not None and latest_champion_pool_size > 0.0:
            warnings.append(
                "promotion gate never passed; champion pool is populated by imported/bootstrap champions, "
                "not promoted trained champions"
            )
        elif (
            latest_champion_pool_size == 0.0
            and latest_snapshot_env_fraction is not None
            and latest_snapshot_env_fraction > 0.0
        ):
            warnings.append(
                "promotion gate never passed; no trained champions were admitted, but probationary snapshot "
                "sampling was active"
            )
        else:
            warnings.append("promotion gate never passed; league did not admit any trained champions")
    if int(promotion_gate["consecutive_failure_count"]) >= 3:
        warnings.append(
            "promotion gate failed "
            f"{int(promotion_gate['consecutive_failure_count'])} consecutive attempts through latest update"
        )
    if vtrace_rho_values and max(vtrace_rho_values) > _OFF_POLICY_RHO_WARN_THRESHOLD:
        warnings.append(
            "vtrace_rho_mean exceeded "
            f"{_OFF_POLICY_RHO_WARN_THRESHOLD:g}; behavior/evaluation log-probs may be mismatched"
        )
    train_rho_tail_values = vtrace_train_rho_p99_values or vtrace_train_rho_p95_values
    max_train_rho_tail = None if not train_rho_tail_values else max(train_rho_tail_values)
    if vtrace_rho_p99_values and max(vtrace_rho_p99_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
        if max_train_rho_tail is not None and max_train_rho_tail <= _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
            warnings.append(
                "raw vtrace_rho_p99 exceeded "
                f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}, but train-mask rho tail stayed below threshold; "
                "large off-policy tails are mostly filtered or non-train rows"
            )
        else:
            warnings.append(
                "vtrace_rho_p99 exceeded "
                f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}; off-policy correction tails are large"
            )
    if vtrace_train_rho_values and max(vtrace_train_rho_values) > _OFF_POLICY_RHO_WARN_THRESHOLD:
        warnings.append(
            "vtrace_train_rho_mean exceeded "
            f"{_OFF_POLICY_RHO_WARN_THRESHOLD:g}; train-mask behavior/evaluation log-probs may be mismatched"
        )
    if vtrace_train_rho_p95_values and max(vtrace_train_rho_p95_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
        warnings.append(
            "vtrace_train_rho_p95 exceeded "
            f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}; train-mask off-policy correction tails are large"
        )
    if vtrace_train_rho_p99_values and max(vtrace_train_rho_p99_values) > _OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:
        warnings.append(
            "vtrace_train_rho_p99 exceeded "
            f"{_OFF_POLICY_RHO_TAIL_WARN_THRESHOLD:g}; train-mask off-policy correction tails are large"
        )
    if vtrace_clip_rate_values and max(vtrace_clip_rate_values) > _VTRACE_CLIP_RATE_WARN_THRESHOLD:
        warnings.append(
            f"vtrace_clip_rate exceeded {_VTRACE_CLIP_RATE_WARN_THRESHOLD:g}; policy updates are heavily clipped"
        )
    if (
        train_logp_delta_abs_p99_values
        and max(train_logp_delta_abs_p99_values) > _TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD
    ):
        warnings.append(
            "target_behavior_train_logp_delta_abs_p99 exceeded "
            f"{_TARGET_BEHAVIOR_LOGP_DELTA_WARN_THRESHOLD:g}; learner and behavior log-probs diverged on train rows"
        )
    if (
        max_consecutive_main_move_values
        and max(max_consecutive_main_move_values) > _MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD
    ):
        warnings.append(
            "collector_max_consecutive_main_moves exceeded "
            f"{_MAX_CONSECUTIVE_MAIN_MOVES_WARN_THRESHOLD:g}; repeated main-move transitions or counter drift suspected"
        )
    if (
        chosen_pass_train_fraction_values
        and _window_summary(chosen_pass_train_fraction_values, window=20)["last_window_mean"] > 0.5
    ):
        warnings.append("chosen_pass_train_fraction averaged above 0.5 in the latest window; pass-collapse suspected")
    if (
        chosen_mulligan_select_share_values
        and _window_summary(chosen_mulligan_select_share_values, window=20)["last_window_mean"]
        > _MULLIGAN_SELECT_SHARE_WARN_THRESHOLD
    ):
        warnings.append(
            "chosen_mulligan_select share among mulligan actions is high in the latest window; "
            "mulligan-confirm collapse suspected"
        )
    if (
        pass_with_nonpass_total_fraction_values
        and _window_summary(pass_with_nonpass_total_fraction_values, window=20)["last_window_mean"] > 0.35
    ):
        warnings.append(
            "collector pass-with-nonpass fraction is high in the latest window; policy may be avoiding play"
        )
    if (
        teacher_public_heuristic_coef_active_values
        and (_window_summary(teacher_public_heuristic_coef_active_values, window=20)["last_window_mean"] or 0.0) > 0.0
        and (
            not teacher_public_heuristic_supported_values
            or (
                _window_summary(teacher_public_heuristic_supported_values, window=20)["last_window_mean"]
                < _TEACHER_SUPPORTED_WARN_THRESHOLD
            )
        )
    ):
        warnings.append(
            "teacher_public_heuristic_coef_active was nonzero but public-teacher support was near zero; "
            "teacher labels or packed metadata may be missing"
        )
    if (
        teacher_hand_coef_active_values
        and (_window_summary(teacher_hand_coef_active_values, window=20)["last_window_mean"] or 0.0) > 0.0
        and (
            not teacher_hand_supported_values
            or (
                _window_summary(teacher_hand_supported_values, window=20)["last_window_mean"]
                < _TEACHER_SUPPORTED_WARN_THRESHOLD
            )
        )
    ):
        warnings.append(
            "teacher_hand_coef_active was nonzero but hand-target support was near zero; "
            "hand metadata or factorized same-family arg0 references may be missing"
        )
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
    summary = {
        "run_dir": run_dir.resolve().as_posix(),
        "training_record_count": len(metrics),
        "update_min": None if not update_counts else int(min(update_counts)),
        "update_max": None if not update_counts else int(max(update_counts)),
        "loss": _window_summary(_numeric_values(metrics, "loss"), window=20),
        "teacher_family_accuracy": _window_summary(
            _numeric_values(records_for_learning, "teacher_family_accuracy"), window=20
        ),
        "teacher_slot_accuracy": _window_summary(
            _numeric_values(records_for_learning, "teacher_slot_accuracy"), window=20
        ),
        "teacher_action_accuracy": _window_summary(
            _numeric_values(records_for_learning, "teacher_action_accuracy"), window=20
        ),
        "teacher_guidance": {
            "teacher_public_heuristic_coef_active": _window_summary(
                teacher_public_heuristic_coef_active_values,
                window=20,
            ),
            "teacher_hand_coef_active": _window_summary(
                teacher_hand_coef_active_values,
                window=20,
            ),
            "teacher_aux_loss": _window_summary(teacher_aux_loss_values, window=20),
            "teacher_main_play_character_slot_accuracy": _window_summary(
                teacher_main_play_slot_accuracy_values,
                window=20,
            ),
            "teacher_hand_accuracy": _window_summary(teacher_hand_accuracy_values, window=20),
            "teacher_main_play_character_hand_accuracy": _window_summary(
                teacher_main_play_hand_accuracy_values,
                window=20,
            ),
            "teacher_clock_from_hand_accuracy": _window_summary(
                teacher_clock_hand_accuracy_values,
                window=20,
            ),
            "teacher_hand_loss": _window_summary(teacher_hand_loss_values, window=20),
            "teacher_hand_supported_fraction": _window_summary(teacher_hand_supported_values, window=20),
            "teacher_same_family_action_accuracy": _window_summary(
                teacher_same_family_action_accuracy_values,
                window=20,
            ),
            "teacher_same_family_main_play_character_accuracy": _window_summary(
                teacher_same_family_main_play_accuracy_values,
                window=20,
            ),
            "teacher_action_margin_mean": _window_summary(
                teacher_action_margin_mean_values,
                window=20,
            ),
            "teacher_action_margin_satisfied_fraction": _window_summary(
                teacher_action_margin_satisfied_values,
                window=20,
            ),
            "teacher_same_family_action_margin_mean": _window_summary(
                teacher_same_family_action_margin_mean_values,
                window=20,
            ),
            "teacher_same_family_action_margin_satisfied_fraction": _window_summary(
                teacher_same_family_action_margin_satisfied_values,
                window=20,
            ),
            "teacher_public_heuristic_loss": _window_summary(
                teacher_public_heuristic_loss_values,
                window=20,
            ),
            "teacher_public_heuristic_supported_fraction": _window_summary(
                teacher_public_heuristic_supported_values,
                window=20,
            ),
            "teacher_public_heuristic_top1_mass": _window_summary(
                teacher_public_heuristic_top1_mass_values,
                window=20,
            ),
            "teacher_public_heuristic_target_entropy": _window_summary(
                teacher_public_heuristic_target_entropy_values,
                window=20,
            ),
            "teacher_tactical_row_fraction_of_total": _window_summary(
                teacher_tactical_row_fraction_values,
                window=20,
            ),
            "policy_anchor_coef_active": _window_summary(policy_anchor_coef_active_values, window=20),
            "policy_anchor_top_action_coef_active": _window_summary(
                policy_anchor_top_action_coef_active_values,
                window=20,
            ),
            "policy_anchor_loss": _window_summary(policy_anchor_loss_values, window=20),
            "policy_anchor_weighted_loss": _window_summary(policy_anchor_weighted_loss_values, window=20),
            "policy_anchor_kl_mean": _window_summary(policy_anchor_kl_mean_values, window=20),
            "policy_anchor_kl_p95": _window_summary(policy_anchor_kl_p95_values, window=20),
            "policy_anchor_top_action_loss": _window_summary(policy_anchor_top_action_loss_values, window=20),
            "policy_anchor_top_action_loss_p95": _window_summary(
                policy_anchor_top_action_loss_p95_values,
                window=20,
            ),
            "policy_anchor_top_action_agreement": _window_summary(
                policy_anchor_top_action_agreement_values,
                window=20,
            ),
            "max_teacher_public_heuristic_coef_active": None
            if not teacher_public_heuristic_coef_active_values
            else max(teacher_public_heuristic_coef_active_values),
            "max_teacher_hand_coef_active": None
            if not teacher_hand_coef_active_values
            else max(teacher_hand_coef_active_values),
            "max_teacher_public_heuristic_supported_fraction": None
            if not teacher_public_heuristic_supported_values
            else max(teacher_public_heuristic_supported_values),
            "max_teacher_hand_supported_fraction": None
            if not teacher_hand_supported_values
            else max(teacher_hand_supported_values),
        },
        "route": {
            "max_actor_heuristic_fraction_active": None if not actor_heuristic_values else max(actor_heuristic_values),
            "max_heuristic_public_mix_fraction_active": None if not heuristic_mix_values else max(heuristic_mix_values),
        },
        "league_sampling": {
            "pfsp_pool_size": _window_summary(pfsp_pool_size_values, window=20),
            "pfsp_champion_pool_size": _window_summary(pfsp_champion_pool_size_values, window=20),
            "pfsp_recent_pool_size": _window_summary(pfsp_recent_pool_size_values, window=20),
            "pfsp_hard_negative_pool_size": _window_summary(pfsp_hard_negative_pool_size_values, window=20),
            "pfsp_quarantined_opponents": _window_summary(pfsp_quarantined_opponent_values, window=20),
            "snapshot_env_fraction": _window_summary(pfsp_snapshot_env_fraction_values, window=20),
            "champion_env_fraction": _window_summary(pfsp_champion_env_fraction_values, window=20),
            "recent_env_fraction": _window_summary(pfsp_recent_env_fraction_values, window=20),
            "hard_negative_env_fraction": _window_summary(pfsp_hard_negative_env_fraction_values, window=20),
            "warmup_snapshot_env_fraction": _window_summary(pfsp_warmup_snapshot_env_fraction_values, window=20),
            "max_snapshot_env_fraction": None
            if not pfsp_snapshot_env_fraction_values
            else max(pfsp_snapshot_env_fraction_values),
            "latest_has_admitted_champion": bool(latest_champion_pool_size and latest_champion_pool_size > 0.0),
            "latest_probationary_recent_sampling_active": bool(
                latest_champion_pool_size == 0.0
                and latest_recent_env_fraction is not None
                and latest_recent_env_fraction > 0.0
            ),
        },
        "actor_model_sync": {
            "policy_version_lag_p50": _window_summary(policy_version_lag_p50_values, window=20),
            "policy_version_lag_p90": _window_summary(policy_version_lag_p90_values, window=20),
            "max_policy_version_lag_p90": None
            if not policy_version_lag_p90_values
            else max(policy_version_lag_p90_values),
            "learner_actor_update_lag_p50": _window_summary(learner_actor_update_lag_p50_values, window=20),
            "learner_actor_update_lag_p90": _window_summary(learner_actor_update_lag_p90_values, window=20),
            "max_learner_actor_update_lag_p90": None
            if not learner_actor_update_lag_p90_values
            else max(learner_actor_update_lag_p90_values),
            "lag_warning_source": actor_lag_warning_source,
            "learner_to_actor_update_lag": _window_summary(actor_lag_warning_values, window=20),
            "max_learner_to_actor_update_lag": None if not actor_lag_warning_values else max(actor_lag_warning_values),
        },
        "league_sync": {
            "league_update_lag": _window_summary(league_update_lag_values, window=20),
            "max_league_update_lag": None if not league_update_lag_values else max(league_update_lag_values),
        },
        "off_policy": {
            "vtrace_rho_mean": _window_summary(vtrace_rho_values, window=20),
            "vtrace_rho_p99": _window_summary(vtrace_rho_p99_values, window=20),
            "vtrace_train_rho_mean": _window_summary(vtrace_train_rho_values, window=20),
            "vtrace_train_rho_p95": _window_summary(vtrace_train_rho_p95_values, window=20),
            "vtrace_train_rho_p99": _window_summary(vtrace_train_rho_p99_values, window=20),
            "vtrace_clip_rate": _window_summary(vtrace_clip_rate_values, window=20),
            "target_behavior_logp_delta_abs_mean": _window_summary(logp_delta_abs_values, window=20),
            "target_behavior_logp_delta_abs_p99": _window_summary(logp_delta_abs_p99_values, window=20),
            "target_behavior_train_logp_delta_abs_mean": _window_summary(train_logp_delta_abs_values, window=20),
            "target_behavior_train_logp_delta_abs_p99": _window_summary(train_logp_delta_abs_p99_values, window=20),
            "max_vtrace_rho_mean": None if not vtrace_rho_values else max(vtrace_rho_values),
            "max_vtrace_rho_p99": None if not vtrace_rho_p99_values else max(vtrace_rho_p99_values),
            "max_vtrace_train_rho_mean": None if not vtrace_train_rho_values else max(vtrace_train_rho_values),
            "max_vtrace_train_rho_p95": None if not vtrace_train_rho_p95_values else max(vtrace_train_rho_p95_values),
            "max_vtrace_train_rho_p99": None if not vtrace_train_rho_p99_values else max(vtrace_train_rho_p99_values),
            "max_vtrace_clip_rate": None if not vtrace_clip_rate_values else max(vtrace_clip_rate_values),
            "max_target_behavior_logp_delta_abs_mean": None
            if not logp_delta_abs_values
            else max(logp_delta_abs_values),
            "max_target_behavior_logp_delta_abs_p99": None
            if not logp_delta_abs_p99_values
            else max(logp_delta_abs_p99_values),
            "max_target_behavior_train_logp_delta_abs_mean": None
            if not train_logp_delta_abs_values
            else max(train_logp_delta_abs_values),
            "max_target_behavior_train_logp_delta_abs_p99": None
            if not train_logp_delta_abs_p99_values
            else max(train_logp_delta_abs_p99_values),
            "stale_policy_lag_source": actor_lag_warning_source,
            "stale_policy_lag_correlations": {
                key: {
                    "paired_update_count": len(pairs),
                    "pearson": _pearson_correlation(pairs),
                }
                for key, pairs in stale_policy_pairs.items()
            },
        },
        "reward_scale": {
            "reward_mean": _window_summary(reward_mean_values, window=20),
            "reward_abs_mean": _window_summary(reward_abs_values, window=20),
            "reward_std": _window_summary(reward_std_values, window=20),
            "reward_nonzero_fraction": _window_summary(reward_nonzero_values, window=20),
            "reward_positive_fraction": _window_summary(reward_positive_values, window=20),
            "reward_negative_fraction": _window_summary(reward_negative_values, window=20),
            "advantage_abs_mean": _window_summary(advantage_abs_values, window=20),
            "target_abs_mean": _window_summary(target_abs_values, window=20),
            "max_reward_abs_mean": None if not reward_abs_values else max(reward_abs_values),
            "max_target_abs_mean": None if not target_abs_values else max(target_abs_values),
        },
        "chosen_action_learning": {
            "chosen_pass_train_fraction": _window_summary(chosen_pass_train_fraction_values, window=20),
            "chosen_pass_train_advantage_mean": _window_summary(chosen_pass_train_advantage_values, window=20),
            "chosen_nonpass_train_advantage_mean": _window_summary(chosen_nonpass_train_advantage_values, window=20),
            "chosen_mulligan_confirm_train_fraction": _window_summary(
                chosen_mulligan_confirm_train_fraction_values,
                window=20,
            ),
            "chosen_mulligan_select_train_fraction": _window_summary(
                chosen_mulligan_select_train_fraction_values,
                window=20,
            ),
            "chosen_mulligan_select_share_of_mulligan": _window_summary(
                chosen_mulligan_select_share_values,
                window=20,
            ),
            "chosen_mulligan_confirm_train_advantage_mean": _window_summary(
                chosen_mulligan_confirm_train_advantage_values,
                window=20,
            ),
            "chosen_mulligan_select_train_advantage_mean": _window_summary(
                chosen_mulligan_select_train_advantage_values,
                window=20,
            ),
            "chosen_main_play_character_train_fraction": _window_summary(chosen_play_train_fraction_values, window=20),
            "chosen_attack_train_fraction": _window_summary(chosen_attack_train_fraction_values, window=20),
        },
        "action_distribution": {
            "main_move_fraction": _window_summary(main_move_fraction_values, window=20),
            "pass_fraction": _window_summary(pass_fraction_values, window=20),
            "pass_with_nonpass_fraction_of_total": _window_summary(
                pass_with_nonpass_total_fraction_values,
                window=20,
            ),
            "pass_with_nonpass_fraction_of_pass": _window_summary(
                pass_with_nonpass_pass_fraction_values,
                window=20,
            ),
            "pass_penalty_fraction_of_total": _window_summary(pass_penalty_total_fraction_values, window=20),
            "pass_penalty_fraction_of_pass": _window_summary(pass_penalty_pass_fraction_values, window=20),
            "mulligan_select_with_confirm_penalty_fraction_of_total": _window_summary(
                mulligan_penalty_total_fraction_values,
                window=20,
            ),
            "mulligan_force_confirm_after_select_rows_fraction_of_total": _window_summary(
                mulligan_guard_rows_total_fraction_values,
                window=20,
            ),
            "mulligan_force_confirm_after_select_actions_fraction_of_total": _window_summary(
                mulligan_guard_actions_total_fraction_values,
                window=20,
            ),
            "main_move_only_force_pass_rows_fraction_of_total": _window_summary(
                main_move_guard_rows_total_fraction_values,
                window=20,
            ),
            "main_move_only_force_pass_actions_fraction_of_total": _window_summary(
                main_move_guard_actions_total_fraction_values,
                window=20,
            ),
            "max_consecutive_main_moves": _window_summary(max_consecutive_main_move_values, window=20),
            "max_max_consecutive_main_moves": None
            if not max_consecutive_main_move_values
            else max(max_consecutive_main_move_values),
        },
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
    return summary


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
