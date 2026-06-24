"""Artifact-backed sections for learning-progress diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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
        raw_reasons = decision.get("reasons") if isinstance(decision, Mapping) else []
        reasons = raw_reasons if isinstance(raw_reasons, list) else []
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


__all__ = [
    "_checkpoint_alias_integrity",
    "_file_sha256_or_none",
    "_final_eval_matrix_summaries",
    "_final_eval_matrix_summary",
    "_json_or_none",
    "_periodic_dev_eval_trend",
    "_policy_id_from_checkpoint_record",
    "_policy_update_map",
    "_promotion_gate_summary",
    "_read_mean_matrix",
    "_read_numeric_matrix_payload",
    "_row_mean_excluding_self",
    "_run_relative_path",
    "_update_from_promotion_gate_path",
]
