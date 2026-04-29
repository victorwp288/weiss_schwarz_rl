"""Checkpoint tracker and alias-record helpers."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from weiss_rl.config import StackConfig, compute_config_hash256
from weiss_rl.training.dev_eval_metrics import (
    checkpoint_candidate_metric,
    dev_eval_aggregate_score,
    dev_eval_ineligibility_reasons,
    extract_anchor_score,
    should_promote_best_checkpoint,
    should_update_secondary_b2_record,
)

CHECKPOINT_TRACKER_FORMAT = "checkpoint_tracker_v2"
CHECKPOINT_TRACKER_FILENAME = "checkpoint_tracker.json"


class CheckpointTrackerPaths(Protocol):
    checkpoint_tracker_path: Path
    latest_checkpoint_path: Path
    best_checkpoint_path: Path


class RunArtifactPaths(Protocol):
    run_dir: Path


class LearnerCheckpointState(Protocol):
    update_count: int

    def get_policy_version(self) -> int: ...


def relative_path_text(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_checkpoint_tracker(training_paths: CheckpointTrackerPaths) -> dict[str, Any]:
    if not training_paths.checkpoint_tracker_path.is_file():
        return {"format": CHECKPOINT_TRACKER_FORMAT, "latest": None, "best": None, "secondary": {}}
    payload = json.loads(training_paths.checkpoint_tracker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint tracker must be a JSON object: {training_paths.checkpoint_tracker_path}")
    payload.setdefault("format", CHECKPOINT_TRACKER_FORMAT)
    payload.setdefault("latest", None)
    payload.setdefault("best", None)
    secondary = payload.get("secondary")
    if not isinstance(secondary, dict):
        payload["secondary"] = {}
    return payload


def write_checkpoint_tracker(training_paths: CheckpointTrackerPaths, payload: Mapping[str, Any]) -> None:
    normalized = dict(payload)
    normalized["format"] = CHECKPOINT_TRACKER_FORMAT
    secondary = normalized.get("secondary")
    if not isinstance(secondary, dict):
        normalized["secondary"] = {}
    training_paths.checkpoint_tracker_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_checkpoint_record(
    *,
    alias_name: str,
    alias_path: Path,
    source_checkpoint_path: Path,
    artifacts: RunArtifactPaths,
    learner: LearnerCheckpointState,
    metric_kind: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias_name,
        "alias_path": relative_path_text(alias_path, root=artifacts.run_dir),
        "source_checkpoint_path": relative_path_text(source_checkpoint_path, root=artifacts.run_dir),
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "metric_kind": metric_kind,
        "metric_value": metric_value,
    }


def build_secondary_checkpoint_record(
    *,
    source_checkpoint_path: Path,
    artifacts: RunArtifactPaths,
    update_count: int,
    policy_version: int,
    metric_kind: str,
    metric_value: float,
    aggregate_score: float | None,
    dev_eval_ineligibility_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_checkpoint_path": relative_path_text(source_checkpoint_path, root=artifacts.run_dir),
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "metric_kind": str(metric_kind),
        "metric_value": float(metric_value),
    }
    if aggregate_score is not None and np.isfinite(float(aggregate_score)):
        record["aggregate_score"] = float(aggregate_score)
    if dev_eval_ineligibility_reasons:
        record["dev_eval_ineligibility_reasons"] = [str(reason) for reason in dev_eval_ineligibility_reasons]
    return record


def build_checkpoint_record_for_update(
    *,
    alias_name: str,
    alias_path: Path,
    source_checkpoint_path: Path,
    artifacts: RunArtifactPaths,
    update_count: int,
    policy_version: int,
    metric_kind: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias_name,
        "alias_path": relative_path_text(alias_path, root=artifacts.run_dir),
        "source_checkpoint_path": relative_path_text(source_checkpoint_path, root=artifacts.run_dir),
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "metric_kind": metric_kind,
        "metric_value": metric_value,
    }


def checkpoint_secondary_records(tracker: dict[str, Any]) -> dict[str, Any]:
    secondary = tracker.get("secondary")
    if isinstance(secondary, dict):
        return secondary
    tracker["secondary"] = {}
    return cast(dict[str, Any], tracker["secondary"])


def update_secondary_b2_checkpoint_record(
    *,
    tracker: dict[str, Any],
    stack: StackConfig,
    artifacts: RunArtifactPaths,
    source_checkpoint_path: Path,
    update_count: int,
    policy_version: int,
    dev_eval_summary: Mapping[str, Any] | None,
    b2_policy_id: str,
) -> None:
    b2_score = extract_anchor_score(dev_eval_summary, b2_policy_id)
    if b2_score is None:
        return
    aggregate_score = dev_eval_aggregate_score(dev_eval_summary)
    secondary_records = checkpoint_secondary_records(tracker)
    existing_record = secondary_records.get("best_b2")
    existing_mapping = existing_record if isinstance(existing_record, Mapping) else None
    if not should_update_secondary_b2_record(
        existing_record=cast(Mapping[str, Any] | None, existing_mapping),
        candidate_b2_score=float(b2_score),
        candidate_aggregate_score=aggregate_score,
        update_count=int(update_count),
        policy_version=int(policy_version),
    ):
        return
    secondary_records["best_b2"] = build_secondary_checkpoint_record(
        source_checkpoint_path=source_checkpoint_path,
        artifacts=artifacts,
        update_count=int(update_count),
        policy_version=int(policy_version),
        metric_kind="b2_score",
        metric_value=float(b2_score),
        aggregate_score=aggregate_score,
        dev_eval_ineligibility_reasons=dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary),
    )


def publish_checkpoint_aliases(
    *,
    stack: StackConfig,
    training_paths: CheckpointTrackerPaths,
    artifacts: RunArtifactPaths,
    checkpoint_path: Path,
    learner: LearnerCheckpointState,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None = None,
    b2_policy_id: str,
) -> dict[str, Any]:
    tracker = load_checkpoint_tracker(training_paths)

    shutil.copy2(checkpoint_path, training_paths.latest_checkpoint_path)
    latest_kind, latest_value = checkpoint_candidate_metric(
        stack=stack,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
    )
    latest_record = build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind=latest_kind,
        metric_value=latest_value,
    )
    tracker["latest"] = latest_record

    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        best_record = None
    should_update_best = latest_kind is not None and (
        best_record is None
        or should_promote_best_checkpoint(
            existing_record=cast(Mapping[str, Any], best_record),
            candidate_kind=latest_kind,
            candidate_value=latest_value,
        )
    )
    if should_update_best:
        shutil.copy2(checkpoint_path, training_paths.best_checkpoint_path)
        tracker["best"] = build_checkpoint_record(
            alias_name="best",
            alias_path=training_paths.best_checkpoint_path,
            source_checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            learner=learner,
            metric_kind=latest_kind,
            metric_value=latest_value,
        )

    update_secondary_b2_checkpoint_record(
        tracker=tracker,
        stack=stack,
        artifacts=artifacts,
        source_checkpoint_path=checkpoint_path,
        update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        dev_eval_summary=dev_eval_summary,
        b2_policy_id=b2_policy_id,
    )
    write_checkpoint_tracker(training_paths, tracker)
    return tracker


def publish_best_checkpoint_from_dev_eval(
    *,
    stack: StackConfig,
    training_paths: CheckpointTrackerPaths,
    artifacts: RunArtifactPaths,
    checkpoint_path: Path,
    update_count: int,
    policy_version: int,
    dev_eval_summary: Mapping[str, Any] | None,
    b2_policy_id: str,
) -> dict[str, Any]:
    tracker = load_checkpoint_tracker(training_paths)
    candidate_kind, candidate_value = checkpoint_candidate_metric(
        stack=stack,
        latest_metrics=None,
        dev_eval_summary=dev_eval_summary,
    )
    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        best_record = None
    should_update_best = candidate_kind is not None and (
        best_record is None
        or should_promote_best_checkpoint(
            existing_record=cast(Mapping[str, Any], best_record),
            candidate_kind=candidate_kind,
            candidate_value=candidate_value,
        )
    )
    if should_update_best:
        shutil.copy2(checkpoint_path, training_paths.best_checkpoint_path)
        tracker["best"] = build_checkpoint_record_for_update(
            alias_name="best",
            alias_path=training_paths.best_checkpoint_path,
            source_checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            update_count=update_count,
            policy_version=policy_version,
            metric_kind=candidate_kind,
            metric_value=candidate_value,
        )
    update_secondary_b2_checkpoint_record(
        tracker=tracker,
        stack=stack,
        artifacts=artifacts,
        source_checkpoint_path=checkpoint_path,
        update_count=int(update_count),
        policy_version=int(policy_version),
        dev_eval_summary=dev_eval_summary,
        b2_policy_id=b2_policy_id,
    )
    write_checkpoint_tracker(training_paths, tracker)
    return tracker


def resolve_resume_source_checkpoint_path(
    *,
    source_run_dir: Path,
    record: Mapping[str, Any],
    key: str,
) -> Path | None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = source_run_dir / path
    return path.resolve()


def seed_checkpoint_tracker_from_resume_best(
    *,
    stack: StackConfig,
    training_paths: CheckpointTrackerPaths,
    artifacts: RunArtifactPaths,
    resume_checkpoint_path: Path,
) -> dict[str, Any] | None:
    target_tracker = load_checkpoint_tracker(training_paths)
    if isinstance(target_tracker.get("best"), Mapping):
        return None

    checkpoint_path = Path(resume_checkpoint_path).resolve()
    checkpoint_dir = checkpoint_path.parent
    training_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints" or training_dir.name != "training":
        return None
    source_run_dir = training_dir.parent
    source_tracker_path = source_run_dir / "training" / "checkpoints" / CHECKPOINT_TRACKER_FILENAME
    if not source_tracker_path.is_file():
        return None
    source_tracker = json.loads(source_tracker_path.read_text(encoding="utf-8"))
    if not isinstance(source_tracker, Mapping):
        return None
    source_best = source_tracker.get("best")
    if not isinstance(source_best, Mapping):
        return None
    metric_kind = source_best.get("metric_kind")
    metric_value = source_best.get("metric_value")
    if str(metric_kind).strip() != "dev_eval_mean":
        return None
    if not isinstance(metric_value, (int, float)) or not np.isfinite(float(metric_value)):
        return None

    candidate_paths = tuple(
        path
        for path in (
            resolve_resume_source_checkpoint_path(
                source_run_dir=source_run_dir,
                record=source_best,
                key="source_checkpoint_path",
            ),
            resolve_resume_source_checkpoint_path(
                source_run_dir=source_run_dir,
                record=source_best,
                key="alias_path",
            ),
        )
        if path is not None
    )
    if checkpoint_path not in candidate_paths:
        return None

    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(checkpoint_payload, Mapping):
        return None
    checkpoint_config_hash = str(checkpoint_payload.get("config_hash256", "")).strip().lower()
    if checkpoint_config_hash != compute_config_hash256(stack):
        return None

    update_count = source_best.get("update_count")
    policy_version = source_best.get("policy_version")
    if not isinstance(update_count, int) or not isinstance(policy_version, int):
        return None

    shutil.copy2(checkpoint_path, training_paths.best_checkpoint_path)
    seeded_record = {
        "alias": "best",
        "alias_path": relative_path_text(training_paths.best_checkpoint_path, root=artifacts.run_dir),
        "source_checkpoint_path": relative_path_text(checkpoint_path, root=artifacts.run_dir),
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "metric_kind": "dev_eval_mean",
        "metric_value": float(metric_value),
        "seeded_from_run_dir": source_run_dir.as_posix(),
    }
    target_tracker["best"] = seeded_record
    write_checkpoint_tracker(training_paths, target_tracker)
    return seeded_record
