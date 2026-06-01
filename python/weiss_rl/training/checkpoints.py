from __future__ import annotations

import json
import math
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from weiss_rl.models.state_dict_compat import load_model_state_dict_with_context_compat
from weiss_rl.training.checkpoint_guard import (
    checkpoint_candidate_metric,
    dev_eval_aggregate_score,
    dev_eval_confidence_stats,
    dev_eval_ineligibility_reasons,
    dev_eval_worst_natural_timeout_rate,
    dev_eval_worst_no_progress_timeout_rate,
    dev_eval_worst_stall_rate,
    dev_eval_worst_truncation_rate,
    should_promote_best_checkpoint,
)
from weiss_rl.training.snapshots import demote_registry_champions_newer_than

LATEST_CHECKPOINT_FILENAME = "latest.pt"
BEST_CHECKPOINT_FILENAME = "best.pt"
OBSERVED_BEST_CHECKPOINT_FILENAME = "observed_best.pt"
CHECKPOINT_TRACKER_FILENAME = "checkpoint_tracker.json"
CHECKPOINT_TRACKER_FORMAT = "checkpoint_tracker_v1"
MINIMAL_TRAIN_CHECKPOINT_FORMAT = "minimal_train_checkpoint_v1"


class CheckpointTrainingPaths(Protocol):
    checkpoint_tracker_path: Path
    logs_dir: Path


class CheckpointWritePaths(Protocol):
    checkpoints_dir: Path


class CheckpointAliasPaths(CheckpointTrainingPaths, Protocol):
    latest_checkpoint_path: Path
    best_checkpoint_path: Path
    snapshots_dir: Path


class LearnerRecordSource(Protocol):
    update_count: int

    def get_policy_version(self) -> int: ...


class CheckpointGuardRuntime(Protocol):
    def maybe_publish_snapshot(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def reset_outcome_tracker(self) -> None: ...

    def refresh_opponent_pool(self) -> None: ...


class CheckpointLearner(Protocol):
    update_count: int
    policy_version: int
    total_samples_processed: int
    start_time: float
    model: Any
    optimizer: Any
    _grad_scaler: Any

    def get_policy_version(self) -> int: ...

    def _optimizer_for_step(self) -> Any: ...

    def policy_anchor_state_dict(self) -> dict[str, Any] | None: ...

    def load_policy_anchor_state_dict(self, state_dict: Mapping[str, Any] | None) -> None: ...

    def reset_policy_anchor_to_current_model(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CheckpointPayloadContract:
    payload: dict[str, Any]
    model_state_dict: dict[str, Any]
    config_hash_mismatch: bool
    expected_config_hash: str
    payload_config_hash: str


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    checkpoint_path: Path
    update_count: int
    policy_version: int
    total_samples_processed: int
    init_schedule_offset_updates: int = 0


def relative_path_text(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def default_checkpoint_tracker_payload() -> dict[str, Any]:
    return {"format": CHECKPOINT_TRACKER_FORMAT, "latest": None, "best": None, "observed_best": None}


def load_checkpoint_tracker(training_paths: CheckpointTrainingPaths) -> dict[str, Any]:
    tracker_path = training_paths.checkpoint_tracker_path
    if not tracker_path.is_file():
        return default_checkpoint_tracker_payload()
    payload = json.loads(tracker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint tracker must be a JSON object: {tracker_path}")
    payload.setdefault("format", CHECKPOINT_TRACKER_FORMAT)
    payload.setdefault("latest", None)
    payload.setdefault("best", None)
    payload.setdefault("observed_best", None)
    return payload


def write_checkpoint_tracker(training_paths: CheckpointTrainingPaths, payload: dict[str, Any]) -> None:
    training_paths.checkpoint_tracker_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def checkpoint_guard_log_path(training_paths: CheckpointTrainingPaths) -> Path:
    return training_paths.logs_dir / "checkpoint_guard.jsonl"


def append_checkpoint_guard_event(training_paths: CheckpointTrainingPaths, payload: Mapping[str, Any]) -> None:
    path = checkpoint_guard_log_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def best_checkpoint_record(training_paths: CheckpointTrainingPaths) -> Mapping[str, Any] | None:
    best_record = load_checkpoint_tracker(training_paths).get("best")
    return best_record if isinstance(best_record, Mapping) else None


def observed_best_checkpoint_path(training_paths: CheckpointTrainingPaths) -> Path:
    return training_paths.checkpoint_tracker_path.parent / OBSERVED_BEST_CHECKPOINT_FILENAME


def extract_structured_guard_b2_anchor_score(dev_eval_summary: Mapping[str, Any] | None) -> float | None:
    if dev_eval_summary is None:
        return None
    anchor_scores = dev_eval_summary.get("anchor_scores")
    if not isinstance(anchor_scores, Mapping):
        return None
    for key, value in anchor_scores.items():
        key_text = str(key).strip().lower()
        if "b2" not in key_text:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def maybe_log_structured_mainmove_guard(
    *,
    training_paths: CheckpointTrainingPaths,
    learner: LearnerRecordSource,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if latest_metrics is None:
        return None
    top1_rate = latest_metrics.get("structured_main_move_0_2_top1_rate")
    move_share = latest_metrics.get("structured_main_move_share_when_play_available")
    if top1_rate is None or move_share is None:
        return None
    if not math.isfinite(float(top1_rate)) or not math.isfinite(float(move_share)):
        return None
    if float(top1_rate) < 0.15 and float(move_share) < 0.35:
        return None

    aggregate_score = dev_eval_aggregate_score(dev_eval_summary) if dev_eval_summary is not None else None
    b2_score = extract_structured_guard_b2_anchor_score(dev_eval_summary)
    if b2_score is not None and float(b2_score) > 0.10:
        return None
    if b2_score is None and aggregate_score is not None and float(aggregate_score) > 0.40:
        return None

    payload = {
        "format": "checkpoint_guard_event_v1",
        "event_kind": "structured_mainmove_warning_v1",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "structured_main_move_0_2_top1_rate": float(top1_rate),
        "structured_main_move_share_when_play_available": float(move_share),
        "dev_eval_aggregate_score": None if aggregate_score is None else float(aggregate_score),
        "b2_anchor_score": None if b2_score is None else float(b2_score),
    }
    append_checkpoint_guard_event(training_paths, payload)
    return payload


def write_scalars_record(
    *,
    scalars_path: Path,
    learner: LearnerRecordSource,
    metrics: dict[str, float],
    start_time: float,
) -> dict[str, Any]:
    wall_clock_seconds = time.time() - start_time
    record = {
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_ms": int(wall_clock_seconds * 1000),
        **metrics,
    }
    with scalars_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def build_checkpoint_record(
    *,
    alias_name: str,
    alias_path: Path,
    source_checkpoint_path: Path,
    run_dir: Path,
    learner: LearnerRecordSource,
    metric_kind: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "alias": alias_name,
        "alias_path": relative_path_text(alias_path, root=run_dir),
        "source_checkpoint_path": relative_path_text(source_checkpoint_path, root=run_dir),
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "metric_kind": metric_kind,
        "metric_value": metric_value,
    }


def dev_eval_candidate_diagnostics(
    *,
    stack: Any,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if dev_eval_summary is None:
        return None
    reasons = dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)
    confidence = dev_eval_confidence_stats(dev_eval_summary)
    return {
        "score": dev_eval_aggregate_score(dev_eval_summary),
        "eligible_for_best": not reasons,
        "ineligibility_reasons": list(reasons),
        "confidence": confidence,
        "worst_truncation_rate": dev_eval_worst_truncation_rate(dev_eval_summary),
        "worst_no_progress_timeout_rate": dev_eval_worst_no_progress_timeout_rate(dev_eval_summary),
        "worst_natural_timeout_rate": dev_eval_worst_natural_timeout_rate(dev_eval_summary),
        "worst_stall_rate": dev_eval_worst_stall_rate(dev_eval_summary),
    }


def current_focal_policy_id(*, learner: LearnerRecordSource) -> str:
    return f"train_u{int(learner.update_count)}_p{int(learner.get_policy_version())}"


def checkpoint_path_for_update(checkpoints_dir: Path, *, update_count: int) -> Path:
    return checkpoints_dir / f"checkpoint_{update_count}.pt"


def ensure_current_checkpoint(
    *,
    training_paths: CheckpointWritePaths,
    learner: LearnerRecordSource,
    write_checkpoint: Any,
) -> Path:
    checkpoint_path = checkpoint_path_for_update(
        training_paths.checkpoints_dir,
        update_count=int(learner.update_count),
    )
    if checkpoint_path.is_file():
        return checkpoint_path

    write_checkpoint(checkpoint_path)
    return checkpoint_path


def publish_checkpoint_aliases(
    *,
    stack: Any,
    training_paths: CheckpointAliasPaths,
    run_dir: Path,
    checkpoint_path: Path,
    learner: LearnerRecordSource,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None = None,
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
        run_dir=run_dir,
        learner=learner,
        metric_kind=latest_kind,
        metric_value=latest_value,
    )
    latest_dev_eval_candidate = dev_eval_candidate_diagnostics(stack=stack, dev_eval_summary=dev_eval_summary)
    if latest_dev_eval_candidate is not None:
        latest_record["dev_eval_candidate"] = latest_dev_eval_candidate
    tracker["latest"] = latest_record

    observed_score = dev_eval_aggregate_score(dev_eval_summary)
    observed_best_record = tracker.get("observed_best")
    if not isinstance(observed_best_record, Mapping):
        observed_best_record = None
    observed_best_value = None if observed_best_record is None else observed_best_record.get("metric_value")
    should_update_observed_best = observed_score is not None and (
        not isinstance(observed_best_value, (int, float))
        or not math.isfinite(float(observed_best_value))
        or float(observed_score) > float(observed_best_value)
    )
    if should_update_observed_best:
        assert observed_score is not None
        observed_best_path = observed_best_checkpoint_path(training_paths)
        shutil.copy2(checkpoint_path, observed_best_path)
        tracker["observed_best"] = build_checkpoint_record(
            alias_name="observed_best",
            alias_path=observed_best_path,
            source_checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            learner=learner,
            metric_kind="dev_eval_observed_mean",
            metric_value=float(observed_score),
        )
        if latest_dev_eval_candidate is not None:
            tracker["observed_best"]["dev_eval_candidate"] = latest_dev_eval_candidate

    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        best_record = None
    should_update_best = should_promote_best_checkpoint(
        existing_record=cast(Mapping[str, Any] | None, best_record),
        candidate_kind=latest_kind,
        candidate_value=latest_value,
    )
    if should_update_best:
        shutil.copy2(checkpoint_path, training_paths.best_checkpoint_path)
        tracker["best"] = build_checkpoint_record(
            alias_name="best",
            alias_path=training_paths.best_checkpoint_path,
            source_checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            learner=learner,
            metric_kind=latest_kind,
            metric_value=latest_value,
        )

    write_checkpoint_tracker(training_paths, tracker)
    return tracker


def maybe_rollback_to_best_checkpoint(
    *,
    stack: Any,
    training_paths: CheckpointAliasPaths,
    run_dir: Path,
    runtime: CheckpointGuardRuntime,
    learner: LearnerRecordSource,
    learner_model: Any,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
    last_rollback_update: int | None,
    restore_checkpoint: Any,
    write_checkpoint: Any,
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    checkpoint_guard = curriculum.checkpoint_guard
    if not checkpoint_guard.enabled or dev_eval_summary is None:
        return None
    if last_rollback_update is not None and (int(learner.update_count) - int(last_rollback_update)) < int(
        checkpoint_guard.cooldown_updates
    ):
        return None

    current_score = dev_eval_aggregate_score(dev_eval_summary)
    if current_score is None:
        return None
    worst_truncation_rate = dev_eval_worst_truncation_rate(dev_eval_summary)
    worst_stall_rate = dev_eval_worst_stall_rate(dev_eval_summary)
    worst_no_progress_timeout_rate = dev_eval_worst_no_progress_timeout_rate(dev_eval_summary)
    worst_natural_timeout_rate = dev_eval_worst_natural_timeout_rate(dev_eval_summary)
    tracker = load_checkpoint_tracker(training_paths)
    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not math.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int) or int(best_update_count) >= int(learner.update_count):
        return None
    best_score = float(best_metric_value)
    if best_score < float(checkpoint_guard.min_best_score):
        return None

    confidence = dev_eval_confidence_stats(dev_eval_summary)
    rollback_reasons: list[str] = []
    if current_score <= best_score - float(checkpoint_guard.rollback_score_margin):
        rollback_reasons.append("score_drop")
    if worst_stall_rate is not None and (
        worst_stall_rate >= float(checkpoint_guard.rollback_truncation_rate_threshold)
    ):
        rollback_reasons.append("truncation")
    max_prob_lt_half = confidence["max_prob_lt_half"]
    if max_prob_lt_half is not None and (float(max_prob_lt_half) >= float(checkpoint_guard.rollback_max_prob_lt_half)):
        rollback_reasons.append("confidence")
    if not rollback_reasons:
        return None

    best_checkpoint_path = training_paths.best_checkpoint_path
    restore_checkpoint(best_checkpoint_path, restore_counters=False)
    demoted_champions = demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    publish_metrics = runtime.maybe_publish_snapshot(
        learner_model=learner_model,
        learner_update_count=int(learner.update_count),
        force=True,
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()

    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "rollback_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "current_score": current_score,
        "best_score": best_score,
        "best_update_count": int(best_update_count),
        "worst_stall_rate": worst_stall_rate,
        "worst_truncation_rate": worst_truncation_rate,
        "worst_no_progress_timeout_rate": worst_no_progress_timeout_rate,
        "worst_natural_timeout_rate": worst_natural_timeout_rate,
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "reasons": rollback_reasons,
        "best_checkpoint_path": relative_path_text(best_checkpoint_path, root=run_dir),
        "latest_checkpoint_path": relative_path_text(training_paths.latest_checkpoint_path, root=run_dir),
        "rolled_back_checkpoint_path": relative_path_text(best_checkpoint_path, root=run_dir),
        "snapshot_publish_latency_ms": publish_metrics.get("snapshot_publish_latency_ms", 0.0),
        "snapshot_apply_latency_ms": publish_metrics.get("snapshot_apply_latency_ms", 0.0),
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "demoted_champions": demoted_champions,
    }
    append_checkpoint_guard_event(training_paths, payload)
    return payload


def maybe_finalize_from_best_checkpoint(
    *,
    stack: Any,
    training_paths: CheckpointAliasPaths,
    run_dir: Path,
    runtime: CheckpointGuardRuntime,
    learner: LearnerRecordSource,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
    restore_checkpoint: Any,
    ensure_current_checkpoint: Any,
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.checkpoint_guard.enabled:
        return None
    best_record = best_checkpoint_record(training_paths)
    if best_record is None:
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not math.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int):
        return None
    current_score = dev_eval_aggregate_score(dev_eval_summary)
    best_score = float(best_metric_value)
    if current_score is None or current_score >= best_score:
        return None
    confidence = dev_eval_confidence_stats(dev_eval_summary)
    best_checkpoint_path = training_paths.best_checkpoint_path
    restore_checkpoint(best_checkpoint_path, restore_counters=False)
    demoted_champions = demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()
    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "finalize_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "current_score": current_score,
        "best_score": best_score,
        "best_update_count": int(best_update_count),
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "best_checkpoint_path": relative_path_text(best_checkpoint_path, root=run_dir),
        "latest_checkpoint_path": relative_path_text(training_paths.latest_checkpoint_path, root=run_dir),
        "demoted_champions": demoted_champions,
    }
    append_checkpoint_guard_event(training_paths, payload)
    return payload


def resolve_resume_checkpoint_path(
    *,
    resume_from: str,
    resume_run_dir: Path | None,
) -> Path | None:
    normalized = str(resume_from).strip()
    if not normalized:
        if resume_run_dir is None:
            return None
        normalized = "latest"
    alias_name = normalized.lower()
    if alias_name in {"latest", "best", "observed_best"}:
        if resume_run_dir is None:
            raise ValueError("--resume-from latest|best|observed_best requires --resume-run-dir")
        filename_by_alias = {
            "latest": LATEST_CHECKPOINT_FILENAME,
            "best": BEST_CHECKPOINT_FILENAME,
            "observed_best": OBSERVED_BEST_CHECKPOINT_FILENAME,
        }
        filename = filename_by_alias[alias_name]
        checkpoint_path = Path(resume_run_dir).resolve() / "training" / "checkpoints" / filename
    else:
        checkpoint_path = Path(normalized).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def validate_checkpoint_payload_contract(
    payload: object,
    *,
    checkpoint_path: Path,
    expected_config_hash: str,
    expected_spec_hash256: str,
    algorithm: str,
    allow_config_mismatch: bool = False,
) -> CheckpointPayloadContract:
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict: {checkpoint_path}")
    if str(payload.get("format", "")).strip() != MINIMAL_TRAIN_CHECKPOINT_FORMAT:
        raise RuntimeError(f"unsupported checkpoint format in {checkpoint_path}")
    payload_config_hash = str(payload.get("config_hash256", "")).strip().lower()
    config_hash_mismatch = payload_config_hash != expected_config_hash
    if config_hash_mismatch and not allow_config_mismatch:
        raise RuntimeError(
            f"checkpoint config hash mismatch for {checkpoint_path}: "
            f"expected {expected_config_hash}, got {payload_config_hash}"
        )
    payload_spec_hash = payload.get("spec_hash256")
    if payload_spec_hash is not None and str(payload_spec_hash).strip().lower() != expected_spec_hash256:
        raise RuntimeError(
            f"checkpoint spec hash mismatch for {checkpoint_path}: "
            f"expected {expected_spec_hash256}, got {payload_spec_hash}"
        )
    payload_algorithm = payload.get("algorithm")
    if payload_algorithm is not None and str(payload_algorithm).strip() and str(payload_algorithm).strip() != algorithm:
        raise RuntimeError(
            f"checkpoint algorithm mismatch for {checkpoint_path}: expected {algorithm}, got {payload_algorithm}"
        )
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"checkpoint is missing a model_state_dict: {checkpoint_path}")
    return CheckpointPayloadContract(
        payload=payload,
        model_state_dict=model_state_dict,
        config_hash_mismatch=config_hash_mismatch,
        expected_config_hash=expected_config_hash,
        payload_config_hash=payload_config_hash,
    )


def build_minimal_train_checkpoint_payload(
    *,
    update_count: int,
    policy_version: int,
    device: str,
    config_hash256: str,
    spec_hash256: str | None,
    algorithm: str | None,
    recurrent_core: object,
    total_samples_processed: int,
    init_schedule_offset_updates: int,
    model_state_dict: dict[str, Any],
    policy_anchor_model_state_dict: dict[str, Any] | None,
    guidance_payload: dict[str, float],
    optimizer_state_dict: object,
    grad_scaler_state_dict: object,
) -> dict[str, Any]:
    return {
        "format": MINIMAL_TRAIN_CHECKPOINT_FORMAT,
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "device": str(device),
        "config_hash256": config_hash256,
        "spec_hash256": spec_hash256,
        "algorithm": algorithm,
        "recurrent_core": recurrent_core,
        "total_samples_processed": int(total_samples_processed),
        "init_schedule_offset_updates": int(init_schedule_offset_updates),
        "model_state_dict": model_state_dict,
        "policy_anchor_model_state_dict": policy_anchor_model_state_dict,
        **guidance_payload,
        "optimizer_state_dict": optimizer_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
    }


def write_minimal_train_checkpoint(
    *,
    checkpoint_path: Path,
    learner: CheckpointLearner,
    device: torch.device,
    config_hash256: str,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
    recurrent_core: object = None,
    guidance_payload: dict[str, float] | None = None,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Cannot write a checkpoint without a learner model")
    policy_anchor_state_fn = getattr(learner, "policy_anchor_state_dict", None)
    policy_anchor_model_state_dict = None if not callable(policy_anchor_state_fn) else policy_anchor_state_fn()
    payload = build_minimal_train_checkpoint_payload(
        update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        device=str(device),
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        recurrent_core=recurrent_core,
        total_samples_processed=int(getattr(learner, "total_samples_processed", 0)),
        init_schedule_offset_updates=int(getattr(learner, "init_schedule_offset_updates", 0)),
        model_state_dict=learner.model.state_dict(),
        policy_anchor_model_state_dict=policy_anchor_model_state_dict,
        guidance_payload={} if guidance_payload is None else guidance_payload,
        optimizer_state_dict=None if learner.optimizer is None else learner.optimizer.state_dict(),
        grad_scaler_state_dict=(
            None if getattr(learner, "_grad_scaler", None) is None else learner._grad_scaler.state_dict()
        ),
    )
    torch.save(payload, checkpoint_path)
    return payload


def restore_minimal_train_checkpoint(
    *,
    checkpoint_path: Path,
    learner: CheckpointLearner,
    device: torch.device,
    expected_config_hash: str,
    expected_spec_hash256: str,
    algorithm: str,
    restore_model_guidance: Any,
    allow_config_mismatch: bool = False,
    restore_counters: bool = True,
) -> ResumeCheckpoint:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    contract = validate_checkpoint_payload_contract(
        payload,
        checkpoint_path=checkpoint_path,
        expected_config_hash=expected_config_hash,
        expected_spec_hash256=expected_spec_hash256,
        algorithm=algorithm,
        allow_config_mismatch=allow_config_mismatch,
    )
    if contract.config_hash_mismatch:
        print(
            "Warning: allowing checkpoint config hash mismatch because "
            "WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH=1: "
            f"expected {contract.expected_config_hash}, got {contract.payload_config_hash}",
            file=sys.stderr,
        )
    if learner.model is None:
        raise RuntimeError(f"checkpoint is missing a model_state_dict: {checkpoint_path}")
    payload = contract.payload
    model_state_dict = contract.model_state_dict
    load_model_state_dict_with_context_compat(
        learner.model,
        model_state_dict,
        context=f"checkpoint resume {checkpoint_path}",
    )
    restore_model_guidance(learner.model, payload)
    load_policy_anchor_state_fn = getattr(learner, "load_policy_anchor_state_dict", None)
    if callable(load_policy_anchor_state_fn):
        anchor_state = payload.get("policy_anchor_model_state_dict")
        if anchor_state is not None and not isinstance(anchor_state, Mapping):
            raise RuntimeError(f"checkpoint policy_anchor_model_state_dict must be a dict: {checkpoint_path}")
        load_policy_anchor_state_fn(cast(Mapping[str, Any] | None, anchor_state))
    optimizer_state_dict = payload.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        optimizer = learner._optimizer_for_step()
        optimizer.load_state_dict(optimizer_state_dict)
    grad_scaler_state_dict = payload.get("grad_scaler_state_dict")
    if grad_scaler_state_dict is not None and getattr(learner, "_grad_scaler", None) is not None:
        learner._grad_scaler.load_state_dict(grad_scaler_state_dict)
    if restore_counters:
        learner.update_count = int(payload.get("update_count", 0))
        learner.policy_version = int(payload.get("policy_version", 0))
        learner.total_samples_processed = int(payload.get("total_samples_processed", 0))
        learner.start_time = time.time()
    init_schedule_offset_updates = int(payload.get("init_schedule_offset_updates", 0))
    cast(Any, learner).init_schedule_offset_updates = init_schedule_offset_updates
    return ResumeCheckpoint(
        checkpoint_path=checkpoint_path.resolve(),
        update_count=learner.update_count,
        policy_version=learner.policy_version,
        total_samples_processed=learner.total_samples_processed,
        init_schedule_offset_updates=init_schedule_offset_updates,
    )


def initialize_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    learner: CheckpointLearner,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
    restore_model_guidance: Any,
) -> ResumeCheckpoint:
    """Load model weights/guidance from a checkpoint without resuming counters or optimizer state."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    contract = validate_checkpoint_payload_contract(
        payload,
        checkpoint_path=checkpoint_path,
        expected_config_hash="",
        expected_spec_hash256=expected_spec_hash256,
        algorithm=algorithm,
        allow_config_mismatch=True,
    )
    if learner.model is None:
        raise RuntimeError(f"checkpoint is missing a model_state_dict: {checkpoint_path}")
    payload = contract.payload
    load_model_state_dict_with_context_compat(
        learner.model,
        contract.model_state_dict,
        context=f"checkpoint init {checkpoint_path}",
    )
    restore_model_guidance(learner.model, payload)
    reset_policy_anchor_fn = getattr(learner, "reset_policy_anchor_to_current_model", None)
    if callable(reset_policy_anchor_fn):
        reset_policy_anchor_fn()
    else:
        load_policy_anchor_state_fn = getattr(learner, "load_policy_anchor_state_dict", None)
        if callable(load_policy_anchor_state_fn):
            load_policy_anchor_state_fn(None)
    return ResumeCheckpoint(
        checkpoint_path=checkpoint_path.resolve(),
        update_count=int(payload.get("update_count", 0)),
        policy_version=int(payload.get("policy_version", 0)),
        total_samples_processed=int(payload.get("total_samples_processed", 0)),
        init_schedule_offset_updates=int(payload.get("init_schedule_offset_updates", 0)),
    )
