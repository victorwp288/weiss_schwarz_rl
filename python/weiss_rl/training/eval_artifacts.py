"""Periodic dev-eval artifact records and follow-up audit requests."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from weiss_rl.config import StackConfig
from weiss_rl.repro import stable_hash64
from weiss_rl.training.checkpoints import relative_path_text
from weiss_rl.training.curriculum_guards import load_json_object, write_json
from weiss_rl.training.dev_eval_metrics import (
    b2_recent_scores_from_persisted_summaries,
    build_b2_warning_flags,
    dev_eval_aggregate_score,
    dev_eval_ineligibility_reasons,
    dev_eval_is_authoritative,
    dev_eval_surface,
    extract_anchor_payload,
    extract_anchor_score,
    extract_anchor_summary,
    extract_anchor_uncertainty,
    weighted_dev_eval_aggregate,
)
from weiss_rl.training.eval_schedule import json_relative_path

PERIODIC_DEV_EVAL_SUMMARY_FORMAT = "periodic_dev_eval_summary_v2"
B2_DISAGREEMENT_AUDIT_REQUESTS_FILENAME = "b2_disagreement_audit_requests.jsonl"


class TrainingLogPaths(Protocol):
    logs_dir: Path


class RunArtifactPaths(Protocol):
    run_dir: Path
    run_summary_path: Path


class LearnerProgress(Protocol):
    update_count: int

    def get_policy_version(self) -> int: ...


class PeriodicDevEvalOpponent(Protocol):
    policy_id: str
    display_name: str


def periodic_dev_eval_summaries_path(training_paths: TrainingLogPaths) -> Path:
    return training_paths.logs_dir / "periodic_dev_eval_summaries.json"


def periodic_dev_eval_fast_screens_path(training_paths: TrainingLogPaths) -> Path:
    return training_paths.logs_dir / "periodic_dev_eval_fast_screens.json"


def checkpoint_guard_log_path(training_paths: TrainingLogPaths) -> Path:
    return training_paths.logs_dir / "checkpoint_guard.jsonl"


def b2_disagreement_audit_requests_path(training_paths: TrainingLogPaths) -> Path:
    return training_paths.logs_dir / B2_DISAGREEMENT_AUDIT_REQUESTS_FILENAME


def append_checkpoint_guard_event(training_paths: TrainingLogPaths, payload: Mapping[str, Any]) -> None:
    path = checkpoint_guard_log_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def append_b2_disagreement_audit_request(training_paths: TrainingLogPaths, payload: Mapping[str, Any]) -> None:
    path = b2_disagreement_audit_requests_path(training_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def periodic_dev_eval_matchup_dir(
    *,
    update_dir: Path,
    opponent_spec: PeriodicDevEvalOpponent,
    duplicate_policy_ids: set[str],
) -> Path:
    if opponent_spec.policy_id not in duplicate_policy_ids:
        return update_dir / opponent_spec.policy_id
    display_hash = f"{stable_hash64(opponent_spec.display_name.encode('utf-8')):016x}"
    return update_dir / f"{opponent_spec.policy_id}__{display_hash}"


def build_periodic_dev_eval_seed_usage_payload(
    *,
    seed_file: Path,
    seed_root: Path,
    seed_file_sha256: str,
    validated_sources: Mapping[str, str],
    artifact_scope: str,
    scheduled_paired_seed_count: int,
    paired_seeds: Sequence[int],
    seat_swap: bool,
    eval_device: str,
    eval_inference_mode: bool,
    eval_sampling_algorithm: str,
    eval_assert_sorted_legal_ids: bool,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    checkpoint_path: Path,
    run_dir: Path,
    opponent_policy_id: str,
    opponent_display_name: str,
    parallel_seed_blocks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "seed_set": "dev_eval",
        "seed_file": {
            "path": json_relative_path(seed_file, root=seed_root),
            "sha256": seed_file_sha256,
            "validated_sources": dict(validated_sources),
        },
        "artifact_scope": artifact_scope,
        "seed_schedule": {
            "configured_paired_seed_count": int(scheduled_paired_seed_count),
            "requested_paired_seed_count": len(paired_seeds),
            "expanded_beyond_seed_file": len(paired_seeds) > int(scheduled_paired_seed_count),
        },
        "paired_seed_count": len(paired_seeds),
        "paired_seeds": [int(seed) for seed in paired_seeds],
        "protocol": {
            "seat_swap": bool(seat_swap),
            "eval_device": str(eval_device),
            "eval_inference_mode": bool(eval_inference_mode),
            "eval_sampling_algorithm": eval_sampling_algorithm,
            "eval_assert_sorted_legal_ids": bool(eval_assert_sorted_legal_ids),
        },
        "focal_policy": {
            "policy_id": focal_policy_id,
            "update_count": int(update_count),
            "policy_version": int(policy_version),
            "checkpoint_path": json_relative_path(checkpoint_path, root=run_dir),
        },
        "opponent_policy": {
            "policy_id": opponent_policy_id,
            "display_name": opponent_display_name,
        },
    }
    if parallel_seed_blocks is not None:
        payload["parallel_seed_blocks"] = [dict(block) for block in parallel_seed_blocks]
    return payload


def build_periodic_dev_eval_matchup_context_payload(
    *,
    artifact_scope: str,
    update_count: int,
    policy_version: int,
    checkpoint_path: Path,
    matchup_dir: Path,
    run_dir: Path,
    anchor_display_name: str,
) -> dict[str, Any]:
    return {
        "artifact_scope": artifact_scope,
        "update_count": int(update_count),
        "policy_version": int(policy_version),
        "checkpoint_path": json_relative_path(checkpoint_path, root=run_dir),
        "matchup_dir": json_relative_path(matchup_dir, root=run_dir),
        "episodes_path": json_relative_path(matchup_dir / "episodes.jsonl", root=run_dir),
        "seed_usage_path": json_relative_path(matchup_dir / "seed_usage.json", root=run_dir),
        "anchor_display_name": anchor_display_name,
    }


def build_periodic_dev_eval_matchup_runtime_payload(
    *,
    wall_clock_seconds: float,
    game_count: int,
    runner_counters: Mapping[str, Any],
    batched_model_inference: bool,
    persistent_env_reuse: bool = True,
    seed_block_count: int | None = None,
    serial_worker_wall_clock_seconds_sum: float | None = None,
) -> dict[str, Any]:
    wall_clock = max(0.0, float(wall_clock_seconds))
    payload: dict[str, Any] = {
        "wall_clock_seconds": wall_clock,
        "games_per_sec": float(int(game_count) / wall_clock) if wall_clock > 0.0 else 0.0,
        "game_count": int(game_count),
        "persistent_env_reuse": bool(persistent_env_reuse),
    }
    if seed_block_count is not None:
        payload["seed_block_count"] = int(seed_block_count)
    payload["batched_model_inference"] = bool(batched_model_inference)
    if serial_worker_wall_clock_seconds_sum is not None:
        payload["serial_worker_wall_clock_seconds_sum"] = float(serial_worker_wall_clock_seconds_sum)
    payload["runner_counters"] = dict(runner_counters)
    return payload


def sum_periodic_dev_eval_counter_payloads(counter_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for payload in counter_payloads:
        for key, value in cast(Mapping[str, Any], payload.get("seconds", {})).items():
            seconds[str(key)] = seconds.get(str(key), 0.0) + float(value)
        for key, value in cast(Mapping[str, Any], payload.get("counts", {})).items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
    return {
        "seconds": {key: float(value) for key, value in sorted(seconds.items())},
        "counts": {key: int(value) for key, value in sorted(counts.items())},
    }


def group_periodic_dev_eval_seed_block_results(
    block_results: Sequence[Mapping[str, Any]],
) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for block_result in block_results:
        grouped.setdefault(int(block_result["opponent_index"]), []).append(dict(block_result))
    return grouped


def collate_periodic_dev_eval_seed_block_matchup(
    *,
    block_results_by_opponent: Mapping[int, Sequence[Mapping[str, Any]]],
    opponent_index: int,
    opponent_display_name: str,
) -> dict[str, Any]:
    opponent_blocks = sorted(
        block_results_by_opponent.get(int(opponent_index), ()),
        key=lambda item: int(item["block_index"]),
    )
    if not opponent_blocks:
        raise RuntimeError(f"Periodic dev eval produced no seed-block results for {opponent_display_name}")

    records = tuple(
        sorted(
            (record for block_result in opponent_blocks for record in cast(Sequence[Any], block_result["records"])),
            key=lambda record: (int(record.pair_index), int(record.swap_index)),
        )
    )
    block_wall_clock_seconds = [float(block_result["wall_clock_seconds"]) for block_result in opponent_blocks]
    return {
        "records": records,
        "parallel_seed_blocks": [
            {
                "block_index": int(block_result["block_index"]),
                "paired_seed_items": [
                    {"pair_index": int(pair_index), "seed": int(seed)}
                    for pair_index, seed in cast(Sequence[tuple[int, int]], block_result["paired_seed_items"])
                ],
                "worker_index": int(block_result["worker_index"]),
                "worker_device": str(block_result["worker_device"]),
            }
            for block_result in opponent_blocks
        ],
        "wall_clock_seconds": max(block_wall_clock_seconds) if block_wall_clock_seconds else 0.0,
        "serial_worker_wall_clock_seconds_sum": float(sum(block_wall_clock_seconds)),
        "seed_block_count": int(len(opponent_blocks)),
        "runner_counters": sum_periodic_dev_eval_counter_payloads(
            [cast(Mapping[str, Any], block_result["runner_counters"]) for block_result in opponent_blocks]
        ),
    }


def promotion_gate_records_by_anchor_index(
    *,
    worker_payloads: Sequence[Mapping[str, Any]],
    anchor_count: int,
) -> dict[int, list[Any]]:
    records_by_anchor_index: dict[int, list[Any]] = {index: [] for index in range(int(anchor_count))}
    for payload in sorted(worker_payloads, key=lambda item: (int(item["anchor_index"]), int(item["block_index"]))):
        records_by_anchor_index[int(payload["anchor_index"])].extend(cast(Sequence[Any], payload["records"]))
    return records_by_anchor_index


def persist_periodic_dev_eval_fast_screen(
    *,
    training_paths: TrainingLogPaths,
    payload: Mapping[str, Any],
) -> None:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    if not focal_policy_id:
        return
    path = periodic_dev_eval_fast_screens_path(training_paths)
    summaries = load_json_object(path, label="periodic dev-eval fast screens") if path.is_file() else {}
    summaries[focal_policy_id] = {
        "format": "periodic_dev_eval_fast_screen_v1",
        "aggregate_score": payload.get("aggregate_score"),
        "anchor_scores": dict(cast(Mapping[str, Any], payload.get("anchor_scores", {}))),
        "update_count": int(payload.get("update_count", 0)),
        "policy_version": int(payload.get("policy_version", 0)),
        "evaluation_surface": dict(dev_eval_surface(payload)),
        "periodic_dev_eval_parallel": dict(cast(Mapping[str, Any], payload.get("periodic_dev_eval_parallel", {}))),
        "periodic_dev_eval_runtime": dict(cast(Mapping[str, Any], payload.get("periodic_dev_eval_runtime", {}))),
    }
    write_json(path, summaries)


def build_periodic_dev_eval_summary_record(
    *,
    payload: Mapping[str, Any],
    prior_summaries: Mapping[str, Any],
    b2_policy_id: str,
) -> dict[str, Any]:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    anchor_scores = dict(cast(Mapping[str, Any], payload.get("anchor_scores", {})))
    record: dict[str, Any] = {
        "format": PERIODIC_DEV_EVAL_SUMMARY_FORMAT,
        "aggregate_score": float(payload.get("aggregate_score", 0.0)),
        "anchor_scores": anchor_scores,
        "update_count": int(payload.get("update_count", 0)),
        "policy_version": int(payload.get("policy_version", 0)),
    }
    for optional_key in (
        "uncertainty",
        "periodic_dev_eval_parallel",
        "stall_monitor",
        "evaluation_surface",
        "aggregate_weighting",
    ):
        optional_payload = payload.get(optional_key)
        if isinstance(optional_payload, Mapping):
            record[optional_key] = dict(optional_payload)
    unweighted_aggregate_score = payload.get("unweighted_aggregate_score")
    if isinstance(unweighted_aggregate_score, (int, float)):
        record["unweighted_aggregate_score"] = float(unweighted_aggregate_score)
    anchors = payload.get("anchors")
    if isinstance(anchors, Mapping):
        record["anchors"] = dict(cast(Mapping[str, Any], anchors))

    b2_summary = extract_anchor_summary(payload, b2_policy_id)
    b2_uncertainty = extract_anchor_uncertainty(payload, b2_policy_id)
    b2_score = extract_anchor_score(payload, b2_policy_id)
    recent_b2_scores = b2_recent_scores_from_persisted_summaries(
        prior_summaries,
        current_policy_id=focal_policy_id,
        heuristic_public_policy_id=b2_policy_id,
    )
    b2_warning_flags = build_b2_warning_flags(
        current_score=b2_score,
        current_summary=b2_summary,
        recent_scores=recent_b2_scores,
    )
    if b2_score is not None or b2_summary is not None or b2_uncertainty is not None:
        record["b2"] = {
            "available": True,
            "score": None if b2_score is None else float(b2_score),
            "summary": None if b2_summary is None else dict(b2_summary),
            "uncertainty": None if b2_uncertainty is None else dict(b2_uncertainty),
            "warning_flags": b2_warning_flags,
        }

    record["warning_flags"] = [*b2_warning_flags]
    return record


def build_periodic_dev_eval_checkpoint_summary(
    *,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    matchup_results: Sequence[Mapping[str, Any]],
    anchor_weight_config: Mapping[str, float],
    effective_parallel_workers: int,
    worker_devices: Sequence[str],
    seed_block_job_count: int,
    batched_inference_enabled: bool,
    total_eval_wall_clock_seconds: float,
) -> dict[str, Any]:
    anchor_payloads: dict[str, dict[str, Any]] = {}
    anchor_scores: dict[str, float] = {}
    primary_summary: dict[str, Any] | None = None
    total_eval_games = 0
    for result in matchup_results:
        display_name = str(result["display_name"])
        opponent_policy_id = str(result["policy_id"])
        matchup_payload = dict(cast(Mapping[str, Any], result["matchup_payload"]))
        anchor_payloads[display_name] = matchup_payload
        anchor_scores[display_name] = float(cast(Mapping[str, Any], matchup_payload["uncertainty"])["mean"])
        runtime_payload = matchup_payload.get("evaluation_runtime", {})
        if isinstance(runtime_payload, Mapping):
            total_eval_games += int(runtime_payload.get("game_count", 0))
        if primary_summary is None or opponent_policy_id == "b0_randomlegal":
            primary_summary = matchup_payload

    if primary_summary is None:
        raise RuntimeError("Periodic dev eval did not produce any matchup summaries")

    unweighted_aggregate_score = sum(anchor_scores.values()) / max(1, len(anchor_scores))
    aggregate_score, aggregate_anchor_weights, aggregate_weight_sum = weighted_dev_eval_aggregate(
        anchor_scores,
        anchor_weights=anchor_weight_config,
    )
    summary_payload = dict(primary_summary)
    summary_payload.update(
        {
            "policy_id": focal_policy_id,
            "update_count": int(update_count),
            "policy_version": int(policy_version),
            "aggregate_score": aggregate_score,
            "unweighted_aggregate_score": unweighted_aggregate_score,
            "anchor_scores": anchor_scores,
            "anchor_seat_diagnostics": {
                str(result["display_name"]): dict(
                    cast(Mapping[str, Any], result["matchup_payload"]).get("seat_diagnostics", {})
                )
                for result in matchup_results
            },
            "aggregate_weighting": {
                "version": "periodic_dev_eval_anchor_weights_v1",
                "anchor_weights": aggregate_anchor_weights,
                "configured_anchor_weights": dict(anchor_weight_config),
                "total_weight": float(aggregate_weight_sum),
                "default_weight": 1.0,
            },
            "anchors": anchor_payloads,
            "periodic_dev_eval_parallel": {
                "enabled": int(effective_parallel_workers) > 1,
                "worker_count": int(max(1, effective_parallel_workers)),
                "worker_devices": list(worker_devices[: max(1, effective_parallel_workers)]),
                "job_count": int(seed_block_job_count),
                "batched_inference_enabled": bool(batched_inference_enabled),
                "seed_block_sharding_enabled": any(
                    int(
                        cast(Mapping[str, Any], result["matchup_payload"])
                        .get("evaluation_runtime", {})
                        .get("seed_block_count", 1)
                    )
                    > 1
                    for result in matchup_results
                ),
            },
            "periodic_dev_eval_runtime": {
                "wall_clock_seconds": float(total_eval_wall_clock_seconds),
                "games_per_sec": float(total_eval_games / total_eval_wall_clock_seconds)
                if total_eval_wall_clock_seconds > 0.0
                else 0.0,
                "game_count": int(total_eval_games),
                "persistent_env_reuse": True,
            },
            "evaluation_surface": {
                "kind": "fast_batched_screen" if batched_inference_enabled else "canonical_scalar",
                "authoritative": not batched_inference_enabled,
                "batched_inference_enabled": bool(batched_inference_enabled),
            },
        }
    )
    return summary_payload


def persist_periodic_dev_eval_summary(
    *,
    training_paths: TrainingLogPaths,
    payload: Mapping[str, Any],
    b2_policy_id: str,
) -> None:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    if not focal_policy_id:
        return
    path = periodic_dev_eval_summaries_path(training_paths)
    summaries = load_json_object(path, label="periodic dev-eval summaries") if path.is_file() else {}
    summaries[focal_policy_id] = build_periodic_dev_eval_summary_record(
        payload=payload,
        prior_summaries=summaries,
        b2_policy_id=b2_policy_id,
    )
    write_json(path, summaries)


def persist_periodic_dev_eval_result(
    *,
    training_paths: TrainingLogPaths,
    payload: Mapping[str, Any],
    b2_policy_id: str,
    force_summary: bool = False,
) -> str:
    if force_summary or dev_eval_is_authoritative(payload):
        persist_periodic_dev_eval_summary(
            training_paths=training_paths,
            payload=payload,
            b2_policy_id=b2_policy_id,
        )
        return "summary"
    persist_periodic_dev_eval_fast_screen(training_paths=training_paths, payload=payload)
    return "fast_screen"


def run_stack_config_path(artifacts: RunArtifactPaths) -> Path | None:
    if not artifacts.run_summary_path.is_file():
        return None
    run_summary = load_json_object(artifacts.run_summary_path, label="run summary")
    raw_path = run_summary.get("stack_config_path")
    if not isinstance(raw_path, str) or not str(raw_path).strip():
        return None
    return Path(raw_path)


def dev_eval_has_confidence_only_block(dev_eval_summary: Mapping[str, Any] | None, *, stack: StackConfig) -> bool:
    reasons = dev_eval_ineligibility_reasons(stack, dev_eval_summary=dev_eval_summary)
    return bool(reasons) and all(reason in {"confidence_prob", "confidence_ci"} for reason in reasons)


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
    training_paths: TrainingLogPaths,
    learner: LearnerProgress,
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


def maybe_request_b2_disagreement_audit(
    *,
    stack: StackConfig,
    training_paths: TrainingLogPaths,
    artifacts: RunArtifactPaths,
    dev_eval_summary: Mapping[str, Any] | None,
    b2_policy_id: str,
) -> dict[str, Any] | None:
    if dev_eval_summary is None:
        return None
    if not dev_eval_is_authoritative(dev_eval_summary):
        return None
    b2_payload = extract_anchor_payload(dev_eval_summary, b2_policy_id)
    if b2_payload is None:
        return None
    evaluation_context = b2_payload.get("evaluation_context")
    if not isinstance(evaluation_context, Mapping):
        return None
    episodes_rel = evaluation_context.get("episodes_path")
    if not isinstance(episodes_rel, str) or not str(episodes_rel).strip():
        return None
    episodes_path = artifacts.run_dir / str(episodes_rel)
    if not episodes_path.is_file():
        return None

    triggers: list[str] = []
    b2_record = dev_eval_summary.get("b2")
    if isinstance(b2_record, Mapping):
        warning_flags = b2_record.get("warning_flags")
        if isinstance(warning_flags, Sequence):
            for warning in warning_flags:
                if not isinstance(warning, Mapping):
                    continue
                if str(warning.get("kind", "")).strip() == "b2_flatline_v1":
                    triggers.append("b2_flatline")
                    break
    if dev_eval_has_confidence_only_block(dev_eval_summary, stack=stack):
        triggers.append("confidence_only_gate")
    if not triggers:
        return None

    canonical_stack_config_path = artifacts.run_dir / "config_canonical.json"
    stack_config_path = (
        canonical_stack_config_path if canonical_stack_config_path.is_file() else run_stack_config_path(artifacts)
    )
    update_count = int(dev_eval_summary.get("update_count", 0))
    policy_version = int(dev_eval_summary.get("policy_version", 0))
    audit_policy_id = (
        f"policy_{policy_version:06d}" if policy_version > 0 else str(dev_eval_summary.get("policy_id", ""))
    )
    output_run_dir = artifacts.run_dir / "eval" / "b2_disagreement_audit" / f"update_{update_count}"
    command: list[str] = []
    if stack_config_path is not None:
        command = [
            sys.executable,
            "python/scripts/b2_disagreement_audit.py",
            "--stack-config",
            stack_config_path.as_posix(),
            "--run-dir",
            artifacts.run_dir.as_posix(),
            "--output-run-dir",
            output_run_dir.as_posix(),
            "--episodes-jsonl",
            episodes_path.as_posix(),
            "--policy-id",
            audit_policy_id,
            "--summary-json",
            (output_run_dir / "audit" / "summary.json").as_posix(),
        ]

    payload = {
        "format": "b2_disagreement_audit_request_v1",
        "event_kind": "b2_disagreement_audit_requested_v1",
        "trigger_reasons": list(dict.fromkeys(triggers)),
        "update_count": update_count,
        "policy_version": policy_version,
        "policy_id": str(dev_eval_summary.get("policy_id", "")),
        "audit_policy_id": audit_policy_id,
        "b2_score": extract_anchor_score(dev_eval_summary, b2_policy_id),
        "episodes_path": relative_path_text(episodes_path, root=artifacts.run_dir),
        "output_run_dir": relative_path_text(output_run_dir, root=artifacts.run_dir),
        "command": command,
    }
    append_b2_disagreement_audit_request(training_paths, payload)
    append_checkpoint_guard_event(training_paths, payload)
    return payload


def format_b2_disagreement_audit_request_message(audit_request: Mapping[str, Any]) -> str:
    return (
        "B2 disagreement audit requested: "
        f"update={int(audit_request['update_count'])} "
        f"reasons={','.join(str(reason) for reason in cast(Sequence[Any], audit_request['trigger_reasons']))} "
        f"episodes={audit_request['episodes_path']}"
    )


def format_periodic_dev_eval_console_message(
    *,
    label: str,
    update_count: int,
    aggregate_score: float,
    anchor_names: Sequence[str],
    opponent_slug: str | None = None,
) -> str:
    opponent_fragment = f" opponent={opponent_slug}" if opponent_slug else ""
    return (
        f"{label}: "
        f"update={int(update_count)}{opponent_fragment} "
        f"aggregate={float(aggregate_score):.4f} "
        f"anchors={','.join(anchor_names)}"
    )


def format_periodic_dev_eval_scheduled_message(
    *,
    update_count: int,
    worker_devices: Sequence[str],
    fallback_eval_device: str,
    anchor_names: Sequence[str],
) -> str:
    devices = ",".join(worker_devices) or str(fallback_eval_device)
    return f"Periodic dev eval scheduled: update={int(update_count)} devices={devices} anchors={','.join(anchor_names)}"
