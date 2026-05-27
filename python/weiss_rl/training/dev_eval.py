from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, hash_seed_file, parse_seed_file, stable_hash64
from weiss_rl.core.masking import assert_strictly_increasing_legal_ids
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.training.guidance import model_guidance_payload, restore_model_guidance_from_payload


class DevEvalTrainingPaths(Protocol):
    logs_dir: Path


def evaluation_config_or_raise(stack: Any) -> Any:
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise RuntimeError("The locked stack is missing the evaluation config block")
    return evaluation


def validate_periodic_dev_eval_contract(stack: Any) -> Any:
    evaluation = evaluation_config_or_raise(stack)
    if not evaluation.seat_swap:
        raise RuntimeError("Periodic dev eval requires evaluation.seat_swap=true")
    if evaluation.eval_device != "cpu":
        raise RuntimeError(f"Periodic dev eval requires evaluation.eval_device='cpu', got {evaluation.eval_device!r}")
    if not evaluation.eval_inference_mode:
        raise RuntimeError("Periodic dev eval requires evaluation.eval_inference_mode=true")
    if evaluation.eval_sampling_algorithm not in {"pinned_cdf_pcg_v1", "model_argmax_pinned_v1"}:
        raise RuntimeError(
            "Periodic dev eval requires evaluation.eval_sampling_algorithm='pinned_cdf_pcg_v1' "
            "or 'model_argmax_pinned_v1', "
            f"got {evaluation.eval_sampling_algorithm!r}"
        )
    return evaluation


def resolve_repo_path(root: Path, path_text: str) -> Path:
    candidate = Path(path_text)
    return candidate if candidate.is_absolute() else root / candidate


def json_relative_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def resolve_periodic_dev_eval_seed_file(stack: Any) -> tuple[Path, dict[str, str]]:
    evaluation = evaluation_config_or_raise(stack)
    reproducibility = stack.config.reproducibility
    resolved_paths: dict[str, Path] = {}
    if "dev_eval" in stack.seed_sets:
        resolved_paths["stack.seed_sets.dev_eval"] = stack.seed_sets["dev_eval"]
    if "dev_eval" in evaluation.seed_files:
        resolved_paths["evaluation.seed_files.dev_eval"] = resolve_repo_path(
            stack.root,
            evaluation.seed_files["dev_eval"],
        )
    if reproducibility is not None and "dev_eval" in reproducibility.seed_files:
        resolved_paths["reproducibility.seed_files.dev_eval"] = resolve_repo_path(
            stack.root,
            reproducibility.seed_files["dev_eval"],
        )
    if not resolved_paths:
        raise RuntimeError("Periodic dev eval requires a configured dev_eval seed file")

    unique_paths = {path.resolve() for path in resolved_paths.values()}
    if len(unique_paths) != 1:
        mismatch = {name: json_relative_path(path, root=stack.root) for name, path in resolved_paths.items()}
        raise RuntimeError(f"Periodic dev eval seed file mismatch: {mismatch}")

    seed_file = next(iter(resolved_paths.values()))
    return seed_file, {name: json_relative_path(path, root=stack.root) for name, path in resolved_paths.items()}


def periodic_dev_eval_schedule(stack: Any) -> tuple[Path, dict[str, str], list[int], str]:
    evaluation = validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources = resolve_periodic_dev_eval_seed_file(stack)
    all_paired_seeds = parse_seed_file(seed_file)
    required_pairs = int(evaluation.periodic_dev_eval_paired_seeds)
    if len(all_paired_seeds) < required_pairs:
        raise RuntimeError(
            f"Periodic dev eval requires {required_pairs} paired seeds, found {len(all_paired_seeds)} in {seed_file}"
        )
    return seed_file, validated_sources, all_paired_seeds[:required_pairs], hash_seed_file(seed_file)


def periodic_dev_eval_seed_usage_payload(
    *,
    seed_file: Path,
    seed_file_root: Path,
    seed_file_sha256: str,
    validated_sources: Mapping[str, str],
    artifact_scope: str,
    scheduled_paired_seeds: list[int],
    paired_seeds: list[int],
    evaluation: Any,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    checkpoint_path: Path | None,
    run_dir: Path,
    opponent_policy_id: str,
    opponent_display_name: str,
) -> dict[str, Any]:
    return {
        "seed_set": "dev_eval",
        "seed_file": {
            "path": json_relative_path(seed_file, root=seed_file_root),
            "sha256": seed_file_sha256,
            "validated_sources": dict(validated_sources),
        },
        "artifact_scope": artifact_scope,
        "seed_schedule": {
            "configured_paired_seed_count": len(scheduled_paired_seeds),
            "requested_paired_seed_count": len(paired_seeds),
            "expanded_beyond_seed_file": len(paired_seeds) > len(scheduled_paired_seeds),
        },
        "paired_seed_count": len(paired_seeds),
        "paired_seeds": list(paired_seeds),
        "protocol": {
            "seat_swap": bool(evaluation.seat_swap),
            "eval_device": evaluation.eval_device,
            "eval_inference_mode": bool(evaluation.eval_inference_mode),
            "eval_sampling_algorithm": evaluation.eval_sampling_algorithm,
            "model_sampling_temperature": float(getattr(evaluation, "model_sampling_temperature", 1.0)),
            "eval_assert_sorted_legal_ids": bool(evaluation.eval_assert_sorted_legal_ids),
        },
        "focal_policy": {
            "policy_id": focal_policy_id,
            "update_count": int(update_count),
            "policy_version": int(policy_version),
            "checkpoint_path": None if checkpoint_path is None else json_relative_path(checkpoint_path, root=run_dir),
        },
        "opponent_policy": {
            "policy_id": opponent_policy_id,
            "display_name": opponent_display_name,
        },
    }


def legal_ids_for_env_row(
    *,
    batch: Any,
    env_index: int,
    require_sorted: bool,
) -> np.ndarray:
    if batch.ids_offsets is None:
        raise RuntimeError("Expected ids_offsets legality during periodic dev eval")
    legal_ids, legal_offsets = batch.ids_offsets
    start = int(legal_offsets[env_index])
    end = int(legal_offsets[env_index + 1])
    row = np.asarray(legal_ids[start:end], dtype=np.uint32)
    if require_sorted:
        assert_strictly_increasing_legal_ids(row)
    return row


def _scheduled_game_rng_seed(*, kind: str, scheduled_game: Any, seat: int) -> int:
    payload = canonical_json_bytes(
        {
            "kind": kind,
            "pair_index": scheduled_game.pair_index,
            "swap_index": scheduled_game.swap_index,
            "episode_seed": scheduled_game.episode_seed,
            "seat": int(seat),
            "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
        }
    )
    return stable_hash64(payload)


def periodic_dev_eval_rng_seed(*, scheduled_game: Any, seat: int) -> int:
    return _scheduled_game_rng_seed(
        kind="periodic_dev_eval_rng_v1",
        scheduled_game=scheduled_game,
        seat=seat,
    )


def promotion_gate_rng_seed(*, scheduled_game: Any, seat: int) -> int:
    return _scheduled_game_rng_seed(
        kind="promotion_gate_rng_v1",
        scheduled_game=scheduled_game,
        seat=seat,
    )


def periodic_dev_eval_bootstrap_seed(*, update_count: int, policy_version: int) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "periodic_dev_eval_bootstrap_v1",
                "update_count": int(update_count),
                "policy_version": int(policy_version),
            }
        )
    )


def promotion_gate_bootstrap_seed(*, update_count: int, policy_version: int) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "promotion_gate_bootstrap_v1",
                "update_count": int(update_count),
                "policy_version": int(policy_version),
            }
        )
    )


def should_run_periodic_dev_eval(stack: Any, *, update_count: int) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    interval = int(evaluation.periodic_dev_eval_interval_updates)
    return interval > 0 and update_count % interval == 0


def periodic_dev_eval_summaries_path(training_paths: DevEvalTrainingPaths) -> Path:
    return training_paths.logs_dir / "periodic_dev_eval_summaries.json"


def stall_monitor_state_path(training_paths: DevEvalTrainingPaths) -> Path:
    return training_paths.logs_dir / "stall_monitor.json"


def clone_cpu_eval_model(
    *,
    learner_model: PolicyValueModel,
    observation_dim: int,
    action_dim: int,
    stack: Any,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    """Clone the learner model into an eval-mode CPU model for deterministic eval."""

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")
    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(torch.device("cpu"))
    cpu_state_dict = {name: value.detach().cpu().clone() for name, value in learner_model.state_dict().items()}
    eval_model.load_state_dict(cpu_state_dict)
    restore_model_guidance_from_payload(eval_model, model_guidance_payload(learner_model))
    eval_model.eval()
    return eval_model


def summary_rate(matchup_summary: Mapping[str, Any], key: str) -> float | None:
    games = matchup_summary.get("games")
    count = matchup_summary.get(key)
    if not isinstance(games, (int, float)) or not isinstance(count, (int, float)):
        return None
    if float(games) <= 0.0:
        return None
    return float(count) / float(games)


def persist_periodic_dev_eval_summary(
    *,
    training_paths: DevEvalTrainingPaths,
    payload: Mapping[str, Any],
) -> None:
    focal_policy_id = str(payload.get("policy_id", "")).strip()
    if not focal_policy_id:
        return
    path = periodic_dev_eval_summaries_path(training_paths)
    summaries = load_json_object(path, label="periodic dev-eval summaries") if path.is_file() else {}
    summaries[focal_policy_id] = {
        "aggregate_score": float(payload.get("aggregate_score", 0.0)),
        "anchor_scores": dict(cast(Mapping[str, Any], payload.get("anchor_scores", {}))),
        "update_count": int(payload.get("update_count", 0)),
        "policy_version": int(payload.get("policy_version", 0)),
    }
    write_json(path, summaries)


def update_stall_monitor(
    *,
    stack: Any,
    training_paths: DevEvalTrainingPaths,
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
        anchor_truncation_rates[str(anchor_name)] = 0.0 if truncation_rate is None else truncation_rate
        anchor_no_progress_rates[str(anchor_name)] = 0.0 if no_progress_rate is None else no_progress_rate
        anchor_natural_timeout_rates[str(anchor_name)] = 0.0 if natural_timeout_rate is None else natural_timeout_rate
        anchor_stall_rates[str(anchor_name)] = (
            anchor_no_progress_rates[str(anchor_name)]
            if no_progress_rate is not None
            else anchor_truncation_rates[str(anchor_name)]
        )
    if not anchor_stall_rates:
        return None

    state_path = stall_monitor_state_path(training_paths)
    state = load_json_object(state_path, label="stall monitor state") if state_path.is_file() else {}
    previous_consecutive = int(state.get("consecutive_trigger_count", 0))
    worst_anchor = max(anchor_stall_rates, key=lambda anchor_name: anchor_stall_rates[anchor_name])
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
