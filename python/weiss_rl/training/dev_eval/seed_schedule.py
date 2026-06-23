"""Seed schedule and deterministic RNG helpers for periodic dev-eval."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, hash_seed_file, parse_seed_file, stable_hash64
from weiss_rl.training.dev_eval.common import json_relative_path, resolve_repo_path
from weiss_rl.training.dev_eval.runtime_contracts import (
    evaluation_config_or_raise,
    validate_periodic_dev_eval_contract,
)


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


__all__ = [
    "periodic_dev_eval_bootstrap_seed",
    "periodic_dev_eval_rng_seed",
    "periodic_dev_eval_schedule",
    "periodic_dev_eval_seed_usage_payload",
    "promotion_gate_bootstrap_seed",
    "promotion_gate_rng_seed",
    "resolve_periodic_dev_eval_seed_file",
]
