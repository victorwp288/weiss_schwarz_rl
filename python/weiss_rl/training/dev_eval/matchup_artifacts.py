"""Per-opponent artifact writing for periodic development evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from weiss_rl.eval import (
    PayoffFoldScheme,
    build_matchup_export,
    build_seat_advantage_diagnostics,
    run_seat_swapped_matchup,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.training.dev_eval.common import json_relative_path
from weiss_rl.training.dev_eval.seed_schedule import (
    periodic_dev_eval_bootstrap_seed,
    periodic_dev_eval_seed_usage_payload,
)

WriteJsonFn = Callable[[Path, Any], None]


def build_and_write_periodic_matchup_artifacts(
    *,
    stack: Any,
    evaluation: Any,
    runner: Any,
    matchup_dir: Path,
    artifact_scope: str,
    focal_policy_id: str,
    opponent_policy_id: str,
    opponent_display_name: str,
    paired_seeds: Sequence[int],
    scheduled_paired_seeds: Sequence[int],
    seed_file: Path,
    seed_file_sha256: str,
    validated_sources: Sequence[Any],
    checkpoint_path: Path,
    run_dir: Path,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    write_json_fn: WriteJsonFn,
) -> dict[str, Any]:
    seed_usage_path = matchup_dir / "seed_usage.json"
    seed_usage_payload = periodic_dev_eval_seed_usage_payload(
        seed_file=seed_file,
        seed_file_root=stack.root,
        seed_file_sha256=seed_file_sha256,
        validated_sources=validated_sources,
        artifact_scope=artifact_scope,
        scheduled_paired_seeds=scheduled_paired_seeds,
        paired_seeds=paired_seeds,
        evaluation=evaluation,
        focal_policy_id=focal_policy_id,
        update_count=update_count,
        policy_version=policy_version,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        opponent_policy_id=opponent_policy_id,
        opponent_display_name=opponent_display_name,
    )
    write_json_fn(seed_usage_path, seed_usage_payload)

    matchup = run_seat_swapped_matchup(
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        paired_seeds=paired_seeds,
        runner=runner,
        episodes_path=matchup_dir / "episodes.jsonl",
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
    )

    matchup_payload = build_matchup_export(
        matchup.records,
        stop_rules=evaluation.stop_rules,
        max_paired_seeds=len(paired_seeds),
        scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
        sample_count=1000,
        seed=periodic_dev_eval_bootstrap_seed(update_count=update_count, policy_version=policy_version),
    )
    matchup_payload["evaluation_context"] = {
        "artifact_scope": artifact_scope,
        "update_count": update_count,
        "policy_version": policy_version,
        "checkpoint_path": json_relative_path(checkpoint_path, root=run_dir),
        "seed_usage_path": json_relative_path(seed_usage_path, root=run_dir),
        "anchor_display_name": opponent_display_name,
    }
    _add_policy_alignment_diagnostics(matchup_payload, runner)

    write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
    write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", matchup_payload)
    write_matchup_diagnostics_json(
        matchup_dir / "diagnostics.json",
        build_seat_advantage_diagnostics(matchup.records),
    )
    return matchup_payload


def _add_policy_alignment_diagnostics(matchup_payload: dict[str, Any], runner: Any) -> None:
    policy_alignment_summary = getattr(runner, "policy_alignment_summary", None)
    if not callable(policy_alignment_summary):
        return
    alignment_payload = policy_alignment_summary()
    if alignment_payload is not None:
        matchup_payload["policy_alignment_diagnostics"] = alignment_payload


__all__ = ["build_and_write_periodic_matchup_artifacts"]
