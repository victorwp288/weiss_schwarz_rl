"""Promotion-gate artifact assembly helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from weiss_rl.config import StackConfig
from weiss_rl.eval import paired_seed_scores, record_completed_game, summarize_game_records, write_episodes_jsonl
from weiss_rl.eval.harness import EvalGameRecord, EvalGameRunner
from weiss_rl.league import (
    PromotionGateAnchor,
    PromotionGateAnchorResult,
    PromotionGatePosterior,
    PromotionGateRate,
    PromotionGateResult,
    build_promotion_gate_result,
    resolve_promotion_gate_anchors,
    resolve_promotion_gate_seed_file,
)
from weiss_rl.repro import parse_seed_file
from weiss_rl.training.anchor_resolution import slug_policy_id
from weiss_rl.training.checkpoints import relative_path_text
from weiss_rl.training.eval_schedule import (
    PeriodicDevEvalOpponentSpec,
    PromotionGateSeedBlockJob,
    build_promotion_gate_seed_block_jobs,
    periodic_dev_eval_schedule_for_seed_items,
    resolved_promotion_gate_worker_devices,
    shard_promotion_gate_seed_block_jobs,
)


class PromotionGateRecordPaths(Protocol):
    run_dir: Path


@dataclass(frozen=True, slots=True)
class ParallelPromotionGatePlan:
    ordered_anchors: tuple[PromotionGateAnchor, ...]
    paired_seeds: tuple[int, ...]
    job_shards: tuple[tuple[PromotionGateSeedBlockJob, ...], ...]
    worker_devices: tuple[str, ...]


def format_promotion_gate_discarded_after_rollback_message(
    *,
    candidate_policy_id: str,
    candidate_update: int,
    rollback_best_update: int,
) -> str:
    return (
        "Promotion gate result discarded after rollback: "
        f"candidate={candidate_policy_id} "
        f"candidate_update={int(candidate_update)} "
        f"rollback_best_update={int(rollback_best_update)}"
    )


def format_promotion_gate_skipped_league_warmup_message(
    *,
    update_count: int,
    effective_update: int,
    threshold: int,
    candidate_policy_id: str,
) -> str:
    return (
        "Promotion gate skipped during league warmup: "
        f"update={int(update_count)} effective_update={int(effective_update)} "
        f"threshold={int(threshold)} candidate={candidate_policy_id}"
    )


def format_promotion_gate_skipped_eval_warmup_gate_message(
    *,
    update_count: int,
    effective_update: int,
    candidate_policy_id: str,
) -> str:
    return (
        "Promotion gate skipped during league eval warmup gate: "
        f"update={int(update_count)} effective_update={int(effective_update)} "
        f"candidate={candidate_policy_id}"
    )


def format_promotion_gate_missing_anchors_message(
    *,
    update_count: int,
    candidate_policy_id: str,
    missing_anchors: Sequence[str],
) -> str:
    return (
        "Promotion gate skipped: "
        f"update={int(update_count)} candidate={candidate_policy_id} "
        f"missing_anchors={','.join(missing_anchors)}"
    )


def format_scheduled_async_promotion_gate_message(
    *,
    update_count: int,
    candidate_policy_id: str,
    anchor_names: Sequence[str],
) -> str:
    return (
        "Scheduled async promotion gate: "
        f"update={int(update_count)} candidate={candidate_policy_id} "
        f"anchors={','.join(anchor_names)}"
    )


def format_optional_heuristic_public_anchors_skipped_message(exc: BaseException) -> str:
    return (
        "Promotion gate note: skipping optional heuristic-public anchors because the active simulator contract "
        f"does not expose the required public action/observation metadata ({exc})."
    )


def build_parallel_promotion_gate_plan(
    *,
    stack: StackConfig,
    anchor_policy_ids: Mapping[str, str],
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device: str,
) -> ParallelPromotionGatePlan:
    league = stack.config.league
    if league is None:
        raise RuntimeError("Parallel promotion gate requires stack.config.league")
    ordered_anchors = resolve_promotion_gate_anchors(stack, anchor_policy_ids)
    configured_parallel_workers = max(1, int(getattr(league.promotion_gate, "parallel_workers", 1)))
    seed_file = resolve_promotion_gate_seed_file(stack)
    paired_seeds = tuple(parse_seed_file(seed_file))
    if len(paired_seeds) != int(league.promotion_gate_paired_seeds):
        raise RuntimeError(
            f"Promotion gate expected {int(league.promotion_gate_paired_seeds)} paired seeds in {seed_file}, "
            f"found {len(paired_seeds)}"
        )
    seed_block_jobs = build_promotion_gate_seed_block_jobs(
        anchor_specs=anchor_specs,
        paired_seeds=paired_seeds,
        configured_parallel_workers=configured_parallel_workers,
    )
    effective_parallel_workers = min(configured_parallel_workers, max(1, len(seed_block_jobs)))
    worker_devices = resolved_promotion_gate_worker_devices(
        stack=stack,
        parallel_workers=max(1, effective_parallel_workers),
        explicit_worker_devices=tuple(getattr(league.promotion_gate, "parallel_worker_devices", ())),
        eval_device=eval_device,
    )
    job_shards = shard_promotion_gate_seed_block_jobs(
        jobs=seed_block_jobs,
        shard_count=max(1, effective_parallel_workers),
    )
    return ParallelPromotionGatePlan(
        ordered_anchors=tuple(ordered_anchors),
        paired_seeds=paired_seeds,
        job_shards=tuple(tuple(shard) for shard in job_shards),
        worker_devices=tuple(worker_devices),
    )


def promotion_gate_policy_maps(
    materialized_opponents: Sequence[tuple[str, str, object | None, object | None]],
) -> tuple[dict[str, object], dict[str, object]]:
    anchor_models = {
        policy_id: opponent_model
        for policy_id, _display_name, opponent_model, _heuristic_policy in materialized_opponents
        if opponent_model is not None
    }
    heuristic_policies = {
        policy_id: heuristic_policy
        for policy_id, _display_name, _opponent_model, heuristic_policy in materialized_opponents
        if heuristic_policy is not None
    }
    return anchor_models, heuristic_policies


def build_promotion_gate_worker_payloads(
    *,
    seed_block_jobs: Sequence[PromotionGateSeedBlockJob],
    runner: EvalGameRunner,
    candidate_policy_id: str,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
) -> list[dict[str, Any]]:
    worker_payloads: list[dict[str, Any]] = []
    for job in seed_block_jobs:
        scheduled_games = periodic_dev_eval_schedule_for_seed_items(
            focal_policy_id=candidate_policy_id,
            opponent_policy_id=job.anchor_spec.policy_id,
            paired_seed_items=job.paired_seed_items,
        )
        records = tuple(
            record_completed_game(
                scheduled_game=scheduled_game,
                result=runner.run_game(scheduled_game),
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
            )
            for scheduled_game in scheduled_games
        )
        worker_payloads.append(
            {
                "anchor_index": int(job.anchor_index),
                "block_index": int(job.block_index),
                "anchor_policy_id": job.anchor_spec.policy_id,
                "anchor_display_name": job.anchor_spec.display_name,
                "paired_seed_items": tuple(job.paired_seed_items),
                "records": records,
            }
        )
    return worker_payloads


def assemble_parallel_promotion_gate_result(
    *,
    stack: StackConfig,
    artifacts: PromotionGateRecordPaths,
    update_count: int,
    policy_version: int,
    focal_policy_id: str,
    anchors: Sequence[PromotionGateAnchor],
    records_by_anchor_index: Mapping[int, Sequence[EvalGameRecord]],
    paired_seeds: Sequence[int],
    sample_count: int = 1000,
) -> PromotionGateResult:
    promotion_run_dir = artifacts.run_dir / "eval" / "promotion_gate" / f"update_{int(update_count)}"
    episodes_dir = promotion_run_dir / "promotion_gate_episodes"
    bootstrap_seed = None
    if policy_version is not None:
        from weiss_rl.training.eval_seeds import promotion_gate_bootstrap_seed

        bootstrap_seed = promotion_gate_bootstrap_seed(
            update_count=int(update_count),
            policy_version=int(policy_version),
        )

    anchor_results: list[PromotionGateAnchorResult] = []
    all_pair_scores: list[float] = []
    total_truncated_games = 0
    total_games = 0
    for anchor_index, anchor in enumerate(anchors):
        records = sorted(
            records_by_anchor_index.get(anchor_index, ()),
            key=lambda record: (int(record.pair_index), int(record.swap_index), int(record.episode_index)),
        )
        episodes_path = episodes_dir / f"{anchor_index:02d}_{slug_policy_id(anchor.name)}.jsonl"
        write_episodes_jsonl(episodes_path, records)
        pair_scores = [float(score) for score in paired_seed_scores(records, scheme="S0")]
        truncated_games = sum(1 for record in records if record.truncated)
        anchor_results.append(
            PromotionGateAnchorResult(
                anchor_name=anchor.name,
                opponent_policy_id=anchor.policy_id,
                episodes_path=relative_path_text(episodes_path, root=artifacts.run_dir),
                matchup_summary=summarize_game_records(records),
                truncation=PromotionGateRate(
                    numerator=int(truncated_games),
                    denominator=int(len(records)),
                    rate=(float(truncated_games) / float(len(records))) if records else 0.0,
                ),
                posterior=PromotionGatePosterior.from_scores(
                    pair_scores,
                    sample_count=sample_count,
                    seed=bootstrap_seed,
                ),
            )
        )
        all_pair_scores.extend(pair_scores)
        total_truncated_games += int(truncated_games)
        total_games += int(len(records))

    result = build_promotion_gate_result(
        stack=stack,
        run_dir=promotion_run_dir,
        focal_policy_id=focal_policy_id,
        anchors=anchors,
        anchor_results=tuple(anchor_results),
        all_pair_scores=tuple(all_pair_scores),
        total_truncated_games=total_truncated_games,
        total_games=total_games,
        paired_seed_count=len(paired_seeds),
        sample_count=sample_count,
        bootstrap_seed=bootstrap_seed,
    )
    league = stack.config.league
    if league is None:
        raise RuntimeError("Promotion gate result assembly requires stack.config.league")
    result.write_json(promotion_run_dir / cast(str, league.promotion_gate.record_file))
    return result
