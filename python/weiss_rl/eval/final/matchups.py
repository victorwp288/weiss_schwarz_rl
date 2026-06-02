"""Single-matchup execution for canonical final evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval.final.matchup_outputs import (
    FinalEvalMatchupArtifactPaths,
    build_matchup_diagnostics,
    build_matchup_payload,
    build_posterior_samples_payload,
    matchup_artifact_paths,
    matchup_posterior_samples,
    write_matchup_artifacts,
)
from weiss_rl.eval.final.matchup_schedule import bootstrap_seed, matchup_dir_name, scheduled_game, slug
from weiss_rl.eval.harness import (
    EvalGameRecord,
    EvalGameRunner,
    ReplaySampleResult,
    record_completed_game,
    write_episodes_jsonl,
)
from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.stage2 import summarize_stage2_records


def run_final_eval_matchup(
    *,
    output_dir: Path,
    focal_index: int,
    opponent_index: int,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    stop_rules: StopRulesConfig,
    runner: EvalGameRunner,
    run_id256: str | bytes,
    config_hash256: str,
    spec_hash256: str,
    scheme: PayoffFoldScheme,
    sample_count: int,
) -> dict[str, Any]:
    matchup_dir = (
        output_dir
        / "matchups"
        / matchup_dir_name(
            focal_index=focal_index,
            opponent_index=opponent_index,
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
        )
    )
    records: list[EvalGameRecord] = []
    replay_samples: list[ReplaySampleResult] = []
    used_paired_seeds: list[int] = []

    for pair_index, episode_seed in enumerate(paired_seeds[:max_paired_seeds]):
        for swap_index in (0, 1):
            scheduled = scheduled_game(
                pair_index=pair_index,
                swap_index=swap_index,
                episode_seed=int(episode_seed),
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
            )
            result = runner.run_game(scheduled)
            if result.replay_sample is not None:
                replay_samples.append(result.replay_sample)
            records.append(
                record_completed_game(
                    scheduled_game=scheduled,
                    result=result,
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                )
            )
        used_paired_seeds.append(int(episode_seed))
        if len(used_paired_seeds) < stage1_paired_seeds:
            continue
        decision = summarize_stage2_records(
            records,
            stop_rules=stop_rules,
            max_paired_seeds=max_paired_seeds,
            scheme=scheme,
            sample_count=sample_count,
            seed=bootstrap_seed(focal_policy_id=focal_policy_id, opponent_policy_id=opponent_policy_id),
        )
        if decision.should_stop:
            break

    episodes_path = matchup_dir / "episodes.jsonl"
    write_episodes_jsonl(episodes_path, records)

    seed = bootstrap_seed(focal_policy_id=focal_policy_id, opponent_policy_id=opponent_policy_id)
    summary_payload = build_matchup_payload(
        records=records,
        stop_rules=stop_rules,
        max_paired_seeds=max_paired_seeds,
        scheme=scheme,
        sample_count=sample_count,
        seed=seed,
    )
    summary_payload["evaluation_context"] = {
        "artifact_scope": "final_eval",
        "focal_policy_index": focal_index,
        "opponent_policy_index": opponent_index,
        "stage1_paired_seeds": stage1_paired_seeds,
        "max_paired_seeds": max_paired_seeds,
        "used_paired_seeds": list(used_paired_seeds),
    }
    diagnostics_payload = build_matchup_diagnostics(records=records, runner=runner)
    posterior_samples = matchup_posterior_samples(
        records=records,
        scheme=scheme,
        sample_count=sample_count,
        seed=seed,
    )

    write_matchup_artifacts(
        matchup_dir=matchup_dir,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        sample_count=sample_count,
        summary_payload=summary_payload,
        diagnostics_payload=diagnostics_payload,
        posterior_samples=posterior_samples,
    )

    return {
        "focal_policy_id": focal_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "focal_index": focal_index,
        "opponent_index": opponent_index,
        "matchup_dir": matchup_dir,
        "episodes_path": episodes_path,
        "summary": summary_payload,
        "diagnostics": diagnostics_payload,
        "posterior_samples": posterior_samples,
        "used_paired_seeds": tuple(used_paired_seeds),
        "records": tuple(records),
        "replay_samples": tuple(replay_samples),
    }


__all__ = [
    "FinalEvalMatchupArtifactPaths",
    "bootstrap_seed",
    "build_matchup_diagnostics",
    "build_matchup_payload",
    "build_posterior_samples_payload",
    "matchup_artifact_paths",
    "matchup_dir_name",
    "matchup_posterior_samples",
    "run_final_eval_matchup",
    "scheduled_game",
    "slug",
    "write_matchup_artifacts",
]
