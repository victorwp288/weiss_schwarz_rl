from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.eval import (
    PayoffFoldScheme,
    build_matchup_export,
    build_seat_advantage_diagnostics,
    run_seat_swapped_matchup,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.model import PolicyValueModel
from weiss_rl.training.dev_eval import (
    clone_cpu_eval_model,
    json_relative_path,
    periodic_dev_eval_bootstrap_seed,
    periodic_dev_eval_schedule,
    periodic_dev_eval_seed_usage_payload,
    persist_periodic_dev_eval_summary,
    update_stall_monitor,
    validate_periodic_dev_eval_contract,
    write_json,
)
from weiss_rl.training.dev_eval.opponents import periodic_dev_eval_opponents
from weiss_rl.training.environments import spec_dimensions


class PeriodicDevEvalArtifacts(Protocol):
    run_dir: Path


class PeriodicDevEvalLearner(Protocol):
    model: PolicyValueModel | None
    update_count: int

    def get_policy_version(self) -> int: ...


CurrentCheckpointFn = Callable[..., Path]
PeriodicDevEvalRunnerFactory = Callable[..., Any]
SpecDimensionsFn = Callable[[SimulatorContract], tuple[int, int]]
CloneEvalModelFn = Callable[..., PolicyValueModel]
OpponentResolverFn = Callable[..., list[tuple[str, str, PolicyValueModel | None, Any | None]]]
PersistSummaryFn = Callable[..., None]
UpdateStallMonitorFn = Callable[..., dict[str, Any] | None]
WriteJsonFn = Callable[[Path, Any], None]


def run_periodic_dev_eval(
    *,
    stack: Any,
    contract: SimulatorContract,
    artifacts: PeriodicDevEvalArtifacts,
    training_paths: Any,
    learner: PeriodicDevEvalLearner,
    device: torch.device,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    runner_cls: PeriodicDevEvalRunnerFactory,
    ensure_current_checkpoint_fn: CurrentCheckpointFn,
    current_focal_policy_id_fn: Callable[..., str],
    artifact_dir_name: str = "dev_eval",
    artifact_scope: str = "periodic_dev_eval",
    paired_seeds_override: Sequence[int] | None = None,
    persist_summary: bool = True,
    update_stall_monitor_enabled: bool = True,
    spec_dimensions_fn: SpecDimensionsFn = spec_dimensions,
    clone_cpu_eval_model_fn: CloneEvalModelFn = clone_cpu_eval_model,
    periodic_dev_eval_opponents_fn: OpponentResolverFn = periodic_dev_eval_opponents,
    persist_summary_fn: PersistSummaryFn = persist_periodic_dev_eval_summary,
    update_stall_monitor_fn: UpdateStallMonitorFn = update_stall_monitor,
    write_json_fn: WriteJsonFn = write_json,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Periodic dev eval requires an attached learner model")

    evaluation = validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources, scheduled_paired_seeds, seed_file_sha256 = periodic_dev_eval_schedule(stack)
    paired_seeds = (
        [int(seed) for seed in paired_seeds_override]
        if paired_seeds_override is not None
        else [int(seed) for seed in scheduled_paired_seeds]
    )
    if not paired_seeds:
        raise RuntimeError("Periodic dev eval requires at least one paired seed")

    observation_dim, action_dim = spec_dimensions_fn(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    update_count = int(learner.update_count)
    policy_version = int(learner.get_policy_version())
    focal_policy_id = current_focal_policy_id_fn(learner=learner)
    checkpoint_path = ensure_current_checkpoint_fn(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=str(stack.config.training.algorithm).strip() if stack.config.training is not None else None,
    )

    update_dir = artifacts.run_dir / "eval" / artifact_dir_name / f"update_{update_count}"
    eval_model = clone_cpu_eval_model_fn(
        learner_model=learner.model,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
    )
    opponents = periodic_dev_eval_opponents_fn(
        stack=stack,
        contract=contract,
        run_dir=artifacts.run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )

    anchor_payloads: dict[str, dict[str, Any]] = {}
    anchor_scores: dict[str, float] = {}
    primary_summary: dict[str, Any] | None = None
    for opponent_policy_id, display_name, opponent_model, heuristic_policy in opponents:
        matchup_dir = update_dir / opponent_policy_id
        runner = runner_cls(
            stack=stack,
            model=eval_model,
            opponent_policy_id=opponent_policy_id,
            opponent_model=opponent_model,
            heuristic_policy=heuristic_policy,
            observation_dim=observation_dim,
            action_dim=action_dim,
            pass_action_id=pass_action_id,
            artifact_dir=matchup_dir,
            focal_policy_id=focal_policy_id,
            require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        )

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
            run_dir=artifacts.run_dir,
            opponent_policy_id=opponent_policy_id,
            opponent_display_name=display_name,
        )
        write_json_fn(matchup_dir / "seed_usage.json", seed_usage_payload)

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
            "checkpoint_path": json_relative_path(checkpoint_path, root=artifacts.run_dir),
            "seed_usage_path": json_relative_path(matchup_dir / "seed_usage.json", root=artifacts.run_dir),
            "anchor_display_name": display_name,
        }
        policy_alignment_summary = getattr(runner, "policy_alignment_summary", None)
        if callable(policy_alignment_summary):
            alignment_payload = policy_alignment_summary()
            if alignment_payload is not None:
                matchup_payload["policy_alignment_diagnostics"] = alignment_payload
        write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
        write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", matchup_payload)
        write_matchup_diagnostics_json(
            matchup_dir / "diagnostics.json",
            build_seat_advantage_diagnostics(matchup.records),
        )
        anchor_payloads[display_name] = matchup_payload
        anchor_scores[display_name] = float(matchup_payload["uncertainty"]["mean"])
        if primary_summary is None or opponent_policy_id == "b0_randomlegal":
            primary_summary = matchup_payload

    if primary_summary is None:
        raise RuntimeError("Periodic dev eval did not produce any matchup summaries")

    aggregate_score = sum(anchor_scores.values()) / max(1, len(anchor_scores))
    summary_payload = dict(primary_summary)
    summary_payload.update(
        {
            "policy_id": focal_policy_id,
            "update_count": update_count,
            "policy_version": policy_version,
            "aggregate_score": aggregate_score,
            "anchor_scores": anchor_scores,
            "anchors": anchor_payloads,
        }
    )
    if persist_summary:
        persist_summary_fn(training_paths=training_paths, payload=summary_payload)
    if update_stall_monitor_enabled:
        stall_monitor = update_stall_monitor_fn(
            stack=stack,
            training_paths=training_paths,
            update_count=update_count,
            summary_payload=summary_payload,
        )
        if stall_monitor is not None:
            summary_payload["stall_monitor"] = stall_monitor
            if bool(stall_monitor.get("stall_risk", False)):
                print(
                    "Stall monitor warning: "
                    f"update={update_count} worst_anchor={stall_monitor['worst_anchor']} "
                    f"stall_rate={float(stall_monitor['worst_stall_rate']):.3f} "
                    f"no_progress_rate={float(stall_monitor['worst_no_progress_timeout_rate']):.3f} "
                    f"truncation_rate={float(stall_monitor['worst_truncation_rate']):.3f} "
                    f"consecutive={int(stall_monitor['consecutive_trigger_count'])}"
                )
    write_json_fn(update_dir / "summary.json", summary_payload)
    return summary_payload
