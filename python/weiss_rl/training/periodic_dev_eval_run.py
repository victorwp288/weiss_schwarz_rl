from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from weiss_rl.core.simulator_contract import SimulatorContract
from weiss_rl.model import PolicyValueModel
from weiss_rl.training.dev_eval.common import write_json
from weiss_rl.training.dev_eval.matchup_artifacts import build_and_write_periodic_matchup_artifacts
from weiss_rl.training.dev_eval.model_clone import clone_cpu_eval_model
from weiss_rl.training.dev_eval.opponents import periodic_dev_eval_opponents
from weiss_rl.training.dev_eval.plan import periodic_dev_eval_plan_payload
from weiss_rl.training.dev_eval.runtime_contracts import validate_periodic_dev_eval_contract
from weiss_rl.training.dev_eval.seed_schedule import (
    periodic_dev_eval_schedule,
)
from weiss_rl.training.dev_eval.summary_state import (
    persist_periodic_dev_eval_summary,
    update_stall_monitor,
)
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

        matchup_payload = build_and_write_periodic_matchup_artifacts(
            stack=stack,
            evaluation=evaluation,
            runner=runner,
            matchup_dir=matchup_dir,
            artifact_scope=artifact_scope,
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
            opponent_display_name=display_name,
            paired_seeds=paired_seeds,
            scheduled_paired_seeds=scheduled_paired_seeds,
            seed_file=seed_file,
            seed_file_sha256=seed_file_sha256,
            validated_sources=validated_sources,
            checkpoint_path=checkpoint_path,
            run_dir=artifacts.run_dir,
            update_count=update_count,
            policy_version=policy_version,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            write_json_fn=write_json_fn,
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
            "periodic_dev_eval_plan": periodic_dev_eval_plan_payload(),
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
