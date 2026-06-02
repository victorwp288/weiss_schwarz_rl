"""Runtime phases for parallel final-eval worker jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig, load_stack_config
from weiss_rl.config.models import EvaluationConfig
from weiss_rl.core.simulator_contract import SimulatorContract, load_verified_simulator_contract
from weiss_rl.eval.final.matchups import run_final_eval_matchup
from weiss_rl.eval.god_search import GodSearchConfig
from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.policies.types import ResolvedEvalPolicy
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies


@dataclass(frozen=True, slots=True)
class FinalEvalWorkerContext:
    stack: StackConfig
    run_dir: Path
    layout: ArtifactLayout
    output_dir: Path
    manifest: dict[str, Any]
    contract: SimulatorContract
    observation_dim: int
    action_dim: int
    pass_action_id: int
    evaluation: EvaluationConfig


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def worker_output_dir(job: dict[str, Any], *, layout: ArtifactLayout) -> Path:
    raw_output_dir = job.get("output_dir")
    if raw_output_dir is None or str(raw_output_dir).strip() == "":
        return layout.final_eval_dir
    return Path(str(raw_output_dir))


def unique_policy_resolution_ids(job: dict[str, Any]) -> list[str]:
    policy_pair = [str(job["focal_policy_id"]), str(job["opponent_policy_id"])]
    # Keep order unique on diagonal while still resolving the policy once.
    return list(dict.fromkeys(policy_pair))


def optional_job_path(job: dict[str, Any], key: str) -> Path | None:
    raw_value = job.get(key)
    if raw_value is None or str(raw_value).strip() == "":
        return None
    return Path(str(raw_value))


def load_final_eval_worker_context(job: dict[str, Any]) -> FinalEvalWorkerContext:
    stack = load_stack_config(Path(job["stack_config"]))
    run_dir = Path(job["run_dir"])
    layout = ArtifactLayout.from_run_dir(run_dir)
    output_dir = worker_output_dir(job, layout=layout)
    manifest = load_json_object(layout.manifest_path)
    contract = load_verified_simulator_contract(
        stack.root,
        expected_spec_hash=str(manifest.get("spec_hash256", "")).strip(),
    )
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")
    return FinalEvalWorkerContext(
        stack=stack,
        run_dir=run_dir,
        layout=layout,
        output_dir=output_dir,
        manifest=manifest,
        contract=contract,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=pass_action_id,
        evaluation=evaluation,
    )


def final_eval_worker_policy_resolution_kwargs(
    job: dict[str, Any],
    context: FinalEvalWorkerContext,
) -> dict[str, Any]:
    return {
        "stack": context.stack,
        "policy_ids": unique_policy_resolution_ids(job),
        "run_dir": context.run_dir,
        "observation_dim": context.observation_dim,
        "action_dim": context.action_dim,
        "spec_bundle": context.contract.spec_bundle,
        "snapshot_registry_path": optional_job_path(job, "snapshot_registry_json"),
        "b1_baseline_run_dir": optional_job_path(job, "b1_baseline_run_dir"),
    }


def resolve_final_eval_worker_policies(
    job: dict[str, Any],
    context: FinalEvalWorkerContext,
) -> dict[str, ResolvedEvalPolicy]:
    return resolve_eval_policies(**final_eval_worker_policy_resolution_kwargs(job, context))


def build_final_eval_worker_runner(
    job: dict[str, Any],
    context: FinalEvalWorkerContext,
    *,
    policies: dict[str, ResolvedEvalPolicy],
) -> SimulatorEvalRunner:
    return SimulatorEvalRunner(
        stack=context.stack,
        policies=policies,
        artifact_layout=context.layout,
        run_id256=str(context.manifest["run_id256"]),
        spec_hash256=str(context.manifest["spec_hash256"]),
        action_dim=context.action_dim,
        pass_action_id=context.pass_action_id,
        require_sorted_legal_ids=bool(context.evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=float(context.evaluation.replay_capture_rate_eval),
        regression_capture_count=int(context.evaluation.regression_capture_count),
        god_search_config=GodSearchConfig.from_mapping(job.get("god_search")),
    )


def run_final_eval_worker_matchup(
    job: dict[str, Any],
    context: FinalEvalWorkerContext,
    *,
    runner: SimulatorEvalRunner,
) -> dict[str, Any]:
    return run_final_eval_matchup(
        output_dir=context.output_dir,
        focal_index=int(job["focal_index"]),
        opponent_index=int(job["opponent_index"]),
        focal_policy_id=str(job["focal_policy_id"]),
        opponent_policy_id=str(job["opponent_policy_id"]),
        paired_seeds=list(job["paired_seeds"]),
        stage1_paired_seeds=int(job["stage1_paired_seeds"]),
        max_paired_seeds=int(job["max_paired_seeds"]),
        stop_rules=context.evaluation.stop_rules,
        runner=runner,
        run_id256=str(context.manifest["run_id256"]),
        config_hash256=str(context.manifest["config_hash256"]),
        spec_hash256=str(context.manifest["spec_hash256"]),
        scheme=cast(PayoffFoldScheme, str(job["scheme"])),
        sample_count=int(job["bootstrap_samples"]),
    )


def run_final_eval_worker(job: dict[str, Any]) -> dict[str, Any]:
    context = load_final_eval_worker_context(job)
    policies = resolve_final_eval_worker_policies(job, context)
    runner = build_final_eval_worker_runner(job, context, policies=policies)
    return run_final_eval_worker_matchup(job, context, runner=runner)


__all__ = [
    "FinalEvalWorkerContext",
    "build_final_eval_worker_runner",
    "final_eval_worker_policy_resolution_kwargs",
    "load_final_eval_worker_context",
    "load_json_object",
    "optional_job_path",
    "resolve_final_eval_worker_policies",
    "run_final_eval_worker",
    "run_final_eval_worker_matchup",
    "unique_policy_resolution_ids",
    "worker_output_dir",
]
