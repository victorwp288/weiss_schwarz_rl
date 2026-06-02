from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState


@dataclass(frozen=True)
class CanonicalEvalPolicyRuntime:
    policy_ids: list[str]
    selection_details: dict[str, Any]
    snapshot_registry_path: Path | None
    dev_eval_summaries_path: Path | None
    runner: Any


def resolve_canonical_eval_policy_runtime(
    *,
    stack: Any,
    run_dir: Path,
    policy_ids: list[str],
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    b1_baseline_run_dir: Path | None,
    run_state: CanonicalEvalRunState,
    dependencies: Any,
) -> CanonicalEvalPolicyRuntime:
    layout = run_state.layout
    evaluation = run_state.evaluation
    (
        resolved_policy_ids,
        selection_details,
        resolved_registry_path,
        resolved_dev_eval_path,
    ) = dependencies.resolve_policy_ids_for_run_fn(
        policy_ids=policy_ids,
        stack=stack,
        manifest=run_state.manifest,
        layout=layout,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
    )
    dependencies.persist_policy_selection_in_manifest_fn(
        layout=layout,
        manifest=run_state.manifest,
        policy_ids=resolved_policy_ids,
        selection_details=selection_details,
    )

    contract = dependencies.load_verified_simulator_contract_fn(
        stack.root,
        expected_spec_hash=str(run_state.manifest.get("spec_hash256", "")).strip(),
    )
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    resolved_policies = dependencies.resolve_eval_policies_fn(
        stack=stack,
        policy_ids=resolved_policy_ids,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        spec_bundle=contract.spec_bundle,
        snapshot_registry_path=resolved_registry_path,
        b1_baseline_run_dir=b1_baseline_run_dir,
    )
    runner = dependencies.simulator_eval_runner_cls(
        stack=stack,
        policies=resolved_policies,
        artifact_layout=layout,
        run_id256=run_state.run_id256,
        spec_hash256=str(run_state.manifest["spec_hash256"]),
        action_dim=action_dim,
        pass_action_id=pass_action_id,
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=float(evaluation.replay_capture_rate_eval),
        regression_capture_count=int(evaluation.regression_capture_count),
    )
    return CanonicalEvalPolicyRuntime(
        policy_ids=resolved_policy_ids,
        selection_details=selection_details,
        snapshot_registry_path=resolved_registry_path,
        dev_eval_summaries_path=resolved_dev_eval_path,
        runner=runner,
    )


def resolve_recommended_focal_policy_id(
    *,
    policy_ids: list[str],
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    dependencies: Any,
) -> str | None:
    if snapshot_registry_path is None or dev_eval_summaries_path is None:
        return None
    try:
        from weiss_rl.league.registry import SnapshotRegistry

        return dependencies.recommend_focal_policy_id_fn(
            snapshot_registry=SnapshotRegistry.load(snapshot_registry_path),
            dev_eval_summaries=dependencies.load_dev_eval_summaries_fn(dev_eval_summaries_path),
            candidate_policy_ids=policy_ids,
        )
    except Exception:
        return None


__all__ = [
    "CanonicalEvalPolicyRuntime",
    "resolve_canonical_eval_policy_runtime",
    "resolve_recommended_focal_policy_id",
]
