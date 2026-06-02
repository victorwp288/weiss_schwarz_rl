"""Promotion-gate execution orchestration for training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from weiss_rl.eval.policies.set import heuristic_public_profile_name_for_policy_id
from weiss_rl.league.registry import REGISTRY_FILENAME, SnapshotRegistry
from weiss_rl.training.promotion import PROMOTION_GATE_RANDOMLEGAL_POLICY_ID

ValidatePeriodicDevEvalContractFn = Callable[[Any], Any]
ResolvePromotionAnchorPolicyIdsFn = Callable[..., tuple[dict[str, str], tuple[str, ...]]]
SpecDimensionsFn = Callable[[Any], tuple[int, int]]
SnapshotMetaByPolicyIdFn = Callable[[SnapshotRegistry], dict[str, Any]]
LoadSnapshotEvalModelFn = Callable[..., Any]
BuildHeuristicPublicPolicyFn = Callable[..., Any]
CloneCpuEvalModelFn = Callable[..., Any]
RunPromotionGateFn = Callable[..., Any]
PromotionGateBootstrapSeedFn = Callable[..., int]
SaveSnapshotRegistryWithRetentionFn = Callable[..., None]
PromotionGateRunnerCls = Callable[..., Any]


def run_snapshot_promotion_gate(
    *,
    stack: Any,
    contract: Any,
    artifacts: Any,
    training_paths: Any,
    learner: Any,
    candidate_policy_id: str,
    update_count: int,
    league_reference_update: int | None,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    validate_periodic_dev_eval_contract_fn: ValidatePeriodicDevEvalContractFn,
    resolve_promotion_anchor_policy_ids_fn: ResolvePromotionAnchorPolicyIdsFn,
    spec_dimensions_fn: SpecDimensionsFn,
    snapshot_meta_by_policy_id_fn: SnapshotMetaByPolicyIdFn,
    load_snapshot_eval_model_fn: LoadSnapshotEvalModelFn,
    build_heuristic_public_policy_fn: BuildHeuristicPublicPolicyFn,
    clone_cpu_eval_model_fn: CloneCpuEvalModelFn,
    promotion_gate_runner_cls: PromotionGateRunnerCls,
    run_promotion_gate_fn: RunPromotionGateFn,
    promotion_gate_bootstrap_seed_fn: PromotionGateBootstrapSeedFn,
    save_snapshot_registry_with_retention_fn: SaveSnapshotRegistryWithRetentionFn,
) -> bool | None:
    league = stack.config.league
    if league is None or not league.enabled or not league.promotion_gate_enabled:
        return None
    reference_update = int(update_count if league_reference_update is None else league_reference_update)
    if reference_update < int(league.warmup.first_updates):
        print(
            "Promotion gate skipped during league warmup: "
            f"update={update_count} effective_update={reference_update} threshold={int(league.warmup.first_updates)} "
            f"candidate={candidate_policy_id}"
        )
        return None
    if learner.model is None:
        raise RuntimeError("Promotion gate requires an attached learner model")

    evaluation = validate_periodic_dev_eval_contract_fn(stack)
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    anchor_policy_ids, missing_required = resolve_promotion_anchor_policy_ids_fn(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        print(
            "Promotion gate skipped: "
            f"update={update_count} candidate={candidate_policy_id} "
            f"missing_anchors={','.join(missing_required)}"
        )
        return None

    observation_dim, action_dim = spec_dimensions_fn(contract)
    snapshot_index = snapshot_meta_by_policy_id_fn(registry)
    anchor_models = {
        policy_id: load_snapshot_eval_model_fn(
            run_dir=artifacts.run_dir,
            snapshot_path=snapshot_index[policy_id].path,
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
            spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
        )
        for policy_id in set(anchor_policy_ids.values())
        if policy_id != PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
        and heuristic_public_profile_name_for_policy_id(policy_id) is None
    }
    heuristic_policies: dict[str, Any] = {}
    heuristic_policy_ids = {
        policy_id
        for policy_id in set(anchor_policy_ids.values())
        if heuristic_public_profile_name_for_policy_id(policy_id) is not None
    }
    if heuristic_policy_ids:
        try:
            heuristic_policies = {
                policy_id: build_heuristic_public_policy_fn(
                    contract.spec_bundle,
                    scoring_profile=cast(str, heuristic_public_profile_name_for_policy_id(policy_id)),
                )
                for policy_id in heuristic_policy_ids
            }
        except Exception as exc:
            assert league is not None
            missing_heuristic_required = [
                policy_id for policy_id in heuristic_policy_ids if policy_id in league.promotion_anchor_set_v1.required
            ]
            if missing_heuristic_required:
                missing_text = ", ".join(missing_heuristic_required)
                raise RuntimeError(
                    f"Promotion gate requires a heuristic-compatible simulator contract for {missing_text}"
                ) from exc
            anchor_policy_ids = {
                anchor_name: policy_id
                for anchor_name, policy_id in anchor_policy_ids.items()
                if heuristic_public_profile_name_for_policy_id(policy_id) is None
            }
            print(
                "Promotion gate note: skipping optional heuristic-public anchors because the active simulator contract "
                f"does not expose the required public action/observation metadata ({exc})."
            )
    runner = promotion_gate_runner_cls(
        stack=stack,
        focal_policy_id=candidate_policy_id,
        focal_model=clone_cpu_eval_model_fn(
            learner_model=cast(Any, learner.model),
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
            spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
        ),
        anchor_models=anchor_models,
        heuristic_policies=heuristic_policies,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        artifact_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
    )
    result = run_promotion_gate_fn(
        stack=stack,
        run_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        focal_policy_id=candidate_policy_id,
        anchor_policy_ids=anchor_policy_ids,
        runner=runner,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        bootstrap_seed=promotion_gate_bootstrap_seed_fn(
            update_count=update_count,
            policy_version=policy_version,
        ),
    )
    if result.passed:
        registry.add_champion(candidate_policy_id)
        save_snapshot_registry_with_retention_fn(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            registry=registry,
        )
        print(
            "Promotion gate passed: "
            f"update={update_count} candidate={candidate_policy_id} "
            f"anchors={','.join(result.ordered_opponents)}"
        )
        return True

    reason_codes = ",".join(str(reason.get("code", "unknown")) for reason in result.reasons) or "unknown"
    print(f"Promotion gate failed: update={update_count} candidate={candidate_policy_id} reasons={reason_codes}")
    return False
