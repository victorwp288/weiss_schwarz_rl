"""Promotion-gate callback assembly for the path-based training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SnapshotPromotionGateRequest:
    stack: Any
    contract: Any
    artifacts: Any
    training_paths: Any
    learner: Any
    candidate_policy_id: str
    update_count: int
    league_reference_update: int | None
    policy_version: int
    run_id256: str
    config_hash256: str
    spec_hash256: str


def run_snapshot_promotion_gate_with_script_hooks(api: Any, request: SnapshotPromotionGateRequest) -> Any:
    return api.run_snapshot_promotion_gate(
        stack=request.stack,
        contract=request.contract,
        artifacts=request.artifacts,
        training_paths=request.training_paths,
        learner=request.learner,
        candidate_policy_id=request.candidate_policy_id,
        update_count=request.update_count,
        league_reference_update=request.league_reference_update,
        policy_version=request.policy_version,
        run_id256=request.run_id256,
        config_hash256=request.config_hash256,
        spec_hash256=request.spec_hash256,
        validate_periodic_dev_eval_contract_fn=api._validate_periodic_dev_eval_contract,
        resolve_promotion_anchor_policy_ids_fn=api._resolve_promotion_anchor_policy_ids,
        spec_dimensions_fn=api._spec_dimensions,
        snapshot_meta_by_policy_id_fn=api._snapshot_meta_by_policy_id,
        load_snapshot_eval_model_fn=api._load_snapshot_eval_model,
        build_heuristic_public_policy_fn=api._build_heuristic_public_policy,
        clone_cpu_eval_model_fn=api._clone_cpu_eval_model,
        promotion_gate_runner_cls=api._PromotionGateRunner,
        run_promotion_gate_fn=api.run_promotion_gate,
        promotion_gate_bootstrap_seed_fn=api._promotion_gate_bootstrap_seed,
        save_snapshot_registry_with_retention_fn=api._save_snapshot_registry_with_retention,
    )
