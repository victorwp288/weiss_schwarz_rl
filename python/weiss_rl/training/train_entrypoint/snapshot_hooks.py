"""Snapshot import and evaluation hooks for the training entrypoint facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from weiss_rl.training.train_entrypoint.checkpoint_requests import (
    BuildHeuristicPublicPolicyRequest,
    EnsureNoLeagueBaselineAnchorRequest,
    ImportNoLeagueBaselineAnchorRequest,
    ImportSeedSnapshotPoolRequest,
    LoadSnapshotEvalModelRequest,
    SeedSnapshotPolicyIdRequest,
    ValidateSeedSnapshotImportContractRequest,
)


def build_heuristic_public_policy_with_entrypoint_hooks(
    api: Any,
    request: BuildHeuristicPublicPolicyRequest,
) -> Any:
    return api.build_heuristic_public_policy(
        request.spec_bundle,
        scoring_profile=request.scoring_profile,
        policy_cls=api.HeuristicPublicPolicy,
    )


def import_noleague_baseline_anchor_with_entrypoint_hooks(
    api: Any,
    request: ImportNoLeagueBaselineAnchorRequest,
) -> tuple[Path, str, int]:
    source_run_dir = Path(request.baseline_run_dir).resolve()
    source_snapshot = api._find_noleague_baseline_snapshot(source_run_dir)
    if source_snapshot is None:
        raise FileNotFoundError(
            "Could not resolve the canonical B1 no-league baseline snapshot in "
            f"{source_run_dir}. Run a dedicated baseline_noleague training job first."
        )

    source_weights_path = source_run_dir / source_snapshot.path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"Resolved B1 baseline snapshot is missing its weights artifact: {source_weights_path}")

    payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload must be a dict: {source_weights_path}")
    api._validate_imported_snapshot_contract(
        source_run_dir=source_run_dir,
        source_policy_id=source_snapshot.policy_id,
        payload=payload,
        expected_model_state_dict=request.expected_model_state_dict,
        expected_config_canonical=request.expected_config_canonical,
        expected_spec_hash256=request.expected_spec_hash256,
    )
    weights_path, weights_sha256 = api.write_imported_snapshot_artifact(
        snapshots_dir=request.training_paths.snapshots_dir,
        run_dir=request.run_dir,
        source_payload=payload,
        source_run_dir=source_run_dir,
        source_policy_id=source_snapshot.policy_id,
        source_snapshot_path=source_snapshot.path,
        target_policy_id=api._PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=int(source_snapshot.update),
        metadata_format="imported_train_snapshot_metadata_v1",
    )
    return weights_path, weights_sha256, int(source_snapshot.update)


def validate_seed_snapshot_import_contract_with_entrypoint_hooks(
    api: Any,
    request: ValidateSeedSnapshotImportContractRequest,
) -> None:
    api.validate_seed_snapshot_import_contract(
        source_run_dir=request.source_run_dir,
        payload=request.payload,
        expected_model_state_dict=request.expected_model_state_dict,
        expected_config_canonical=request.expected_config_canonical,
        expected_spec_hash256=request.expected_spec_hash256,
    )


def seed_snapshot_policy_id_with_entrypoint_hooks(api: Any, request: SeedSnapshotPolicyIdRequest) -> str:
    return api.seed_snapshot_policy_id(
        source_run_dir=request.source_run_dir,
        source_policy_id=request.source_policy_id,
    )


def import_seed_snapshot_pool_with_entrypoint_hooks(api: Any, request: ImportSeedSnapshotPoolRequest) -> list[str]:
    return api.import_seed_snapshot_pool(
        stack=request.stack,
        training_paths=request.training_paths,
        run_dir=request.run_dir,
        seed_snapshot_run_dir=request.seed_snapshot_run_dir,
        expected_model_state_dict=request.expected_model_state_dict,
        expected_config_canonical=request.expected_config_canonical,
        expected_spec_hash256=request.expected_spec_hash256,
    )


def ensure_noleague_baseline_anchor_with_entrypoint_hooks(
    api: Any,
    request: EnsureNoLeagueBaselineAnchorRequest,
) -> str | None:
    return api.ensure_noleague_baseline_anchor(
        stack=request.stack,
        training_paths=request.training_paths,
        run_dir=request.run_dir,
        learner=request.learner,
        device=request.device,
        config_hash256=request.config_hash256,
        spec_hash256=request.spec_hash256,
        baseline_run_dir=request.baseline_run_dir,
        permit_current_run_alias=request.permit_current_run_alias,
        source_checkpoint_path=request.source_checkpoint_path,
        update=request.update,
        write_checkpoint_fn=api._write_checkpoint,
        import_noleague_baseline_anchor_fn=api._import_noleague_baseline_anchor,
        model_guidance_payload_fn=api._model_guidance_payload,
        write_snapshot_artifact_fn=api._write_snapshot_artifact,
        experiment_role_fn=api._experiment_role,
    )


def load_snapshot_eval_model_with_entrypoint_hooks(api: Any, request: LoadSnapshotEvalModelRequest) -> Any:
    return api.load_snapshot_eval_model(
        run_dir=request.run_dir,
        snapshot_path=request.snapshot_path,
        observation_dim=request.observation_dim,
        action_dim=request.action_dim,
        stack=request.stack,
        observation_spec=request.observation_spec,
        spec_bundle=request.spec_bundle,
    )


__all__ = [
    "build_heuristic_public_policy_with_entrypoint_hooks",
    "ensure_noleague_baseline_anchor_with_entrypoint_hooks",
    "import_noleague_baseline_anchor_with_entrypoint_hooks",
    "import_seed_snapshot_pool_with_entrypoint_hooks",
    "load_snapshot_eval_model_with_entrypoint_hooks",
    "seed_snapshot_policy_id_with_entrypoint_hooks",
    "validate_seed_snapshot_import_contract_with_entrypoint_hooks",
]
