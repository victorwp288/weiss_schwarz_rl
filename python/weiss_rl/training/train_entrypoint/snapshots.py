"""Snapshot and B1-anchor wrapper installation for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from weiss_rl.training.train_entrypoint.checkpoints import (
    BuildHeuristicPublicPolicyRequest,
    EnsureNoLeagueBaselineAnchorRequest,
    ImportNoLeagueBaselineAnchorRequest,
    ImportSeedSnapshotPoolRequest,
    LoadSnapshotEvalModelRequest,
    SeedSnapshotPolicyIdRequest,
    ValidateSeedSnapshotImportContractRequest,
    build_heuristic_public_policy_with_entrypoint_hooks,
    ensure_noleague_baseline_anchor_with_entrypoint_hooks,
    import_noleague_baseline_anchor_with_entrypoint_hooks,
    import_seed_snapshot_pool_with_entrypoint_hooks,
    load_snapshot_eval_model_with_entrypoint_hooks,
    seed_snapshot_policy_id_with_entrypoint_hooks,
    validate_seed_snapshot_import_contract_with_entrypoint_hooks,
)


def install_snapshot_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _build_heuristic_public_policy(spec_bundle: Mapping[str, object], *, scoring_profile: str) -> Any:
        return build_heuristic_public_policy_with_entrypoint_hooks(
            entrypoint_api(),
            BuildHeuristicPublicPolicyRequest(
                spec_bundle=spec_bundle,
                scoring_profile=scoring_profile,
            ),
        )

    def _import_noleague_baseline_anchor(
        *,
        training_paths: Any,
        run_dir: Path,
        baseline_run_dir: Path,
        expected_model_state_dict: dict[str, Any],
        expected_config_canonical: dict[str, Any] | None,
        expected_spec_hash256: str | None,
    ) -> tuple[Path, str, int]:
        return import_noleague_baseline_anchor_with_entrypoint_hooks(
            entrypoint_api(),
            ImportNoLeagueBaselineAnchorRequest(
                training_paths=training_paths,
                run_dir=run_dir,
                baseline_run_dir=baseline_run_dir,
                expected_model_state_dict=expected_model_state_dict,
                expected_config_canonical=expected_config_canonical,
                expected_spec_hash256=expected_spec_hash256,
            ),
        )

    def _validate_seed_snapshot_import_contract(
        *,
        source_run_dir: Path,
        payload: dict[str, Any],
        expected_model_state_dict: dict[str, Any],
        expected_config_canonical: dict[str, Any] | None,
        expected_spec_hash256: str | None,
    ) -> None:
        validate_seed_snapshot_import_contract_with_entrypoint_hooks(
            entrypoint_api(),
            ValidateSeedSnapshotImportContractRequest(
                source_run_dir=source_run_dir,
                payload=payload,
                expected_model_state_dict=expected_model_state_dict,
                expected_config_canonical=expected_config_canonical,
                expected_spec_hash256=expected_spec_hash256,
            ),
        )

    def _seed_snapshot_policy_id(*, source_run_dir: Path, source_policy_id: str) -> str:
        return seed_snapshot_policy_id_with_entrypoint_hooks(
            entrypoint_api(),
            SeedSnapshotPolicyIdRequest(source_run_dir=source_run_dir, source_policy_id=source_policy_id),
        )

    def _import_seed_snapshot_pool(
        *,
        stack: Any,
        training_paths: Any,
        run_dir: Path,
        seed_snapshot_run_dir: Path,
        expected_model_state_dict: dict[str, Any],
        expected_config_canonical: dict[str, Any] | None,
        expected_spec_hash256: str | None,
    ) -> list[str]:
        return import_seed_snapshot_pool_with_entrypoint_hooks(
            entrypoint_api(),
            ImportSeedSnapshotPoolRequest(
                stack=stack,
                training_paths=training_paths,
                run_dir=run_dir,
                seed_snapshot_run_dir=seed_snapshot_run_dir,
                expected_model_state_dict=expected_model_state_dict,
                expected_config_canonical=expected_config_canonical,
                expected_spec_hash256=expected_spec_hash256,
            ),
        )

    def _ensure_noleague_baseline_anchor(
        *,
        stack: Any,
        training_paths: Any,
        run_dir: Path,
        learner: Any,
        device: Any,
        config_hash256: str,
        spec_hash256: str | None = None,
        baseline_run_dir: Path | None = None,
        permit_current_run_alias: bool = False,
        source_checkpoint_path: Path | None = None,
        update: int | None = None,
    ) -> str | None:
        return ensure_noleague_baseline_anchor_with_entrypoint_hooks(
            entrypoint_api(),
            EnsureNoLeagueBaselineAnchorRequest(
                stack=stack,
                training_paths=training_paths,
                run_dir=run_dir,
                learner=learner,
                device=device,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                baseline_run_dir=baseline_run_dir,
                permit_current_run_alias=permit_current_run_alias,
                source_checkpoint_path=source_checkpoint_path,
                update=update,
            ),
        )

    def _load_snapshot_eval_model(
        *,
        run_dir: Path,
        snapshot_path: str,
        observation_dim: int,
        action_dim: int,
        stack: Any,
        observation_spec: dict[str, Any] | None = None,
        spec_bundle: dict[str, Any] | None = None,
    ) -> Any:
        return load_snapshot_eval_model_with_entrypoint_hooks(
            entrypoint_api(),
            LoadSnapshotEvalModelRequest(
                run_dir=run_dir,
                snapshot_path=snapshot_path,
                observation_dim=observation_dim,
                action_dim=action_dim,
                stack=stack,
                observation_spec=observation_spec,
                spec_bundle=spec_bundle,
            ),
        )

    namespace.update(
        {
            "_build_heuristic_public_policy": _build_heuristic_public_policy,
            "_import_noleague_baseline_anchor": _import_noleague_baseline_anchor,
            "_validate_seed_snapshot_import_contract": _validate_seed_snapshot_import_contract,
            "_seed_snapshot_policy_id": _seed_snapshot_policy_id,
            "_import_seed_snapshot_pool": _import_seed_snapshot_pool,
            "_ensure_noleague_baseline_anchor": _ensure_noleague_baseline_anchor,
            "_load_snapshot_eval_model": _load_snapshot_eval_model,
        }
    )


__all__ = ["install_snapshot_wrappers"]
