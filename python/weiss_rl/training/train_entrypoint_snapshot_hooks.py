"""Compatibility facade for snapshot and B1-anchor hook requests."""

from __future__ import annotations

from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_hooks import (
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

__all__ = [
    "BuildHeuristicPublicPolicyRequest",
    "EnsureNoLeagueBaselineAnchorRequest",
    "ImportNoLeagueBaselineAnchorRequest",
    "ImportSeedSnapshotPoolRequest",
    "LoadSnapshotEvalModelRequest",
    "SeedSnapshotPolicyIdRequest",
    "ValidateSeedSnapshotImportContractRequest",
    "build_heuristic_public_policy_with_entrypoint_hooks",
    "ensure_noleague_baseline_anchor_with_entrypoint_hooks",
    "import_noleague_baseline_anchor_with_entrypoint_hooks",
    "import_seed_snapshot_pool_with_entrypoint_hooks",
    "load_snapshot_eval_model_with_entrypoint_hooks",
    "seed_snapshot_policy_id_with_entrypoint_hooks",
    "validate_seed_snapshot_import_contract_with_entrypoint_hooks",
]
