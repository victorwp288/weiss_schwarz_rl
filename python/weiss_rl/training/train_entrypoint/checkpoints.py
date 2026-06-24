"""Checkpoint and learner hook glue for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from weiss_rl.training.train_entrypoint.checkpoint_io_hooks import (
    build_checkpoint_record_with_entrypoint_hooks,
    initialize_learner_from_checkpoint_with_entrypoint_hooks,
    install_checkpoint_io_wrappers,
    publish_checkpoint_aliases_with_entrypoint_hooks,
    restore_learner_from_checkpoint_with_entrypoint_hooks,
    write_checkpoint_with_entrypoint_hooks,
)
from weiss_rl.training.train_entrypoint.checkpoint_requests import (
    BuildCheckpointRecordRequest,
    BuildHeuristicPublicPolicyRequest,
    BuildTrainingLearnerRequest,
    EnsureNoLeagueBaselineAnchorRequest,
    ImportNoLeagueBaselineAnchorRequest,
    ImportSeedSnapshotPoolRequest,
    InitializeLearnerCheckpointRequest,
    LoadSnapshotEvalModelRequest,
    PublishCheckpointAliasesRequest,
    RestoreLearnerCheckpointRequest,
    SeedSnapshotPolicyIdRequest,
    StructuredWarmstartRequest,
    ValidateSeedSnapshotImportContractRequest,
    WriteCheckpointRequest,
)
from weiss_rl.training.train_entrypoint.learner_hooks import (
    build_training_learner_with_entrypoint_hooks,
    install_learner_wrappers,
    run_structured_warmstart_with_entrypoint_hooks,
)
from weiss_rl.training.train_entrypoint.snapshot_hooks import (
    build_heuristic_public_policy_with_entrypoint_hooks,
    ensure_noleague_baseline_anchor_with_entrypoint_hooks,
    import_noleague_baseline_anchor_with_entrypoint_hooks,
    import_seed_snapshot_pool_with_entrypoint_hooks,
    load_snapshot_eval_model_with_entrypoint_hooks,
    seed_snapshot_policy_id_with_entrypoint_hooks,
    validate_seed_snapshot_import_contract_with_entrypoint_hooks,
)


def install_checkpoint_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    from weiss_rl.training.train_entrypoint.snapshots import install_snapshot_wrappers

    install_checkpoint_io_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_learner_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_snapshot_wrappers(namespace, entrypoint_api=entrypoint_api)


__all__ = [
    "BuildCheckpointRecordRequest",
    "BuildHeuristicPublicPolicyRequest",
    "BuildTrainingLearnerRequest",
    "EnsureNoLeagueBaselineAnchorRequest",
    "ImportNoLeagueBaselineAnchorRequest",
    "ImportSeedSnapshotPoolRequest",
    "InitializeLearnerCheckpointRequest",
    "LoadSnapshotEvalModelRequest",
    "PublishCheckpointAliasesRequest",
    "RestoreLearnerCheckpointRequest",
    "SeedSnapshotPolicyIdRequest",
    "StructuredWarmstartRequest",
    "ValidateSeedSnapshotImportContractRequest",
    "WriteCheckpointRequest",
    "build_checkpoint_record_with_entrypoint_hooks",
    "build_heuristic_public_policy_with_entrypoint_hooks",
    "build_training_learner_with_entrypoint_hooks",
    "ensure_noleague_baseline_anchor_with_entrypoint_hooks",
    "import_noleague_baseline_anchor_with_entrypoint_hooks",
    "import_seed_snapshot_pool_with_entrypoint_hooks",
    "install_checkpoint_io_wrappers",
    "install_checkpoint_wrappers",
    "install_learner_wrappers",
    "initialize_learner_from_checkpoint_with_entrypoint_hooks",
    "load_snapshot_eval_model_with_entrypoint_hooks",
    "publish_checkpoint_aliases_with_entrypoint_hooks",
    "restore_learner_from_checkpoint_with_entrypoint_hooks",
    "run_structured_warmstart_with_entrypoint_hooks",
    "seed_snapshot_policy_id_with_entrypoint_hooks",
    "validate_seed_snapshot_import_contract_with_entrypoint_hooks",
    "write_checkpoint_with_entrypoint_hooks",
]
