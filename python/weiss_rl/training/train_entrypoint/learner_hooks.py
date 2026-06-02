"""Compatibility facade for learner lifecycle hook requests."""

from __future__ import annotations

from weiss_rl.training.train_entrypoint.checkpoint_lifecycle_hooks import (
    BuildTrainingLearnerRequest,
    InitializeLearnerCheckpointRequest,
    RestoreLearnerCheckpointRequest,
    StructuredWarmstartRequest,
    build_training_learner_with_entrypoint_hooks,
    initialize_learner_from_checkpoint_with_entrypoint_hooks,
    restore_learner_from_checkpoint_with_entrypoint_hooks,
    run_structured_warmstart_with_entrypoint_hooks,
)

__all__ = [
    "BuildTrainingLearnerRequest",
    "InitializeLearnerCheckpointRequest",
    "RestoreLearnerCheckpointRequest",
    "StructuredWarmstartRequest",
    "build_training_learner_with_entrypoint_hooks",
    "initialize_learner_from_checkpoint_with_entrypoint_hooks",
    "restore_learner_from_checkpoint_with_entrypoint_hooks",
    "run_structured_warmstart_with_entrypoint_hooks",
]
