"""Compatibility facade for checkpoint IO hook requests."""

from __future__ import annotations

from weiss_rl.training.train_entrypoint_checkpoint_lifecycle_hooks import (
    BuildCheckpointRecordRequest,
    PublishCheckpointAliasesRequest,
    WriteCheckpointRequest,
    build_checkpoint_record_with_entrypoint_hooks,
    publish_checkpoint_aliases_with_entrypoint_hooks,
    write_checkpoint_with_entrypoint_hooks,
)

__all__ = [
    "BuildCheckpointRecordRequest",
    "PublishCheckpointAliasesRequest",
    "WriteCheckpointRequest",
    "build_checkpoint_record_with_entrypoint_hooks",
    "publish_checkpoint_aliases_with_entrypoint_hooks",
    "write_checkpoint_with_entrypoint_hooks",
]
