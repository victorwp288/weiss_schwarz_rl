"""Current-checkpoint callback assembly for the path-based training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnsureCurrentCheckpointRequest:
    training_paths: Any
    learner: Any
    stack: Any
    device: Any
    spec_hash256: str | None = None
    algorithm: str | None = None


def ensure_current_checkpoint_with_script_hooks(api: Any, request: EnsureCurrentCheckpointRequest) -> Any:
    training_paths = request.training_paths
    learner = request.learner
    stack = request.stack
    device = request.device
    return api.ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        write_checkpoint=lambda checkpoint_path: api._write_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            spec_hash256=request.spec_hash256,
            algorithm=request.algorithm,
        ),
    )
