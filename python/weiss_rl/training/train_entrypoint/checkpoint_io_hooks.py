"""Checkpoint I/O hook wrappers for the training entrypoint facade."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from weiss_rl.training.train_entrypoint.checkpoint_requests import (
    BuildCheckpointRecordRequest,
    InitializeLearnerCheckpointRequest,
    PublishCheckpointAliasesRequest,
    RestoreLearnerCheckpointRequest,
    WriteCheckpointRequest,
)


def write_checkpoint_with_entrypoint_hooks(api: Any, request: WriteCheckpointRequest) -> dict[str, Any]:
    return api.write_minimal_train_checkpoint(
        checkpoint_path=request.checkpoint_path,
        learner=request.learner,
        device=request.device,
        config_hash256=api.compute_config_hash256(request.stack),
        spec_hash256=request.spec_hash256,
        algorithm=request.algorithm,
        recurrent_core=getattr(request.stack.config.model, "recurrent_core", None),
        guidance_payload=api._model_guidance_payload(request.learner.model),
    )


def build_checkpoint_record_with_entrypoint_hooks(api: Any, request: BuildCheckpointRecordRequest) -> dict[str, Any]:
    return api.build_checkpoint_record(
        alias_name=request.alias_name,
        alias_path=request.alias_path,
        source_checkpoint_path=request.source_checkpoint_path,
        run_dir=request.artifacts.run_dir,
        learner=request.learner,
        metric_kind=request.metric_kind,
        metric_value=request.metric_value,
    )


def publish_checkpoint_aliases_with_entrypoint_hooks(
    api: Any,
    request: PublishCheckpointAliasesRequest,
) -> dict[str, Any]:
    return api.publish_checkpoint_aliases(
        stack=request.stack,
        training_paths=request.training_paths,
        run_dir=request.artifacts.run_dir,
        checkpoint_path=request.checkpoint_path,
        learner=request.learner,
        latest_metrics=request.latest_metrics,
        dev_eval_summary=request.dev_eval_summary,
    )


def restore_learner_from_checkpoint_with_entrypoint_hooks(
    api: Any,
    request: RestoreLearnerCheckpointRequest,
) -> Any:
    allow_config_mismatch = os.environ.get("WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH", "").strip() == "1"
    return api.restore_minimal_train_checkpoint(
        checkpoint_path=request.checkpoint_path,
        learner=request.learner,
        device=request.device,
        expected_config_hash=api.compute_config_hash256(request.stack),
        expected_spec_hash256=request.expected_spec_hash256,
        algorithm=request.algorithm,
        restore_model_guidance=api.restore_model_guidance_from_payload,
        allow_config_mismatch=allow_config_mismatch,
        restore_counters=request.restore_counters,
    )


def initialize_learner_from_checkpoint_with_entrypoint_hooks(
    api: Any,
    request: InitializeLearnerCheckpointRequest,
) -> Any:
    return api.initialize_model_from_checkpoint(
        checkpoint_path=request.checkpoint_path,
        learner=request.learner,
        device=request.device,
        expected_spec_hash256=request.expected_spec_hash256,
        algorithm=request.algorithm,
        restore_model_guidance=api.restore_model_guidance_from_payload,
    )


def install_checkpoint_io_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _write_checkpoint(
        *,
        checkpoint_path: Path,
        learner: Any,
        stack: Any,
        device: Any,
        spec_hash256: str | None = None,
        algorithm: str | None = None,
    ) -> dict[str, Any]:
        return write_checkpoint_with_entrypoint_hooks(
            entrypoint_api(),
            WriteCheckpointRequest(
                checkpoint_path=checkpoint_path,
                learner=learner,
                stack=stack,
                device=device,
                spec_hash256=spec_hash256,
                algorithm=algorithm,
            ),
        )

    def _build_checkpoint_record(
        *,
        alias_name: str,
        alias_path: Path,
        source_checkpoint_path: Path,
        artifacts: Any,
        learner: Any,
        metric_kind: str | None = None,
        metric_value: float | None = None,
    ) -> dict[str, Any]:
        return build_checkpoint_record_with_entrypoint_hooks(
            entrypoint_api(),
            BuildCheckpointRecordRequest(
                alias_name=alias_name,
                alias_path=alias_path,
                source_checkpoint_path=source_checkpoint_path,
                artifacts=artifacts,
                learner=learner,
                metric_kind=metric_kind,
                metric_value=metric_value,
            ),
        )

    def _publish_checkpoint_aliases(
        *,
        stack: Any,
        training_paths: Any,
        artifacts: Any,
        checkpoint_path: Path,
        learner: Any,
        latest_metrics: Mapping[str, float] | None,
        dev_eval_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return publish_checkpoint_aliases_with_entrypoint_hooks(
            entrypoint_api(),
            PublishCheckpointAliasesRequest(
                stack=stack,
                training_paths=training_paths,
                artifacts=artifacts,
                checkpoint_path=checkpoint_path,
                learner=learner,
                latest_metrics=latest_metrics,
                dev_eval_summary=dev_eval_summary,
            ),
        )

    namespace.update(
        {
            "_write_checkpoint": _write_checkpoint,
            "_build_checkpoint_record": _build_checkpoint_record,
            "_publish_checkpoint_aliases": _publish_checkpoint_aliases,
        }
    )


__all__ = [
    "build_checkpoint_record_with_entrypoint_hooks",
    "initialize_learner_from_checkpoint_with_entrypoint_hooks",
    "install_checkpoint_io_wrappers",
    "publish_checkpoint_aliases_with_entrypoint_hooks",
    "restore_learner_from_checkpoint_with_entrypoint_hooks",
    "write_checkpoint_with_entrypoint_hooks",
]
