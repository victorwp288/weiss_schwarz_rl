"""Checkpoint IO wrapper installation for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from weiss_rl.training.train_entrypoint.checkpoint_lifecycle_hooks import (
    BuildCheckpointRecordRequest,
    PublishCheckpointAliasesRequest,
    WriteCheckpointRequest,
    build_checkpoint_record_with_entrypoint_hooks,
    publish_checkpoint_aliases_with_entrypoint_hooks,
    write_checkpoint_with_entrypoint_hooks,
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


__all__ = ["install_checkpoint_io_wrappers"]
