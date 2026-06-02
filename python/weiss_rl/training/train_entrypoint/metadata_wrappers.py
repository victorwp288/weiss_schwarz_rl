"""Metadata wrapper installation for the training entrypoint facade."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from weiss_rl.training.train_entrypoint.metadata_hooks import (
    PersistSnapshotRegistryEntryRequest,
    experiment_role_with_entrypoint_hooks,
    git_commit_with_entrypoint_hooks,
    git_dirty_with_entrypoint_hooks,
    git_output_with_entrypoint_hooks,
    persist_snapshot_registry_entry_with_entrypoint_hooks,
    repo_root_with_entrypoint_hooks,
)


def install_metadata_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _persist_snapshot_registry_entry(
        *,
        stack: Any,
        training_paths: Any,
        run_dir: Path,
        checkpoint_path: Path,
        model_state_dict: dict[str, Any],
        config_hash256: str,
        device: Any,
        update: int,
        policy_version: int,
        model: Any | None = None,
    ) -> str:
        return persist_snapshot_registry_entry_with_entrypoint_hooks(
            entrypoint_api(),
            PersistSnapshotRegistryEntryRequest(
                stack=stack,
                training_paths=training_paths,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                model_state_dict=model_state_dict,
                config_hash256=config_hash256,
                device=device,
                update=update,
                policy_version=policy_version,
                model=model,
            ),
        )

    def _repo_root() -> Path:
        return repo_root_with_entrypoint_hooks(entrypoint_api(), source_file=Path(namespace["__file__"]))

    def _git_output(args: list[str]) -> str:
        return git_output_with_entrypoint_hooks(entrypoint_api(), args=args, source_file=Path(namespace["__file__"]))

    def _git_commit() -> str:
        return git_commit_with_entrypoint_hooks(entrypoint_api(), source_file=Path(namespace["__file__"]))

    def _git_dirty() -> bool:
        return git_dirty_with_entrypoint_hooks(entrypoint_api(), source_file=Path(namespace["__file__"]))

    def _experiment_role(stack: Any) -> str:
        return experiment_role_with_entrypoint_hooks(stack)

    namespace.update(
        {
            "_persist_snapshot_registry_entry": _persist_snapshot_registry_entry,
            "_repo_root": _repo_root,
            "_git_output": _git_output,
            "_git_commit": _git_commit,
            "_git_dirty": _git_dirty,
            "_experiment_role": _experiment_role,
        }
    )
