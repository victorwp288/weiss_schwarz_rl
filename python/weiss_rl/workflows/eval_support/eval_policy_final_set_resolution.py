from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval import resolve_final_policy_set


def _default_dev_eval_summaries_path(layout: ArtifactLayout) -> Path | None:
    for candidate in (
        layout.training_logs_dir / "dev_eval_summaries.json",
        layout.training_logs_dir / "periodic_dev_eval_summaries.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _resolve_available_policy_source_paths(
    *,
    layout: ArtifactLayout,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    manifest_snapshot_registry: Path | None,
    manifest_dev_eval: Path | None,
) -> tuple[Path | None, Path | None]:
    resolved_snapshot_registry: Path | None = (
        snapshot_registry_path or manifest_snapshot_registry or (layout.training_snapshots_dir / "registry.json")
    )
    if resolved_snapshot_registry is None or not resolved_snapshot_registry.is_file():
        resolved_snapshot_registry = None
    resolved_dev_eval = dev_eval_summaries_path or manifest_dev_eval
    if resolved_dev_eval is None or not resolved_dev_eval.is_file():
        resolved_dev_eval = _default_dev_eval_summaries_path(layout)
    return resolved_snapshot_registry, resolved_dev_eval


def _resolve_deterministic_final_policy_set(
    *,
    evaluation: Any,
    resolved_snapshot_registry: Path | None,
    resolved_dev_eval: Path | None,
    resolve_final_policy_set_fn: Callable[..., list[str]] = resolve_final_policy_set,
) -> tuple[list[str], dict[str, Any]] | None:
    if resolved_snapshot_registry is None or resolved_dev_eval is None:
        return None
    resolved = resolve_final_policy_set_fn(
        snapshot_registry_path=resolved_snapshot_registry,
        dev_eval_summaries_path=resolved_dev_eval,
        config=evaluation.final_policy_set_selection,
        final_policy_set_size=evaluation.final_policy_set_size,
    )
    return (
        resolved,
        {
            "mode": "deterministic_v1",
            "policy_count": len(resolved),
            "snapshot_registry_path": resolved_snapshot_registry.as_posix(),
            "dev_eval_summaries_path": resolved_dev_eval.as_posix(),
            "final_policy_set_size": int(evaluation.final_policy_set_size),
        },
    )


def _raise_missing_final_policy_inputs(
    *,
    layout: ArtifactLayout,
    resolved_snapshot_registry: Path | None,
    resolved_dev_eval: Path | None,
    snapshot_registry_path: Path | None,
    manifest_snapshot_registry: Path | None,
    dev_eval_summaries_path: Path | None,
    manifest_dev_eval: Path | None,
) -> NoReturn:
    if resolved_snapshot_registry is None:
        raise FileNotFoundError(
            "final policy-set resolution requires a snapshot registry; checked "
            f"{snapshot_registry_path or manifest_snapshot_registry or (layout.training_snapshots_dir / 'registry.json')}"
        )
    if resolved_dev_eval is None:
        checked_paths = [
            path.as_posix()
            for path in (
                dev_eval_summaries_path,
                manifest_dev_eval,
                layout.training_logs_dir / "dev_eval_summaries.json",
                layout.training_logs_dir / "periodic_dev_eval_summaries.json",
            )
            if path is not None
        ]
        raise FileNotFoundError(
            "final policy-set resolution requires dev-eval summaries; checked "
            + (", ".join(checked_paths) if checked_paths else "<none>")
        )
    raise AssertionError("policy-set resolution should have returned or raised before reaching this point")


__all__ = [
    "_default_dev_eval_summaries_path",
    "_raise_missing_final_policy_inputs",
    "_resolve_available_policy_source_paths",
    "_resolve_deterministic_final_policy_set",
]
