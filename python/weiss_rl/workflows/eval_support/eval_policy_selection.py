from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import (
    _default_dev_eval_summaries_path,
    _raise_missing_final_policy_inputs,
    _resolve_available_policy_source_paths,
    _resolve_deterministic_final_policy_set,
)
from weiss_rl.workflows.eval_support.eval_policy_manifest_selection import (
    _authoritative_manifest_policy_selection,
    _effective_manifest_git_commit,
    _persist_policy_selection_in_manifest,
    _policy_selection_mode,
    _resolve_selection_inputs_from_manifest,
    _run_summary_marks_canonical_eval_completed,
)
from weiss_rl.workflows.eval_support.eval_policy_selection_results import (
    _explicit_policy_selection,
    _manifest_policy_selection_fallback,
)


def _resolve_policy_ids_for_run(
    *,
    policy_ids: list[str],
    stack: Any,
    manifest: dict[str, Any],
    layout: ArtifactLayout,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
) -> tuple[list[str], dict[str, Any], Path | None, Path | None]:
    manifest_snapshot_registry, manifest_dev_eval = _resolve_selection_inputs_from_manifest(
        stack_root=stack.root,
        manifest=manifest,
    )
    resolved_snapshot_registry, resolved_dev_eval = _resolve_available_policy_source_paths(
        layout=layout,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        manifest_snapshot_registry=manifest_snapshot_registry,
        manifest_dev_eval=manifest_dev_eval,
    )

    explicit_selection = _explicit_policy_selection(policy_ids)
    if explicit_selection is not None:
        resolved, selection_details = explicit_selection
        return resolved, selection_details, resolved_snapshot_registry, resolved_dev_eval

    authoritative_manifest_selection = _authoritative_manifest_policy_selection(
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
    )
    if authoritative_manifest_selection is not None:
        resolved_from_manifest, selection_details = authoritative_manifest_selection
        return resolved_from_manifest, selection_details, resolved_snapshot_registry, resolved_dev_eval

    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    deterministic_selection = _resolve_deterministic_final_policy_set(
        evaluation=evaluation,
        resolved_snapshot_registry=resolved_snapshot_registry,
        resolved_dev_eval=resolved_dev_eval,
    )
    if deterministic_selection is not None:
        resolved, selection_details = deterministic_selection
        return resolved, selection_details, resolved_snapshot_registry, resolved_dev_eval

    fallback_selection = _manifest_policy_selection_fallback(manifest)
    if fallback_selection is not None:
        resolved, selection_details = fallback_selection
        return resolved, selection_details, resolved_snapshot_registry, resolved_dev_eval

    _raise_missing_final_policy_inputs(
        layout=layout,
        resolved_snapshot_registry=resolved_snapshot_registry,
        resolved_dev_eval=resolved_dev_eval,
        snapshot_registry_path=snapshot_registry_path,
        manifest_snapshot_registry=manifest_snapshot_registry,
        dev_eval_summaries_path=dev_eval_summaries_path,
        manifest_dev_eval=manifest_dev_eval,
    )


__all__ = [
    "_authoritative_manifest_policy_selection",
    "_default_dev_eval_summaries_path",
    "_effective_manifest_git_commit",
    "_explicit_policy_selection",
    "_manifest_policy_selection_fallback",
    "_persist_policy_selection_in_manifest",
    "_policy_selection_mode",
    "_resolve_policy_ids_for_run",
    "_resolve_selection_inputs_from_manifest",
    "_run_summary_marks_canonical_eval_completed",
]
