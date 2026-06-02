from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.workflows.eval_support.eval_report_io import _load_json_object, _normalize_git_commit, _write_json


def _effective_manifest_git_commit(
    *,
    manifest: dict[str, Any],
    git_commit_override: str,
) -> str:
    current = _normalize_git_commit(str(manifest.get("git_commit", "")))
    if current:
        return current
    return _normalize_git_commit(git_commit_override)


def _persist_policy_selection_in_manifest(
    *,
    layout: ArtifactLayout,
    manifest: dict[str, Any],
    policy_ids: list[str],
    selection_details: dict[str, Any],
) -> None:
    manifest["policy_set_selection"] = list(policy_ids)
    merged_details = dict(selection_details)
    merged_details.setdefault("status", "resolved")
    merged_details["resolved_by"] = "canonical_eval_pipeline_v1"
    merged_details["policy_count"] = len(policy_ids)
    manifest["policy_set_selection_details"] = merged_details
    _write_json(layout.manifest_path, manifest)


def _policy_selection_mode(selection_details: dict[str, Any]) -> str:
    return str(selection_details.get("mode", "")).strip().lower()


def _resolve_selection_inputs_from_manifest(
    *,
    stack_root: Path,
    manifest: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    details = manifest.get("policy_set_selection_details")
    if not isinstance(details, dict):
        return None, None
    source_paths = details.get("source_paths")
    if not isinstance(source_paths, dict):
        return None, None

    def _resolve(path_value: Any) -> Path | None:
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = stack_root / candidate
        return candidate

    return _resolve(source_paths.get("snapshot_registry_json")), _resolve(source_paths.get("dev_eval_summaries_json"))


def _run_summary_marks_canonical_eval_completed(layout: ArtifactLayout) -> bool:
    if not layout.run_summary_path.is_file():
        return False
    try:
        run_summary = _load_json_object(layout.run_summary_path, label="run summary")
    except Exception:
        return False
    return bool(run_summary.get("canonical_eval_completed", False))


def _authoritative_manifest_policy_selection(
    *,
    manifest: dict[str, Any],
    layout: ArtifactLayout,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
) -> tuple[list[str], dict[str, Any]] | None:
    if snapshot_registry_path is not None or dev_eval_summaries_path is not None:
        return None

    manifest_policy_ids = manifest.get("policy_set_selection")
    if not isinstance(manifest_policy_ids, list):
        return None
    resolved_from_manifest = [str(policy_id).strip() for policy_id in manifest_policy_ids if str(policy_id).strip()]
    if not resolved_from_manifest:
        return None

    details = manifest.get("policy_set_selection_details")
    status = ""
    selection_details: dict[str, Any] = {}
    if isinstance(details, dict):
        selection_details = dict(details)
        status = str(details.get("status", "")).strip().lower()
    if status == "unresolved":
        return None
    if _policy_selection_mode(selection_details) == "explicit_cli":
        return None

    has_completed_eval_artifacts = bool(
        layout.final_eval_summary_json().is_file() or _run_summary_marks_canonical_eval_completed(layout)
    )
    if not has_completed_eval_artifacts:
        return None

    selection_details.setdefault("mode", "manifest_policy_set_selection")
    selection_details.setdefault("status", "resolved")
    selection_details["policy_count"] = len(resolved_from_manifest)
    return resolved_from_manifest, selection_details


__all__ = [
    "_authoritative_manifest_policy_selection",
    "_effective_manifest_git_commit",
    "_persist_policy_selection_in_manifest",
    "_policy_selection_mode",
    "_resolve_selection_inputs_from_manifest",
    "_run_summary_marks_canonical_eval_completed",
]
