"""Policy-set resolution helpers for canonical eval runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval.final_eval import resolve_final_policy_set


def resolve_policy_ids_for_run(
    *,
    policy_ids: list[str],
    stack: Any,
    manifest: dict[str, Any],
    layout: ArtifactLayout,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    allow_completed_manifest_policy_selection: bool = True,
) -> tuple[list[str], dict[str, Any], Path | None, Path | None]:
    manifest_snapshot_registry, manifest_dev_eval = _resolve_selection_inputs_from_manifest(
        stack_root=stack.root,
        manifest=manifest,
    )
    resolved_snapshot_registry: Path | None = (
        snapshot_registry_path or manifest_snapshot_registry or (layout.training_snapshots_dir / "registry.json")
    )
    if snapshot_registry_path is not None and not snapshot_registry_path.is_file():
        raise FileNotFoundError(f"Explicit snapshot registry path does not exist: {snapshot_registry_path}")
    if manifest_snapshot_registry is not None and not manifest_snapshot_registry.is_file():
        raise FileNotFoundError(f"Manifest snapshot registry path does not exist: {manifest_snapshot_registry}")
    if resolved_snapshot_registry is None or not resolved_snapshot_registry.is_file():
        resolved_snapshot_registry = None
    resolved_dev_eval = dev_eval_summaries_path or manifest_dev_eval
    if dev_eval_summaries_path is not None and not dev_eval_summaries_path.is_file():
        raise FileNotFoundError(f"Explicit dev eval summaries path does not exist: {dev_eval_summaries_path}")
    if manifest_dev_eval is not None and not manifest_dev_eval.is_file():
        raise FileNotFoundError(f"Manifest dev eval summaries path does not exist: {manifest_dev_eval}")
    if resolved_dev_eval is None or not resolved_dev_eval.is_file():
        resolved_dev_eval = _default_dev_eval_summaries_path(layout)

    explicit_policy_ids = [policy_id.strip() for policy_id in policy_ids if policy_id.strip()]
    if explicit_policy_ids:
        return (
            explicit_policy_ids,
            {"mode": "explicit_cli", "policy_count": len(explicit_policy_ids)},
            resolved_snapshot_registry,
            resolved_dev_eval,
        )

    authoritative_manifest_selection = _authoritative_manifest_policy_selection(
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        allow_completed_manifest_policy_selection=allow_completed_manifest_policy_selection,
    )
    if authoritative_manifest_selection is not None:
        resolved_from_manifest, selection_details = authoritative_manifest_selection
        return resolved_from_manifest, selection_details, resolved_snapshot_registry, resolved_dev_eval

    manifest_policy_ids = manifest.get("policy_set_selection")
    resolved_from_manifest = (
        [str(policy_id).strip() for policy_id in manifest_policy_ids if str(policy_id).strip()]
        if isinstance(manifest_policy_ids, list)
        else []
    )

    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    if resolved_snapshot_registry is not None and resolved_dev_eval is not None:
        resolved = resolve_final_policy_set(
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
            resolved_snapshot_registry,
            resolved_dev_eval,
        )

    if resolved_from_manifest and allow_completed_manifest_policy_selection:
        return (
            resolved_from_manifest,
            {
                "mode": "manifest_policy_set_selection_fallback",
                "policy_count": len(resolved_from_manifest),
            },
            resolved_snapshot_registry,
            resolved_dev_eval,
        )

    if resolved_from_manifest and not allow_completed_manifest_policy_selection:
        raise FileNotFoundError(
            "current final policy-set inputs could not be resolved from run artifacts, and completed manifest reuse "
            "is disabled. Provide current snapshot/dev-eval inputs or pass --reuse-manifest-policy-selection."
        )

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


def _default_dev_eval_summaries_path(layout: ArtifactLayout) -> Path | None:
    for candidate in (
        layout.training_logs_dir / "periodic_dev_eval_summaries.json",
        layout.training_logs_dir / "dev_eval_summaries.json",
    ):
        if candidate.is_file():
            return candidate
    return None


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
    allow_completed_manifest_policy_selection: bool,
) -> tuple[list[str], dict[str, Any]] | None:
    if not allow_completed_manifest_policy_selection:
        return None
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

    if not _run_summary_marks_canonical_eval_completed(layout):
        return None

    selection_details.setdefault("mode", "manifest_policy_set_selection")
    selection_details.setdefault("status", "resolved")
    selection_details["policy_count"] = len(resolved_from_manifest)
    return resolved_from_manifest, selection_details


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload
