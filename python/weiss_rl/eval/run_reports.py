"""Run-level report helpers for canonical eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout

GIT_COMMIT_HEX_LENGTH = 40


def normalize_git_commit(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != GIT_COMMIT_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def effective_manifest_git_commit(
    *,
    manifest: dict[str, Any],
    git_commit_override: str,
) -> str:
    current = normalize_git_commit(str(manifest.get("git_commit", "")))
    if current:
        return current
    return normalize_git_commit(git_commit_override)


def persist_policy_selection_in_manifest(
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
    write_json(layout.manifest_path, manifest)


def update_run_level_reports(
    *,
    layout: ArtifactLayout,
    run_dir: Path,
    policy_ids: list[str],
    selection_details: dict[str, Any],
    final_eval_payload: dict[str, Any],
    metagame_payload: dict[str, Any] | None,
    figure_paths: tuple[Path, ...],
    readiness_payload: dict[str, Any] | None,
) -> None:
    run_summary = load_json_object(layout.run_summary_path, label="run summary")
    run_summary.update(
        {
            "final_eval_dir": layout.relative(layout.final_eval_dir),
            "policy_ids": list(policy_ids),
            "policy_set_selection_mode": selection_details.get("mode", "unknown"),
            "metagame_dir": None if metagame_payload is None else layout.relative(layout.metagame_dir),
            "figure_outputs": [layout.relative(path) for path in figure_paths],
            "paper_readiness_summary_path": layout.relative(layout.paper_readiness_summary_path),
            "paper_grade": bool(readiness_payload and readiness_payload.get("passed", False)),
            "canonical_eval_completed": True,
        }
    )
    write_json(layout.run_summary_path, run_summary)

    determinism_report = load_json_object(layout.determinism_report_path, label="determinism report")
    replay_verification = load_json_object(layout.replay_verification_json(), label="replay verification summary")
    artifact_hashes = load_json_object(layout.final_eval_aggregate_hashes_json(), label="final eval artifact hashes")
    determinism_report.update(
        {
            "run_dir": run_dir.as_posix(),
            "policy_selection_mode": selection_details.get("mode", "unknown"),
            "replay_verification": {
                "path": layout.relative(layout.replay_verification_json()),
                "status": replay_verification.get("status", "unknown"),
                "sampled_episode_count": replay_verification.get("sampled_episode_count", 0),
                "verified_episode_count": replay_verification.get("verified_episode_count", 0),
                "failed_episode_count": replay_verification.get("failed_episode_count", 0),
            },
            "canonical_artifact_hashes": dict(cast(dict[str, Any], artifact_hashes.get("artifacts", {}))),
            "final_eval": {
                "path": layout.relative(layout.final_eval_summary_json()),
                "policy_ids": list(policy_ids),
                "selection": dict(selection_details),
                "matchup_count": len(cast(list[Any], final_eval_payload.get("matchups", []))),
            },
        }
    )
    write_json(layout.determinism_report_path, determinism_report)


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
