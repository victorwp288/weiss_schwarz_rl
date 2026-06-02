from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout


@dataclass(frozen=True, slots=True)
class RunLevelReportUpdateInputs:
    layout: ArtifactLayout
    run_dir: Path
    policy_ids: list[str]
    selection_details: dict[str, Any]
    final_eval_payload: dict[str, Any]
    metagame_payload: dict[str, Any] | None
    figure_paths: tuple[Path, ...]
    readiness_payload: dict[str, Any] | None


def build_run_summary_update_fields(inputs: RunLevelReportUpdateInputs) -> dict[str, Any]:
    layout = inputs.layout
    return {
        "final_eval_dir": layout.relative(layout.final_eval_dir),
        "policy_ids": list(inputs.policy_ids),
        "policy_set_selection_mode": inputs.selection_details.get("mode", "unknown"),
        "metagame_dir": None if inputs.metagame_payload is None else layout.relative(layout.metagame_dir),
        "figure_outputs": [layout.relative(path) for path in inputs.figure_paths],
        "paper_readiness_summary_path": layout.relative(layout.paper_readiness_summary_path),
        "paper_grade": bool(inputs.readiness_payload and inputs.readiness_payload.get("passed", False)),
        "canonical_eval_completed": True,
    }


def build_determinism_report_update_fields(
    inputs: RunLevelReportUpdateInputs,
    *,
    replay_verification: dict[str, Any],
    artifact_hashes: dict[str, Any],
) -> dict[str, Any]:
    layout = inputs.layout
    return {
        "run_dir": inputs.run_dir.as_posix(),
        "policy_selection_mode": inputs.selection_details.get("mode", "unknown"),
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
            "policy_ids": list(inputs.policy_ids),
            "selection": dict(inputs.selection_details),
            "matchup_count": len(cast(list[Any], inputs.final_eval_payload.get("matchups", []))),
        },
    }


__all__ = [
    "RunLevelReportUpdateInputs",
    "build_determinism_report_update_fields",
    "build_run_summary_update_fields",
]
