"""Dry-run payload helpers for public evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.workflows.command_surface import (
    B2_AUDIT_COMMAND,
    EVAL_FINAL_COMMAND,
    FIGURES_COMMAND,
    SMOKE_EVAL_COMMAND,
    PublicWorkflowCommand,
    public_workflow_command_payload,
)
from weiss_rl.workflows.evaluation_workflow.stages import evaluation_workflow_stage_payload


def evaluation_evidence_targets(command: PublicWorkflowCommand) -> list[str]:
    if command.name == SMOKE_EVAL_COMMAND.name:
        return [
            "eval/final_eval/summary.json",
            "eval/final_eval/episodes.jsonl",
        ]
    if command.name == EVAL_FINAL_COMMAND.name:
        return [
            "eval/final_eval/summary.json",
            "eval/final_eval/matchups.csv",
            "eval/final_eval/matrices/mean.csv",
            "paper_readiness_summary.json",
        ]
    if command.name == FIGURES_COMMAND.name:
        return [
            "figures/",
            "eval/final_eval/summary.json",
        ]
    if command.name == B2_AUDIT_COMMAND.name:
        return [
            "diagnostics/b2_disagreement_summary.json",
            "diagnostics/b2_disagreement_top_states.json",
        ]
    return []


def evaluation_workflow_payload(
    *,
    command: PublicWorkflowCommand,
    run_dir: Path,
    **extra: Any,
) -> dict[str, Any]:
    """Return the reader-facing dry-run payload for an evaluation workflow."""

    payload: dict[str, Any] = public_workflow_command_payload(command)
    payload["run_dir"] = run_dir.as_posix()
    payload["evidence_targets"] = evaluation_evidence_targets(command)
    payload["workflow_stages"] = evaluation_workflow_stage_payload()
    payload.update(extra)
    return payload


__all__ = ["evaluation_evidence_targets", "evaluation_workflow_payload", "evaluation_workflow_stage_payload"]
