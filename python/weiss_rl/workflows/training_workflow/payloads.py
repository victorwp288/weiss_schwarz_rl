"""Dry-run payload helpers for public training workflows."""

from __future__ import annotations

from weiss_rl.workflows.command_surface import (
    TRAIN_B1_COMMAND,
    TRAIN_MAIN_COMMAND,
    PublicWorkflowCommand,
)


def training_evidence_targets(command: PublicWorkflowCommand) -> list[str]:
    common_targets = [
        "manifest.json",
        "run_summary.json",
        "determinism_report.json",
        "training/logs/training_metrics.jsonl",
        "training/checkpoints/checkpoint_tracker.json",
        "training/snapshots/registry.json",
    ]
    if command.name == TRAIN_B1_COMMAND.name:
        return common_targets
    if command.name == TRAIN_MAIN_COMMAND.name:
        return [
            *common_targets,
            "training/logs/periodic_dev_eval_summaries.json",
            "eval/promotion_gate/",
        ]
    return []


__all__ = ["training_evidence_targets"]
