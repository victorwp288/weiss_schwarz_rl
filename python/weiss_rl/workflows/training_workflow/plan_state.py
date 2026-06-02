from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingWorkflowRequest:
    args: argparse.Namespace
    repo_root: Path
    python_exe: str

    @property
    def command(self) -> str:
        return str(self.args.command)

    @property
    def dry_run(self) -> bool:
        return bool(getattr(self.args, "dry_run", False))


@dataclass(frozen=True, slots=True)
class TrainingWorkflowPlan:
    plan_name: str
    command: list[str]
    payload: dict[str, Any]


def training_workflow_request(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    python_exe: str,
) -> TrainingWorkflowRequest:
    return TrainingWorkflowRequest(args=args, repo_root=repo_root, python_exe=python_exe)


__all__ = ["TrainingWorkflowPlan", "TrainingWorkflowRequest", "training_workflow_request"]
