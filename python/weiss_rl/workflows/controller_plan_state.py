from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ControllerWorkflowRequest:
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
class ControllerWorkflowPlan:
    plan_name: str
    command: list[str]
    payload: dict[str, Any]


def controller_workflow_request(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    python_exe: str,
) -> ControllerWorkflowRequest:
    return ControllerWorkflowRequest(args=args, repo_root=repo_root, python_exe=python_exe)


__all__ = ["ControllerWorkflowPlan", "ControllerWorkflowRequest", "controller_workflow_request"]
