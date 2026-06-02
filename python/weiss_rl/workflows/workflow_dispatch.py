from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from weiss_rl.workflows.evaluation_workflow.dispatch import dispatch_evaluation_request
from weiss_rl.workflows.evaluation_workflow.plan_state import EvaluationWorkflowRequest, evaluation_workflow_request
from weiss_rl.workflows.training_workflow.dispatch import dispatch_training_request
from weiss_rl.workflows.training_workflow.plan_state import TrainingWorkflowRequest, training_workflow_request


@dataclass(frozen=True, slots=True)
class WorkflowDispatchRequest:
    args: argparse.Namespace
    repo_root: Path
    python_exe: str

    @property
    def command(self) -> str:
        return str(self.args.command)

    @property
    def dry_run(self) -> bool:
        return bool(getattr(self.args, "dry_run", False))

    def training_request(self) -> TrainingWorkflowRequest:
        return training_workflow_request(args=self.args, repo_root=self.repo_root, python_exe=self.python_exe)

    def evaluation_request(self) -> EvaluationWorkflowRequest:
        return evaluation_workflow_request(args=self.args, repo_root=self.repo_root, python_exe=self.python_exe)


def workflow_dispatch_request(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    python_exe: str,
) -> WorkflowDispatchRequest:
    return WorkflowDispatchRequest(args=args, repo_root=repo_root, python_exe=python_exe)


def dispatch_workflow_request(request: WorkflowDispatchRequest) -> bool:
    if dispatch_training_request(request.training_request()):
        return True
    if dispatch_evaluation_request(request.evaluation_request()):
        return True
    return False


def dispatch_workflow_command(*, args: argparse.Namespace, repo_root: Path, python_exe: str) -> bool:
    return dispatch_workflow_request(workflow_dispatch_request(args=args, repo_root=repo_root, python_exe=python_exe))


__all__ = [
    "WorkflowDispatchRequest",
    "dispatch_workflow_command",
    "dispatch_workflow_request",
    "workflow_dispatch_request",
]
