from __future__ import annotations

from pathlib import Path

from weiss_rl.workflows.controller_plan import ControllerWorkflowPlan
from weiss_rl.workflows.planning import _run_or_plan


def _run_controller_workflow_plan(*, plan: ControllerWorkflowPlan, repo_root: Path, dry_run: bool) -> None:
    _run_or_plan(
        repo_root=repo_root,
        plan_name=plan.plan_name,
        command=plan.command,
        dry_run=dry_run,
        payload=plan.payload,
    )
