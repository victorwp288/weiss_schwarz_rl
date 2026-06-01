"""Dispatch parsed package-CLI workflow commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Protocol

from weiss_rl.workflows.dispatch_bootstrap import (
    dispatch_guarded_league_bootstrap,
    dispatch_guided_bootstrap_loop,
)
from weiss_rl.workflows.dispatch_evaluation import (
    dispatch_b2_audit,
    dispatch_eval,
    dispatch_figures,
    dispatch_guard_run,
)
from weiss_rl.workflows.dispatch_training import (
    dispatch_train_b1,
    dispatch_train_b1_guided_seed,
    dispatch_train_main,
    dispatch_train_main_guided_bootstrap,
)
from weiss_rl.workflows.profiles import resolve_repo_root


class WorkflowHandler(Protocol):
    def __call__(self, args: argparse.Namespace, *, repo_root: Path, python_exe: str) -> None: ...


_WORKFLOW_HANDLERS: dict[str, WorkflowHandler] = {
    "train-b1": dispatch_train_b1,
    "train-b1-guided-seed": dispatch_train_b1_guided_seed,
    "train-main": dispatch_train_main,
    "train-main-guided-bootstrap": dispatch_train_main_guided_bootstrap,
    "smoke-eval": dispatch_eval,
    "eval-final": dispatch_eval,
    "figures": dispatch_figures,
    "b2-audit": dispatch_b2_audit,
    "guard-run": dispatch_guard_run,
    "guided-bootstrap-loop": dispatch_guided_bootstrap_loop,
    "guarded-league-bootstrap": dispatch_guarded_league_bootstrap,
}


def dispatch_workflow_command(args: argparse.Namespace, *, python_exe: str | None = None) -> None:
    """Run the workflow selected by already-parsed package CLI arguments."""

    repo_root = resolve_repo_root(args.repo_root)
    resolved_python_exe = sys.executable if python_exe is None else python_exe
    command = str(args.command)
    handler = _WORKFLOW_HANDLERS.get(command)
    if handler is None:
        raise AssertionError(f"Unhandled workflow command: {args.command}")
    handler(args, repo_root=repo_root, python_exe=resolved_python_exe)
