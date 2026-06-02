from __future__ import annotations

import sys

from weiss_rl.workflows.evaluation_workflow.dispatch import dispatch_evaluation_command
from weiss_rl.workflows.parsers import _add_common, _parse_args, build_parser
from weiss_rl.workflows.planning import _repo_root
from weiss_rl.workflows.public_api import PUBLIC_WORKFLOW_EXPORTS, export_public_workflow_symbols
from weiss_rl.workflows.training_workflow.dispatch import dispatch_training_command
from weiss_rl.workflows.workflow_dispatch import dispatch_workflow_command

__all__ = [
    *PUBLIC_WORKFLOW_EXPORTS,
    "_add_common",
    "_parse_args",
    "build_parser",
    "dispatch_evaluation_command",
    "dispatch_training_command",
    "main",
]

export_public_workflow_symbols(globals())


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root(args.repo_root)
    python_exe = sys.executable

    if dispatch_workflow_command(args=args, repo_root=repo_root, python_exe=python_exe):
        return

    raise AssertionError(f"Unhandled workflow command: {args.command}")


if __name__ == "__main__":
    main()
