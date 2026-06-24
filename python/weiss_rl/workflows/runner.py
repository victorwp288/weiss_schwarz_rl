"""Dispatch the small public thesis workflow surface."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from weiss_rl.workflows.command_surface import PUBLIC_THESIS_COMMANDS
from weiss_rl.workflows.parsers import _parse_args, build_parser
from weiss_rl.workflows.planning import _repo_root
from weiss_rl.workflows.workflow_dispatch import dispatch_workflow_command

__all__ = ["PUBLIC_THESIS_COMMANDS", "build_parser", "main"]


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = _repo_root(args.repo_root)
    python_exe = sys.executable

    if dispatch_workflow_command(args=args, repo_root=repo_root, python_exe=python_exe):
        return

    raise AssertionError(f"Unhandled workflow command: {args.command}")


if __name__ == "__main__":
    main()
