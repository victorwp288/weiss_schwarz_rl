from __future__ import annotations

from weiss_rl.workflows.cli_dispatch import dispatch_workflow_command
from weiss_rl.workflows.cli_parser import parse_workflow_args


def main() -> None:
    dispatch_workflow_command(parse_workflow_args())


if __name__ == "__main__":
    main()
