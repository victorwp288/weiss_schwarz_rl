"""Stable six-command thesis workflow front door."""

from __future__ import annotations

from weiss_rl.workflows.runner import PUBLIC_THESIS_COMMANDS, build_parser, main

__all__ = ["PUBLIC_THESIS_COMMANDS", "build_parser", "main"]


if __name__ == "__main__":
    main()
