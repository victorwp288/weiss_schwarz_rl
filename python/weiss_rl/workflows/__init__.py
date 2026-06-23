"""Public thesis workflow command surface."""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    from weiss_rl.workflows.runner import main as runner_main

    runner_main()
