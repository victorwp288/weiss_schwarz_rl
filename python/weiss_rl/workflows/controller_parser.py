from __future__ import annotations

import argparse
from collections.abc import Callable

from weiss_rl.workflows.controller_parser_arguments import (
    add_guard_run_parser,
    add_guarded_league_bootstrap_parser,
    add_guided_bootstrap_loop_parser,
)


def add_controller_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> None:
    add_guard_run_parser(subparsers, add_common)
    add_guided_bootstrap_loop_parser(subparsers, add_common)
    add_guarded_league_bootstrap_parser(subparsers, add_common)


__all__ = [
    "add_controller_parsers",
    "add_guard_run_parser",
    "add_guarded_league_bootstrap_parser",
    "add_guided_bootstrap_loop_parser",
]
