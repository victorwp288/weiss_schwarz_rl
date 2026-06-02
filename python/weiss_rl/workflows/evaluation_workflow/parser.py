from __future__ import annotations

import argparse
from collections.abc import Callable

from weiss_rl.workflows.evaluation_workflow.parser_arguments import (
    add_b2_audit_parser,
    add_eval_final_parser,
    add_figures_parser,
    add_smoke_eval_parser,
)


def add_evaluation_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> None:
    add_smoke_eval_parser(subparsers, add_common)
    add_eval_final_parser(subparsers, add_common)
    add_figures_parser(subparsers, add_common)
    add_b2_audit_parser(subparsers, add_common)


__all__ = [
    "add_b2_audit_parser",
    "add_eval_final_parser",
    "add_evaluation_parsers",
    "add_figures_parser",
    "add_smoke_eval_parser",
]
