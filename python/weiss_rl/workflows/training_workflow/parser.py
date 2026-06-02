from __future__ import annotations

import argparse
from collections.abc import Callable

from weiss_rl.workflows.training_workflow.parser_arguments import (
    add_train_b1_parser,
    add_train_main_parser,
)


def add_training_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    add_common: Callable[[argparse.ArgumentParser], None],
) -> None:
    add_train_b1_parser(subparsers, add_common)
    add_train_main_parser(subparsers, add_common)


__all__ = [
    "add_train_b1_parser",
    "add_train_main_parser",
    "add_training_parsers",
]
