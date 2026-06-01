from __future__ import annotations

import argparse

from weiss_rl.workflows.eval_parser_arguments import (
    add_canonical_eval_arguments,
    add_eval_common_arguments,
    add_public_demo_arguments,
    add_summary_only_arguments,
)
from weiss_rl.workflows.eval_parser_validation import _require_positive_int, _resolve_run_label


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation entrypoint for canonical final_eval or summary-only reports"
    )
    add_eval_common_arguments(parser)
    add_public_demo_arguments(parser)
    add_canonical_eval_arguments(parser)
    add_summary_only_arguments(parser)
    return parser


__all__ = [
    "_require_positive_int",
    "_resolve_run_label",
    "add_canonical_eval_arguments",
    "add_eval_common_arguments",
    "add_public_demo_arguments",
    "add_summary_only_arguments",
    "build_eval_parser",
]
