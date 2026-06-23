from __future__ import annotations

import argparse

from weiss_rl.workflows.eval_support import eval_parser_arguments as _arguments


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluation entrypoint for canonical final_eval or summary-only reports"
    )
    _arguments.add_eval_common_arguments(parser)
    _arguments.add_public_demo_arguments(parser)
    _arguments.add_canonical_eval_arguments(parser)
    _arguments.add_summary_only_arguments(parser)
    return parser


__all__ = [
    "build_eval_parser",
]
