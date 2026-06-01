#!/usr/bin/env python3
"""Gate paired-flip target coverage before repair dataset construction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from weiss_rl.experiments.paired_flip_targets_gate import (
    build_paired_flip_targets_gate_config,
    evaluate_paired_flip_targets_gate,
    write_paired_flip_targets_gate,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-json", action="append", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-total-targets", type=int, default=1)
    parser.add_argument("--min-target-opponents", type=int, default=1)
    parser.add_argument("--min-distinct-pair-indices", type=int, default=1)
    parser.add_argument("--excluded-pair-index", action="append", default=[], type=int)
    parser.add_argument("--required-opponent", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_paired_flip_targets_gate(
        build_paired_flip_targets_gate_config(
            target_jsons=args.target_json,
            min_total_targets=args.min_total_targets,
            min_target_opponents=args.min_target_opponents,
            min_distinct_pair_indices=args.min_distinct_pair_indices,
            excluded_pair_indices=args.excluded_pair_index,
            required_opponents=args.required_opponent,
        )
    )
    write_paired_flip_targets_gate(args.output_json, report)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "passed": bool(report["passed"]),
                "failures": report["failures"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
