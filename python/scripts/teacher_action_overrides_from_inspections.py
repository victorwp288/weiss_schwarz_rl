#!/usr/bin/env python3
"""Export teacher-action overrides from replay inspection top differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.teacher_action_overrides import (
    TeacherActionOverrideExportConfig,
    build_teacher_action_overrides_from_inspections,
    write_teacher_action_overrides_jsonl,
    write_teacher_action_overrides_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspection-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing replay inspection JSON files. May be repeated.",
    )
    parser.add_argument(
        "--inspection-json",
        type=Path,
        action="append",
        default=[],
        help="Single replay inspection JSON file. May be repeated.",
    )
    parser.add_argument("--min-total-variation", type=float, default=0.0)
    parser.add_argument("--include-matches", action="store_true")
    parser.add_argument("--max-rows-per-bundle", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspection_jsons = list(args.inspection_json)
    for inspection_dir in args.inspection_dir:
        if not inspection_dir.is_dir():
            raise SystemExit(f"inspection directory not found: {inspection_dir}")
        inspection_jsons.extend(sorted(inspection_dir.glob("*.json")))
    if not inspection_jsons:
        raise SystemExit("provide --inspection-dir or --inspection-json")

    rows, summary = build_teacher_action_overrides_from_inspections(
        TeacherActionOverrideExportConfig(
            inspection_jsons=tuple(path.resolve() for path in inspection_jsons),
            min_total_variation=float(args.min_total_variation),
            include_matches=bool(args.include_matches),
            max_rows_per_bundle=args.max_rows_per_bundle,
            max_rows=args.max_rows,
        )
    )
    write_teacher_action_overrides_jsonl(args.output_jsonl, rows)
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    write_teacher_action_overrides_summary(summary_path, summary)
    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl.as_posix(),
                "summary_json": summary_path.as_posix(),
                "row_count": len(rows),
                "bundle_count": summary["bundle_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
