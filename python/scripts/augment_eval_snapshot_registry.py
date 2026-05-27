#!/usr/bin/env python3
"""Build an eval-only snapshot registry by copying source champion snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.experiments.eval_registry_augmentation import augment_eval_snapshot_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment a run's eval registry with source snapshots")
    parser.add_argument("--target-run-dir", required=True, type=Path)
    parser.add_argument("--source-registry-json", required=True, type=Path)
    parser.add_argument("--output-registry-json", type=Path, default=None)
    parser.add_argument("--include-policy-id", action="append", default=[])
    parser.add_argument("--include-source-champions", action="store_true")
    parser.add_argument("--no-mark-imported-champions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = augment_eval_snapshot_registry(
        target_run_dir=args.target_run_dir,
        source_registry_json=args.source_registry_json,
        output_registry_json=args.output_registry_json,
        include_policy_ids=tuple(args.include_policy_id),
        include_source_champions=bool(args.include_source_champions),
        mark_imported_champions=not bool(args.no_mark_imported_champions),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
