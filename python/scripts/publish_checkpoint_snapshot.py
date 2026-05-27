#!/usr/bin/env python3
"""Publish a numbered training checkpoint as a registry snapshot candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weiss_rl.training.checkpoint_publish import publish_checkpoint_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument(
        "--policy-id",
        default=None,
        help="Snapshot policy id to publish. Defaults to checkpoint_<update:06d>.",
    )
    parser.add_argument("--pin", action="store_true", help="Pin the published snapshot in the registry.")
    parser.add_argument("--replace", action="store_true", help="Replace an existing snapshot with the same policy id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = publish_checkpoint_snapshot(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint_path,
        policy_id=args.policy_id,
        pin=bool(args.pin),
        replace=bool(args.replace),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
