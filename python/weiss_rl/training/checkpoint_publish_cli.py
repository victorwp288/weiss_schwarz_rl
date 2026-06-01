"""CLI parser for publishing checkpoint snapshots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_checkpoint_publish_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a numbered training checkpoint as a registry snapshot candidate."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument(
        "--policy-id",
        default=None,
        help="Snapshot policy id to publish. Defaults to checkpoint_<update:06d>.",
    )
    parser.add_argument("--pin", action="store_true", help="Pin the published snapshot in the registry.")
    parser.add_argument("--replace", action="store_true", help="Replace an existing snapshot with the same policy id.")
    return parser


def parse_checkpoint_publish_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_checkpoint_publish_parser().parse_args(argv)


__all__ = ["build_checkpoint_publish_parser", "parse_checkpoint_publish_args"]
