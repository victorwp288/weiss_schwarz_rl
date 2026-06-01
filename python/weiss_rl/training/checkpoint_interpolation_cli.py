"""CLI parser for checkpoint interpolation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_checkpoint_interpolation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linearly interpolate two compatible RL checkpoints")
    parser.add_argument("--first-checkpoint", type=Path, required=True)
    parser.add_argument("--second-checkpoint", type=Path, required=True)
    parser.add_argument("--first-run-dir", type=Path, required=True)
    parser.add_argument("--second-run-dir", type=Path, required=True)
    parser.add_argument("--second-weight", type=float, required=True, help="Interpolation weight for second checkpoint")
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--policy-id", default="trajectory_bc_latest")
    parser.add_argument(
        "--allow-config-hash-mismatch",
        action="store_true",
        help="Allow interpolation when config_hash256 differs but spec/model checkpoint fields are compatible.",
    )
    return parser


def parse_checkpoint_interpolation_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_checkpoint_interpolation_parser().parse_args(argv)


__all__ = ["build_checkpoint_interpolation_parser", "parse_checkpoint_interpolation_args"]
