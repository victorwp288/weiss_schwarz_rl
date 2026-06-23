#!/usr/bin/env python3
"""Linearly interpolate two compatible RL checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.training.checkpointing.interpolation_cli import (
    parse_checkpoint_interpolation_args as _parse_checkpoint_interpolation_args,
)
from weiss_rl.training.checkpointing.interpolation_reporting import (
    checkpoint_interpolation_output_line as _checkpoint_interpolation_output_line,
)
from weiss_rl.training.checkpointing.interpolation_runtime import (
    run_checkpoint_interpolation as _run_checkpoint_interpolation,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parse_checkpoint_interpolation_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = _run_checkpoint_interpolation(parse_args(argv))
    print(
        _checkpoint_interpolation_output_line(
            checkpoint_path=result.checkpoint_path,
            summary_path=result.summary_path,
            second_weight=float(result.summary["second_weight"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
