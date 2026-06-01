#!/usr/bin/env python3
"""Publish a numbered training checkpoint as a registry snapshot candidate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.training import checkpoint_publish as _checkpoint_publish
from weiss_rl.training.checkpoint_publish_cli import (
    build_checkpoint_publish_parser,
    parse_checkpoint_publish_args,
)
from weiss_rl.training.checkpoint_publish_reporting import checkpoint_publish_output_text
from weiss_rl.training.checkpoint_publish_runtime import run_checkpoint_publish

publish_checkpoint_snapshot = _checkpoint_publish.publish_checkpoint_snapshot
_build_parser = build_checkpoint_publish_parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_checkpoint_publish_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run_checkpoint_publish(parse_args(argv))
    print(checkpoint_publish_output_text(result.result), flush=True)


if __name__ == "__main__":
    main()
