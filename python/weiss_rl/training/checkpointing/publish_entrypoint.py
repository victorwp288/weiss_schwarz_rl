#!/usr/bin/env python3
"""Publish a numbered training checkpoint as a registry snapshot candidate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.training.checkpointing.publish_cli import (
    parse_checkpoint_publish_args as _parse_checkpoint_publish_args,
)
from weiss_rl.training.checkpointing.publish_reporting import (
    checkpoint_publish_output_text as _checkpoint_publish_output_text,
)
from weiss_rl.training.checkpointing.publish_runtime import run_checkpoint_publish as _run_checkpoint_publish


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parse_checkpoint_publish_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = _run_checkpoint_publish(parse_args(argv))
    print(_checkpoint_publish_output_text(result.result), flush=True)


if __name__ == "__main__":
    main()
