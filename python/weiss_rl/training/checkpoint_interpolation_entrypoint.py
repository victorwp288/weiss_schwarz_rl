#!/usr/bin/env python3
"""Linearly interpolate two compatible RL checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weiss_rl.training import checkpoint_interpolation as _checkpoint_interpolation
from weiss_rl.training.checkpoint_interpolation_cli import (
    build_checkpoint_interpolation_parser,
    parse_checkpoint_interpolation_args,
)
from weiss_rl.training.checkpoint_interpolation_reporting import checkpoint_interpolation_output_line
from weiss_rl.training.checkpoint_interpolation_runtime import (
    copy_contract_artifacts,
    load_checkpoint,
    model_state_dict,
    publish_interpolated_snapshot,
    run_checkpoint_interpolation,
    sha256_file,
    validate_checkpoint_contracts,
)

interpolate_model_state_dicts = _checkpoint_interpolation.interpolate_model_state_dicts
_build_parser = build_checkpoint_interpolation_parser
_load_checkpoint = load_checkpoint
_model_state_dict = model_state_dict
_validate_checkpoint_contracts = validate_checkpoint_contracts
_copy_contract_artifacts = copy_contract_artifacts
_publish_snapshot = publish_interpolated_snapshot
_sha256_file = sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return parse_checkpoint_interpolation_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_checkpoint_interpolation(parse_args(argv))
    print(
        checkpoint_interpolation_output_line(
            checkpoint_path=result.checkpoint_path,
            summary_path=result.summary_path,
            second_weight=float(result.summary["second_weight"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
