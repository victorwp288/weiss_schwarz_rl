"""Entrypoint facade for trajectory BC warmstart runs."""

from __future__ import annotations

from collections.abc import Sequence

from weiss_rl.training.trajectory_bc_warmstart_cli import (
    build_trajectory_bc_warmstart_parser,
    parse_trajectory_bc_warmstart_args,
    validate_trajectory_bc_warmstart_args,
)
from weiss_rl.training.trajectory_bc_warmstart_runtime import (
    _initial_hidden_state,
    _publish_trajectory_bc_snapshot,
    _sha256_file,
    _write_run_contract_artifacts,
    run_trajectory_bc_warmstart,
)

_build_parser = build_trajectory_bc_warmstart_parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_trajectory_bc_warmstart(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_build_parser",
    "_initial_hidden_state",
    "_publish_trajectory_bc_snapshot",
    "_sha256_file",
    "_write_run_contract_artifacts",
    "main",
    "parse_trajectory_bc_warmstart_args",
    "run_trajectory_bc_warmstart",
    "validate_trajectory_bc_warmstart_args",
]
