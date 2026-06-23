"""Entrypoint for trajectory BC warmstart runs."""

from __future__ import annotations

from collections.abc import Sequence

from weiss_rl.training.warmstarts.trajectory_bc_warmstart_runtime import (
    run_trajectory_bc_warmstart as _run_trajectory_bc_warmstart,
)


def main(argv: Sequence[str] | None = None) -> int:
    return _run_trajectory_bc_warmstart(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
