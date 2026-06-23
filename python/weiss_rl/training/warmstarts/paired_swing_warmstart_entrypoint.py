"""Entrypoint for paired-swing warmstart runs."""

from __future__ import annotations

from collections.abc import Sequence

from weiss_rl.training.warmstarts.paired_swing_warmstart_runtime import (
    run_paired_swing_warmstart as _run_paired_swing_warmstart,
)


def main(argv: Sequence[str] | None = None) -> int:
    return _run_paired_swing_warmstart(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
