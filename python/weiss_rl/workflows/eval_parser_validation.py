from __future__ import annotations

import argparse
import sys


def _resolve_run_label(parser: argparse.ArgumentParser, run_label: str, run_id_alias: str) -> str:
    normalized_label = run_label.strip()
    normalized_alias = run_id_alias.strip()
    if normalized_label and normalized_alias and normalized_label != normalized_alias:
        parser.error("--run-label and deprecated --run-id must match when both are provided")
    if normalized_alias:
        print("Warning: --run-id is deprecated; use --run-label instead.", file=sys.stderr)
    return normalized_label or normalized_alias


def _require_positive_int(parser: argparse.ArgumentParser, flag_name: str, value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized < 1:
        parser.error(f"{flag_name} must be >= 1")
    return normalized


__all__ = ["_require_positive_int", "_resolve_run_label"]
