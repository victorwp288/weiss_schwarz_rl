from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.experiments.main_league_multiobjective_gate import FIXED_THESIS_OPPONENTS


def merge_targeted_confirm_summaries(paths: Sequence[Path], *, label: str = "merged") -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one targeted-confirm summary is required")
    rows_by_opponent: dict[str, dict[str, Any]] = {}
    source_summaries: list[str] = []
    paired_seeds: set[int] = set()
    for path in paths:
        payload = _read_json_object(path)
        source_summaries.append(path.as_posix())
        raw_paired_seeds = _optional_int(payload.get("paired_seeds"))
        if raw_paired_seeds is not None:
            paired_seeds.add(raw_paired_seeds)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"targeted-confirm summary missing rows: {path}")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            opponent = row.get("opponent_policy_id")
            if not isinstance(opponent, str) or not opponent:
                continue
            if opponent in rows_by_opponent:
                raise ValueError(f"duplicate opponent while merging targeted-confirm summaries: {opponent}")
            rows_by_opponent[opponent] = dict(row)
    rows = list(rows_by_opponent.values())
    return {
        "kind": "targeted_confirm_merged_summary_v1",
        "label": str(label),
        "created_unix": time.time(),
        "source_summaries": source_summaries,
        "paired_seeds": paired_seeds.pop() if len(paired_seeds) == 1 else None,
        "rows": rows,
        "anchor_subset": _summarize_rows(
            row for row in rows if _matches_any(str(row.get("opponent_policy_id") or ""), FIXED_THESIS_OPPONENTS)
        ),
        "legacy_subset": _summarize_rows(
            row for row in rows if not _matches_any(str(row.get("opponent_policy_id") or ""), FIXED_THESIS_OPPONENTS)
        ),
        "overall": _summarize_rows(rows),
    }


def write_merged_targeted_confirm_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_rows(rows: Sequence[Mapping[str, Any]] | Any) -> dict[str, float | int | None]:
    row_list = list(rows)
    wins = 0
    games = 0
    for row in row_list:
        row_wins = _optional_int(row.get("wins"))
        row_games = _optional_int(row.get("games"))
        if row_wins is None or row_games is None:
            continue
        wins += row_wins
        games += row_games
    return {
        "wins": wins,
        "games": games,
        "mean": None if games <= 0 else wins / games,
    }


def _matches_any(value: str, candidates: Sequence[str]) -> bool:
    return any(_is_seed_wrapped_suffix_match(value, candidate) for candidate in candidates)


def _is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
