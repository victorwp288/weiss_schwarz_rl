from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.eval.export import load_eval_game_records


@dataclass(frozen=True, slots=True)
class TargetedConfirmPrefixConfig:
    source_summary_json: Path
    output_summary_json: Path
    paired_seeds: int


def derive_targeted_confirm_prefix_summary(config: TargetedConfirmPrefixConfig) -> dict[str, Any]:
    paired_seeds = int(config.paired_seeds)
    if paired_seeds < 1:
        raise ValueError("paired_seeds must be >= 1")

    source_summary_json = config.source_summary_json.resolve()
    source = _read_json_object(source_summary_json)
    rows = source.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"targeted-confirm summary missing rows: {source_summary_json}")

    prefix_rows = [
        _derive_row_prefix(row, source_summary_json=source_summary_json, paired_seeds=paired_seeds)
        for row in rows
        if isinstance(row, dict)
    ]
    anchor_rows = prefix_rows[:5]
    league_rows = prefix_rows[5:]
    output = {
        **source,
        "derived_from": {
            "kind": "targeted_confirm_prefix_v1",
            "source_summary_json": source_summary_json.as_posix(),
            "source_paired_seeds": source.get("paired_seeds"),
            "paired_seed_prefix": paired_seeds,
        },
        "paired_seeds": paired_seeds,
        "games_per_row": paired_seeds * 2,
        "rows": prefix_rows,
        "overall": _summarize_rows(prefix_rows),
        "anchor_subset": _summarize_rows(anchor_rows),
        "legacy_subset": _summarize_rows(league_rows),
    }
    output["output_dir"] = str(config.output_summary_json.parent.as_posix())
    config.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _derive_row_prefix(row: dict[str, Any], *, source_summary_json: Path, paired_seeds: int) -> dict[str, Any]:
    raw_summary_path = row.get("summary_path")
    if not isinstance(raw_summary_path, str) or not raw_summary_path:
        raise ValueError(f"targeted-confirm row missing summary_path in {source_summary_json}")
    matchup_summary_path = _resolve_path(raw_summary_path, base=source_summary_json.parent)
    episodes_path = matchup_summary_path.with_name("episodes.jsonl")
    records = tuple(record for record in load_eval_game_records(episodes_path) if int(record.pair_index) < paired_seeds)
    if not records:
        raise ValueError(f"no prefix records found for {row.get('opponent_policy_id')} in {episodes_path}")
    observed_pairs = len({int(record.pair_index) for record in records})
    if observed_pairs != paired_seeds:
        raise ValueError(
            f"expected {paired_seeds} paired seeds for {row.get('opponent_policy_id')}, "
            f"found {observed_pairs} in {episodes_path}"
        )
    if len(records) != paired_seeds * 2:
        raise ValueError(
            f"expected {paired_seeds * 2} prefix games for {row.get('opponent_policy_id')}, "
            f"found {len(records)} in {episodes_path}"
        )
    wins = sum(1 for record in records if record.outcome == "W")
    losses = sum(1 for record in records if record.outcome == "L")
    draws = sum(1 for record in records if record.outcome == "D")
    truncations = sum(1 for record in records if bool(record.truncated))
    engine_errors = sum(1 for record in records if int(record.engine_status) != 0)
    return {
        **row,
        "paired_seeds": paired_seeds,
        "games": len(records),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "truncations": truncations,
        "engine_errors": engine_errors,
        "mean": wins / len(records),
        "ci_low": None,
        "ci_high": None,
        "prob_gt_half": None,
        "prefix_source_summary_path": row.get("summary_path"),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(_int(row.get("wins")) for row in rows)
    games = sum(_int(row.get("games")) for row in rows)
    summary: dict[str, Any] = {"wins": wins, "games": games}
    if games > 0:
        summary["mean"] = wins / games
    return summary


def _resolve_path(raw_path: str, *, base: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    return base / path


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
