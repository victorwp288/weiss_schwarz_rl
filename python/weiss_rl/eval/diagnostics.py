"""Diagnostic reports for seat-swapped evaluation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weiss_rl.eval.harness import EvalGameRecord
from weiss_rl.eval.payoff_folding import paired_seed_scores

__all__ = [
    "build_seat_advantage_diagnostics",
    "write_matchup_diagnostics_json",
]


def build_seat_advantage_diagnostics(records: tuple[EvalGameRecord, ...] | list[EvalGameRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("seat diagnostics require at least one record")

    paired_seed_scores(records, scheme="S0")
    focal_policy_id, opponent_policy_id = _matchup_ids(records)
    policy_ids = (focal_policy_id, opponent_policy_id)
    policy_breakdown = {
        policy_id: {
            "games_as_seat0": 0,
            "games_as_seat1": 0,
            "wins_as_seat0": 0,
            "wins_as_seat1": 0,
            "total_wins": 0,
        }
        for policy_id in policy_ids
    }

    seat0_wins = 0
    seat1_wins = 0
    draws = 0
    truncations = 0
    engine_errors = 0

    for record in records:
        policy_breakdown[record.seat0_policy_id]["games_as_seat0"] += 1
        policy_breakdown[record.seat1_policy_id]["games_as_seat1"] += 1
        if int(record.engine_status) != 0:
            engine_errors += 1

        winner_seat = _winner_seat(record)
        if winner_seat is None:
            if record.outcome == "D":
                draws += 1
            else:
                truncations += 1
            continue

        if winner_seat == 0:
            seat0_wins += 1
            policy_breakdown[record.seat0_policy_id]["wins_as_seat0"] += 1
            policy_breakdown[record.seat0_policy_id]["total_wins"] += 1
        else:
            seat1_wins += 1
            policy_breakdown[record.seat1_policy_id]["wins_as_seat1"] += 1
            policy_breakdown[record.seat1_policy_id]["total_wins"] += 1

    decisive_games = seat0_wins + seat1_wins
    total_games = decisive_games + draws + truncations
    return {
        "focal_policy_id": focal_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "pair_count": len({int(record.pair_index) for record in records}),
        "seat_results": {
            "seat0_wins": seat0_wins,
            "seat1_wins": seat1_wins,
            "draws": draws,
            "truncations": truncations,
            "engine_errors": engine_errors,
            "decisive_games": decisive_games,
            "total_games": total_games,
            "seat0_win_rate": (seat0_wins / decisive_games) if decisive_games else None,
            "seat1_win_rate": (seat1_wins / decisive_games) if decisive_games else None,
        },
        "policy_breakdown": policy_breakdown,
    }


def write_matchup_diagnostics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _matchup_ids(records: tuple[EvalGameRecord, ...] | list[EvalGameRecord]) -> tuple[str, str]:
    focal_ids = {record.focal_policy_id for record in records}
    opponent_ids = {record.opponent_policy_id for record in records}
    if len(focal_ids) != 1 or len(opponent_ids) != 1:
        raise ValueError("seat diagnostics expect records for exactly one focal/opponent matchup")
    return next(iter(focal_ids)), next(iter(opponent_ids))


def _winner_seat(record: EvalGameRecord) -> int | None:
    if record.outcome == "W":
        return int(record.focal_seat)
    if record.outcome == "L":
        return 1 - int(record.focal_seat)
    return None
