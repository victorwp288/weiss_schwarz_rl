"""Seat-swapped matchup schedules and execution."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from weiss_rl.eval.policies.fixed_panel import deck_id_for_policy_id
from weiss_rl.eval.simulator.completed_games import record_completed_game, summarize_game_records, write_episodes_jsonl
from weiss_rl.eval.simulator.records import EvalGameRunner, EvalRunResult, ScheduledGame


def build_seat_swapped_schedule(
    *,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seeds: Sequence[int],
) -> list[ScheduledGame]:
    """Build two games per seed so each policy plays first and second seat."""

    schedule: list[ScheduledGame] = []
    for pair_index, raw_seed in enumerate(paired_seeds):
        episode_seed = int(raw_seed)
        schedule.append(
            ScheduledGame(
                pair_index=pair_index,
                swap_index=0,
                episode_index=len(schedule),
                episode_seed=episode_seed,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                seat0_policy_id=focal_policy_id,
                seat1_policy_id=opponent_policy_id,
                focal_seat=0,
                seat0_deck=deck_id_for_policy_id(focal_policy_id),
                seat1_deck=deck_id_for_policy_id(opponent_policy_id),
            )
        )
        schedule.append(
            ScheduledGame(
                pair_index=pair_index,
                swap_index=1,
                episode_index=len(schedule),
                episode_seed=episode_seed,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                seat0_policy_id=opponent_policy_id,
                seat1_policy_id=focal_policy_id,
                focal_seat=1,
                seat0_deck=deck_id_for_policy_id(opponent_policy_id),
                seat1_deck=deck_id_for_policy_id(focal_policy_id),
            )
        )
    return schedule


def run_seat_swapped_matchup(
    *,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seeds: Sequence[int],
    runner: EvalGameRunner,
    episodes_path: Path,
    run_id256: str | bytes,
    config_hash256: str,
    spec_hash256: str,
) -> EvalRunResult:
    """Run the paired-seat schedule, write episode records, and summarize it."""

    schedule = build_seat_swapped_schedule(
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        paired_seeds=paired_seeds,
    )
    records = [
        record_completed_game(
            scheduled_game=game,
            result=runner.run_game(game),
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
        )
        for game in schedule
    ]
    write_episodes_jsonl(episodes_path, records)
    return EvalRunResult(
        episodes_path=episodes_path,
        records=tuple(records),
        summary=summarize_game_records(records),
    )


__all__ = [
    "build_seat_swapped_schedule",
    "run_seat_swapped_matchup",
]
