"""Outcome records for simulator god-search rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.envs.decision_env import DecisionBoundaryBatch
from weiss_rl.eval.simulator.harness import ScheduledGame, game_result_from_step


@dataclass(frozen=True)
class RolloutScore:
    """Scored rollout result plus the stats bucket it belongs to."""

    score: float
    detail: dict[str, Any]
    stat_bucket: str


def build_prefix_replay_failure_detail(
    *,
    reason: str,
    scheduled_game: ScheduledGame,
    root_decision_id: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    detail = {
        "score": 0.0,
        "status": "prefix_replay_failed",
        "reason": reason,
        "pair_index": int(scheduled_game.pair_index),
        "swap_index": int(scheduled_game.swap_index),
        "episode_seed": int(scheduled_game.episode_seed),
        "root_decision_id": int(root_decision_id),
    }
    if extra:
        detail.update(extra)
    return detail


def score_terminal_rollout(
    *,
    batch: DecisionBoundaryBatch,
    scheduled_game: ScheduledGame,
    last_acting_seat: int | None,
    rollout_decisions: int,
    cutoff: bool,
) -> RolloutScore:
    result = game_result_from_step(
        batch,
        env_index=0,
        acting_seat=last_acting_seat,
        episode_seed=scheduled_game.episode_seed,
        max_decisions=getattr(batch, "max_decisions", None),
        max_ticks=getattr(batch, "max_ticks", None),
        max_no_progress_decisions=None,
    )
    if bool(result.truncated) or cutoff:
        return RolloutScore(
            score=0.0,
            stat_bucket="truncated",
            detail={
                "score": 0.0,
                "status": "truncated",
                "winner_seat": result.winner_seat,
                "rollout_decisions": int(rollout_decisions),
            },
        )

    if result.winner_seat is None:
        score = 0.0
    else:
        score = 1.0 if int(result.winner_seat) == int(scheduled_game.focal_seat) else -1.0
    return RolloutScore(
        score=float(score),
        stat_bucket="terminal",
        detail={
            "score": float(score),
            "status": "terminal",
            "winner_seat": result.winner_seat,
            "rollout_decisions": int(rollout_decisions),
        },
    )


__all__ = [
    "RolloutScore",
    "build_prefix_replay_failure_detail",
    "score_terminal_rollout",
]
