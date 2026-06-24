"""Candidate scoring summaries for simulator god-search decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from weiss_rl.eval.simulator.harness import ScheduledGame


def root_logit_by_action(*, root_logits: np.ndarray, candidates: Sequence[int]) -> dict[int, float]:
    logits = np.asarray(root_logits, dtype=np.float32)
    return {int(action): float(logits[int(action)]) for action in candidates}


def mean_candidate_scores(candidate_scores: Mapping[int, Sequence[float]]) -> dict[int, float]:
    return {
        int(action): (sum(scores) / float(len(scores)) if scores else float("-inf"))
        for action, scores in candidate_scores.items()
    }


def select_god_search_candidate(
    *,
    candidates: Sequence[int],
    mean_scores: Mapping[int, float],
    root_logits: Mapping[int, float],
) -> int:
    return int(max(candidates, key=lambda action: (mean_scores[int(action)], root_logits[int(action)])))


def build_god_search_trace(
    *,
    scheduled_game: ScheduledGame,
    decision_id: int,
    current_seat: int,
    current_policy_id: str,
    opponent_policy_id: str,
    base_action: int,
    selected_action: int,
    candidates: Sequence[int],
    mean_scores: Mapping[int, float],
    root_logits: Mapping[int, float],
    rollout_details: Mapping[int, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "pair_index": int(scheduled_game.pair_index),
        "swap_index": int(scheduled_game.swap_index),
        "episode_seed": int(scheduled_game.episode_seed),
        "decision_id": int(decision_id),
        "actor_seat": int(current_seat),
        "current_policy_id": current_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "base_action": int(base_action),
        "selected_action": int(selected_action),
        "candidates": [
            {
                "action": int(action),
                "mean_score": float(mean_scores[int(action)]),
                "root_logit": float(root_logits[int(action)]),
                "rollouts": list(rollout_details[int(action)]),
            }
            for action in candidates
        ],
    }


__all__ = [
    "build_god_search_trace",
    "mean_candidate_scores",
    "root_logit_by_action",
    "select_god_search_candidate",
]
