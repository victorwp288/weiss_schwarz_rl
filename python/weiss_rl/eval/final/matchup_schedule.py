"""Seat-swap scheduling and stable names for final-eval matchups."""

from __future__ import annotations

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, stable_hash64
from weiss_rl.eval.harness import ScheduledGame
from weiss_rl.eval.policies.set import deck_id_for_policy_id


def scheduled_game(
    *,
    pair_index: int,
    swap_index: int,
    episode_seed: int,
    focal_policy_id: str,
    opponent_policy_id: str,
) -> ScheduledGame:
    if swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1
    return ScheduledGame(
        pair_index=pair_index,
        swap_index=swap_index,
        episode_index=pair_index * 2 + swap_index,
        episode_seed=episode_seed,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id,
        seat1_policy_id=seat1_policy_id,
        focal_seat=focal_seat,
        seat0_deck=deck_id_for_policy_id(seat0_policy_id),
        seat1_deck=deck_id_for_policy_id(seat1_policy_id),
    )


def matchup_dir_name(*, focal_index: int, opponent_index: int, focal_policy_id: str, opponent_policy_id: str) -> str:
    return f"{focal_index:02d}_{slug(focal_policy_id)}__vs__{opponent_index:02d}_{slug(opponent_policy_id)}"


def slug(value: str) -> str:
    parts = [
        "".join(char.lower() for char in chunk if char.isalnum())
        for chunk in str(value).replace("-", " ").replace("_", " ").split()
    ]
    normalized = "_".join(part for part in parts if part)
    return normalized or "policy"


def bootstrap_seed(*, focal_policy_id: str, opponent_policy_id: str) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "final_eval_bootstrap_v1",
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
            }
        )
    )


__all__ = [
    "bootstrap_seed",
    "matchup_dir_name",
    "scheduled_game",
    "slug",
]
