"""Deterministic seed formulas for training-time eval jobs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from weiss_rl.repro import canonical_json_bytes, stable_hash64

U64_MASK = (1 << 64) - 1


class ScheduledGameSeedFields(Protocol):
    pair_index: int
    swap_index: int
    episode_seed: int
    seat0_policy_id: str
    seat1_policy_id: str


def periodic_dev_eval_rng_seed(*, scheduled_game: ScheduledGameSeedFields, seat: int) -> int:
    payload = canonical_json_bytes(
        {
            "kind": "periodic_dev_eval_rng_v1",
            "pair_index": scheduled_game.pair_index,
            "swap_index": scheduled_game.swap_index,
            "episode_seed": scheduled_game.episode_seed,
            "seat": int(seat),
            "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
        }
    )
    return stable_hash64(payload)


def promotion_gate_rng_seed(*, scheduled_game: ScheduledGameSeedFields, seat: int) -> int:
    payload = canonical_json_bytes(
        {
            "kind": "promotion_gate_rng_v1",
            "pair_index": scheduled_game.pair_index,
            "swap_index": scheduled_game.swap_index,
            "episode_seed": scheduled_game.episode_seed,
            "seat": int(seat),
            "seat_policy_id": scheduled_game.seat0_policy_id if seat == 0 else scheduled_game.seat1_policy_id,
        }
    )
    return stable_hash64(payload)


def periodic_dev_eval_bootstrap_seed(*, update_count: int, policy_version: int) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "periodic_dev_eval_bootstrap_v1",
                "update_count": int(update_count),
                "policy_version": int(policy_version),
            }
        )
    )


def promotion_gate_bootstrap_seed(*, update_count: int, policy_version: int) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "promotion_gate_bootstrap_v1",
                "update_count": int(update_count),
                "policy_version": int(policy_version),
            }
        )
    )


def expand_periodic_dev_eval_paired_seeds(
    base_paired_seeds: Sequence[int],
    *,
    requested_pairs: int,
    seed_file_sha256: str,
    update_count: int,
    policy_version: int,
    scope: str,
) -> list[int]:
    requested_pairs_i = int(requested_pairs)
    paired_seeds = [int(seed) for seed in base_paired_seeds[:requested_pairs_i]]
    seen = set(paired_seeds)
    extra_index = 0
    while len(paired_seeds) < requested_pairs_i:
        derived_seed = (
            stable_hash64(
                canonical_json_bytes(
                    {
                        "kind": "periodic_dev_eval_confirmatory_seed_v1",
                        "scope": str(scope),
                        "seed_file_sha256": str(seed_file_sha256),
                        "update_count": int(update_count),
                        "policy_version": int(policy_version),
                        "extra_index": int(extra_index),
                    }
                )
            )
            & U64_MASK
        )
        extra_index += 1
        if derived_seed in seen:
            continue
        paired_seeds.append(int(derived_seed))
        seen.add(int(derived_seed))
    return paired_seeds
