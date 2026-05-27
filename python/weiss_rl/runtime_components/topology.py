from __future__ import annotations

from typing import Literal

from weiss_rl.artifacts.reproducibility import derive_actor_seed

QueueRuntimeMode = Literal["train_ordered", "train_async_fast"]


def resolve_actor_topology(
    *,
    num_envs: int,
    runtime_mode: QueueRuntimeMode,
    configured_actor_count: int,
    configured_envs_per_actor: int,
) -> tuple[int, int]:
    if runtime_mode != "train_async_fast":
        if int(num_envs) == int(configured_actor_count) * int(configured_envs_per_actor):
            return int(configured_actor_count), int(configured_envs_per_actor)
        return 1, int(num_envs)

    candidate_max_actors = max(1, int(configured_actor_count))
    divisors = [actor_count for actor_count in range(1, candidate_max_actors + 1) if num_envs % actor_count == 0]
    if not divisors:
        return 1, int(num_envs)

    def _score(actor_count: int) -> tuple[int, int, int]:
        envs_per_actor = int(num_envs // actor_count)
        in_band = 32 <= envs_per_actor <= 64
        band_penalty = 0 if in_band else 1
        target = 64 if in_band else 48
        return (band_penalty, abs(target - envs_per_actor), actor_count)

    best_actor_count = min(divisors, key=_score)
    return int(best_actor_count), int(num_envs // best_actor_count)


def actor_seed(base_seed: int, actor_id: int) -> int:
    return derive_actor_seed(int(base_seed), actor_id=int(actor_id))
