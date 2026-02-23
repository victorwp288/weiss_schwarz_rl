from __future__ import annotations

from weiss_rl.repro import derive_actor_seed, derive_episode_seed


def test_seed_derivation_is_deterministic() -> None:
    actor_seed_a = derive_actor_seed(20260212, actor_id=4)
    actor_seed_b = derive_actor_seed(20260212, actor_id=4)
    assert actor_seed_a == actor_seed_b

    episode_seed_a = derive_episode_seed(actor_seed_a, env_id=1, episode_index=77)
    episode_seed_b = derive_episode_seed(actor_seed_b, env_id=1, episode_index=77)
    assert episode_seed_a == episode_seed_b
