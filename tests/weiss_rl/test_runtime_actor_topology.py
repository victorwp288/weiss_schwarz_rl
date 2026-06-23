from __future__ import annotations

from weiss_rl.artifacts.reproducibility import derive_actor_seed
from weiss_rl.runtime.components.topology import actor_seed as topology_actor_seed
from weiss_rl.runtime.components.topology import resolve_actor_topology


def test_resolve_actor_topology_keeps_ordered_runtime_strict_layout() -> None:
    actor_count, envs_per_actor = resolve_actor_topology(
        num_envs=96,
        runtime_mode="train_ordered",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 12
    assert envs_per_actor == 8


def test_resolve_actor_topology_prefers_fatter_async_collectors() -> None:
    actor_count, envs_per_actor = resolve_actor_topology(
        num_envs=96,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 2
    assert envs_per_actor == 48


def test_resolve_actor_topology_prefers_64_envs_per_actor_when_available() -> None:
    actor_count, envs_per_actor = resolve_actor_topology(
        num_envs=128,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 2
    assert envs_per_actor == 64


def test_resolve_actor_topology_prefers_6x64_over_8x48_for_384_envs() -> None:
    actor_count, envs_per_actor = resolve_actor_topology(
        num_envs=384,
        runtime_mode="train_async_fast",
        configured_actor_count=12,
        configured_envs_per_actor=8,
    )

    assert actor_count == 6
    assert envs_per_actor == 64


def test_topology_actor_seed_matches_reproducibility_contract() -> None:
    assert topology_actor_seed(20260514, 0) == derive_actor_seed(20260514, actor_id=0)
    assert topology_actor_seed(20260514, 5) == derive_actor_seed(20260514, actor_id=5)
