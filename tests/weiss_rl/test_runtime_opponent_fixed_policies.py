from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from weiss_rl.runtime.components.opponents import (
    active_assigned_opponent_policy_ids,
    configured_fixed_opponent_policy_ids,
    configured_resident_opponent_policy_ids,
    fixed_opponent_policy_is_active,
    fixed_opponent_policy_slots,
)


def test_fixed_opponent_policy_slots_returns_anchor_prefix_or_none() -> None:
    assert (
        fixed_opponent_policy_slots(
            envs_per_actor=3,
            heuristic_reserved_envs=0,
            noleague_reserved_envs=0,
            heuristic_policy_id="heuristic",
            noleague_policy_id="baseline",
        )
        is None
    )

    slots = fixed_opponent_policy_slots(
        envs_per_actor=3,
        heuristic_reserved_envs=2,
        noleague_reserved_envs=4,
        heuristic_policy_id="heuristic",
        noleague_policy_id="baseline",
    )

    assert slots is not None
    assert slots.dtype == np.dtype(object)
    assert slots.tolist() == ["heuristic", "heuristic", "baseline"]


def test_fixed_opponent_policy_is_active_preserves_forced_and_scheduled_rules() -> None:
    league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=5),
        sampling=SimpleNamespace(heuristic_public_start_updates=3),
    )

    assert (
        fixed_opponent_policy_is_active(
            policy_id="heuristic",
            forced_policy_ids=(),
            heuristic_policy_ids=("heuristic",),
            opponent_model_ids=(),
            league_config=league_config,
            reference_update=2,
            noleague_policy_id="baseline",
        )
        is False
    )
    assert (
        fixed_opponent_policy_is_active(
            policy_id="heuristic",
            forced_policy_ids=("heuristic",),
            heuristic_policy_ids=("heuristic",),
            opponent_model_ids=(),
            league_config=None,
            reference_update=0,
            noleague_policy_id="baseline",
        )
        is True
    )
    assert (
        fixed_opponent_policy_is_active(
            policy_id="baseline",
            forced_policy_ids=(),
            heuristic_policy_ids=(),
            opponent_model_ids=("baseline",),
            league_config=league_config,
            reference_update=5,
            noleague_policy_id="baseline",
        )
        is True
    )


def test_active_assigned_opponent_policy_ids_filters_mirror_empty_and_duplicates() -> None:
    actors = (
        SimpleNamespace(opponent_policy_id_by_env=np.asarray(["mirror", " policy_a ", "", "policy_b"], dtype=object)),
        SimpleNamespace(opponent_policy_id_by_env=np.asarray(["policy_a", "policy_c"], dtype=object)),
        SimpleNamespace(opponent_policy_id_by_env=None),
    )

    assert active_assigned_opponent_policy_ids(actors=actors, mirror_policy_id="mirror") == (
        "policy_a",
        "policy_b",
        "policy_c",
    )


def test_configured_fixed_and_resident_opponent_policy_ids_preserve_order_and_deduplicate() -> None:
    fixed_ids = configured_fixed_opponent_policy_ids(
        heuristic_reserved_envs_per_actor=2,
        noleague_baseline_reserved_envs_per_actor=1,
        heuristic_policy_id="heuristic",
        noleague_policy_id="baseline",
        heuristic_policy_ids=("heuristic",),
    )

    assert fixed_ids == ("heuristic", "baseline")
    assert (
        configured_fixed_opponent_policy_ids(
            heuristic_reserved_envs_per_actor=2,
            noleague_baseline_reserved_envs_per_actor=0,
            heuristic_policy_id="heuristic",
            noleague_policy_id="baseline",
            heuristic_policy_ids=(),
        )
        == ()
    )
    assert configured_resident_opponent_policy_ids(
        fixed_policy_ids=fixed_ids,
        heuristic_variant_mix_fraction=0.25,
        noleague_mix_fraction=0.4,
        heuristic_variant_policy_ids=("variant_a", "variant_b"),
        heuristic_policy_ids=("variant_b",),
        noleague_policy_id="baseline",
    ) == ("heuristic", "baseline", "variant_b")
