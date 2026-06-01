from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath
from weiss_rl.runtime_components.opponents import (
    RuntimeOpponentGroup,
    active_actor_heuristic_fraction,
    active_assigned_opponent_policy_ids,
    active_heuristic_public_mix_fraction,
    active_heuristic_public_variant_mix_fraction,
    active_mirror_mix_fraction,
    active_noleague_baseline_mix_fraction,
    active_warmup_snapshot_mix_fraction,
    apply_opponent_pool_diversity_floor,
    build_runtime_opponent_sampling_groups,
    build_runtime_opponent_sampling_plan,
    configured_fixed_opponent_policy_ids,
    configured_hard_negative_focus_policy_ids,
    configured_resident_opponent_policy_ids,
    configured_row_deficit_policy_weights,
    filter_timeout_heavy_opponents,
    fixed_opponent_policy_is_active,
    fixed_opponent_policy_slots,
    hard_negative_focus_policy_id_matches,
    hard_negative_focus_weight_multipliers,
    promotion_gated_recent_reservoir_size,
    row_deficit_weight_multipliers,
    sample_runtime_opponent_group_policy_ids,
    sample_runtime_opponent_policy_ids,
    sample_warmup_snapshot_policy_ids,
    select_hard_negative_ids,
)


def test_active_mix_fractions_preserve_anneal_and_expiry_rules() -> None:
    league_config = SimpleNamespace(
        warmup=SimpleNamespace(first_updates=10),
        sampling=SimpleNamespace(
            heuristic_public_mix_fraction=1.0,
            heuristic_public_mix_end_updates=5,
            heuristic_public_final_mix_fraction=0.25,
            heuristic_public_variant_mix_fraction=0.4,
            heuristic_public_variant_mix_end_updates=4,
            heuristic_public_variant_final_mix_fraction=0.1,
            mirror_mix_fraction=0.6,
            mirror_mix_end_updates=6,
            mirror_final_mix_fraction=0.3,
            noleague_baseline_mix_fraction=0.3,
            noleague_baseline_mix_end_updates=3,
            warmup_snapshot_mix_fraction=0.2,
        ),
    )

    assert active_heuristic_public_mix_fraction(league_config=league_config, reference_update=3) == pytest.approx(0.55)
    assert active_heuristic_public_mix_fraction(league_config=league_config, reference_update=5) == pytest.approx(0.25)
    assert active_heuristic_public_variant_mix_fraction(
        league_config=league_config, reference_update=2
    ) == pytest.approx(0.25)
    assert active_heuristic_public_variant_mix_fraction(
        league_config=league_config, reference_update=4
    ) == pytest.approx(0.1)
    assert active_mirror_mix_fraction(league_config=league_config, reference_update=3) == pytest.approx(0.45)
    assert active_mirror_mix_fraction(league_config=league_config, reference_update=6) == pytest.approx(0.3)
    assert active_noleague_baseline_mix_fraction(league_config=league_config, reference_update=2) == pytest.approx(0.3)
    assert active_noleague_baseline_mix_fraction(league_config=league_config, reference_update=3) == pytest.approx(0.0)
    assert active_warmup_snapshot_mix_fraction(
        league_config=league_config,
        reference_update=9,
        has_opponent_candidates=True,
        has_opponent_models=True,
    ) == pytest.approx(0.2)
    assert active_warmup_snapshot_mix_fraction(
        league_config=league_config,
        reference_update=10,
        has_opponent_candidates=True,
        has_opponent_models=True,
    ) == pytest.approx(0.0)


def test_active_actor_heuristic_fraction_clamps_and_respects_delayed_start() -> None:
    assert active_actor_heuristic_fraction(
        initial_fraction=1.5,
        final_fraction=-0.5,
        start_updates=4,
        end_updates=8,
        reference_update=6,
    ) == pytest.approx(0.5)


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


def test_promotion_recent_reservoir_matches_champion_gate_rules() -> None:
    assert (
        promotion_gated_recent_reservoir_size(
            base_recent_size=0,
            champion_size=8,
            admitted_champion_ids=("champion",),
            min_recent_size=2,
        )
        == 0
    )
    assert (
        promotion_gated_recent_reservoir_size(
            base_recent_size=8,
            champion_size=4,
            admitted_champion_ids=(),
            min_recent_size=2,
        )
        == 4
    )
    assert (
        promotion_gated_recent_reservoir_size(
            base_recent_size=8,
            champion_size=4,
            admitted_champion_ids=("champion",),
            min_recent_size=2,
        )
        == 2
    )


def test_filter_timeout_heavy_opponents_and_diversity_floor_preserve_pool_safety() -> None:
    assert filter_timeout_heavy_opponents(
        candidate_ids=("a",),
        league_config=SimpleNamespace(promotion_gate_enabled=False),
        outcomes=None,
        min_samples=32,
    ) == ("a",)

    league_config = SimpleNamespace(
        promotion_gate_enabled=True,
        promotion=SimpleNamespace(gate=SimpleNamespace(guardrails=SimpleNamespace(max_truncation_rate=0.05))),
    )
    outcomes = SimpleNamespace(
        counts=lambda policy_id: {
            "timeout_heavy": (0, 0, 0, 40),
            "healthy": (40, 0, 0, 0),
            "too_few": (0, 0, 0, 2),
        }[policy_id]
    )

    filtered = filter_timeout_heavy_opponents(
        candidate_ids=("timeout_heavy", "healthy", "too_few"),
        league_config=league_config,
        outcomes=outcomes,
        min_samples=32,
    )

    assert filtered == ("healthy", "too_few")
    assert apply_opponent_pool_diversity_floor(
        candidate_ids=("timeout_heavy", "healthy", "too_few"),
        filtered_candidate_ids=filtered[:1],
        minimum_floor_size=2,
    ) == (("healthy", "timeout_heavy"), 2)
    assert apply_opponent_pool_diversity_floor(
        candidate_ids=("timeout_heavy", "healthy"),
        filtered_candidate_ids=(),
        minimum_floor_size=2,
    ) == (("timeout_heavy", "healthy"), 0)


def test_select_hard_negative_ids_filters_by_samples_and_win_rate(tmp_path: Path) -> None:
    registry = SnapshotRegistry(recent_size=4, champion_size=1)
    registry.add_snapshot(
        policy_id="hard_old",
        update=10,
        weights_sha256="a" * 64,
        path=snapshot_weights_relpath("hard_old"),
    )
    registry.add_snapshot(
        policy_id="hard_new",
        update=20,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("hard_new"),
    )
    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)

    outcomes = SimpleNamespace(
        counts=lambda policy_id: {
            "hard_low": (1, 9, 0, 0),
            "hard_old": (4, 6, 0, 0),
            "hard_new": (4, 6, 0, 0),
            "too_easy": (6, 4, 0, 0),
            "too_few": (0, 1, 0, 0),
        }[policy_id],
        win_rate=lambda policy_id: {
            "hard_low": 0.1,
            "hard_old": 0.4,
            "hard_new": 0.4,
            "too_easy": 0.6,
            "too_few": 0.0,
        }[policy_id],
    )
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            hard_negative_min_samples=4,
            hard_negative_max_win_rate=0.45,
        )
    )

    assert select_hard_negative_ids(
        candidate_ids=("too_easy", "hard_old", "too_few", "hard_new", "hard_low"),
        league_config=league_config,
        outcomes=outcomes,
        registry_path=registry_path,
    ) == ("hard_low", "hard_new", "hard_old")


def test_select_hard_negative_ids_pins_configured_focus_candidates_by_suffix() -> None:
    outcomes = SimpleNamespace(
        counts=lambda policy_id: {
            "seed_outer_seed_source_checkpoint_000025": (0, 1, 0, 0),
            "too_easy": (9, 1, 0, 0),
        }[policy_id],
        win_rate=lambda policy_id: {
            "seed_outer_seed_source_checkpoint_000025": 0.0,
            "too_easy": 0.9,
        }[policy_id],
    )
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            hard_negative_min_samples=4,
            hard_negative_max_win_rate=0.45,
            hard_negative_focus_policy_ids=("seed_source_checkpoint_000025",),
            hard_negative_focus_weight_multiplier=4.0,
        )
    )

    assert select_hard_negative_ids(
        candidate_ids=("too_easy", "seed_outer_seed_source_checkpoint_000025"),
        league_config=league_config,
        outcomes=outcomes,
        registry_path=None,
    ) == ("seed_outer_seed_source_checkpoint_000025",)


def test_select_hard_negative_ids_requires_candidates_league_and_outcomes() -> None:
    assert (
        select_hard_negative_ids(
            candidate_ids=("policy",),
            league_config=None,
            outcomes=SimpleNamespace(),
            registry_path=None,
        )
        == ()
    )


def test_hard_negative_focus_helpers_match_imported_suffix_and_build_multipliers() -> None:
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            hard_negative_focus_policy_ids=("seed_source_policy_000002",),
            hard_negative_focus_weight_multiplier=3.5,
        )
    )

    assert configured_hard_negative_focus_policy_ids(league_config=league_config) == ("seed_source_policy_000002",)
    assert hard_negative_focus_policy_id_matches(
        "seed_outer_seed_source_policy_000002",
        "seed_source_policy_000002",
    )
    assert hard_negative_focus_weight_multipliers(
        policy_ids=("seed_outer_seed_source_policy_000002", "other"),
        league_config=league_config,
    ) == {"seed_outer_seed_source_policy_000002": 3.5}


def test_row_deficit_helpers_match_imported_suffix_and_build_multipliers() -> None:
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            row_deficit_policy_weights=(
                ("seed_source_policy_000002", 2.0),
                ("seed_source_policy_000004", 3.0),
            ),
        )
    )

    assert configured_row_deficit_policy_weights(league_config=league_config) == (
        ("seed_source_policy_000002", 2.0),
        ("seed_source_policy_000004", 3.0),
    )
    assert row_deficit_weight_multipliers(
        policy_ids=("seed_outer_seed_source_policy_000002", "seed_outer_seed_source_policy_000004", "other"),
        league_config=league_config,
    ) == {
        "seed_outer_seed_source_policy_000002": 2.0,
        "seed_outer_seed_source_policy_000004": 3.0,
    }


class _SamplingOutcomes:
    _win_rates = {
        "hard_a": 0.2,
        "hard_b": 0.4,
        "champ_a": 0.5,
        "recent_a": 0.6,
        "recent_b": 0.1,
        "warm_a": 0.2,
        "warm_b": 0.8,
    }

    def win_rate(self, policy_id: str) -> float:
        return float(self._win_rates.get(str(policy_id), 0.5))


def test_build_runtime_opponent_sampling_groups_preserves_pre_pfsp_lane_order_and_mirror_remainder() -> None:
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=0.9,
            hard_negative_mix_fraction=0.9,
        )
    )

    groups = build_runtime_opponent_sampling_groups(
        league_config=league_config,
        pfsp_ready=False,
        reference_update=0,
        mirror_weight=0.9,
        heuristic_public_weight=0.2,
        heuristic_public_variant_weight=0.2,
        noleague_baseline_weight=0.2,
        warmup_snapshot_weight=0.2,
        opponent_candidate_ids=("warm_a", "warm_b"),
        opponent_hard_negative_ids=("hard_a",),
        opponent_champion_ids=("champ_a",),
        opponent_recent_ids=("recent_a",),
        opponent_heuristic_policy_ids=("heuristic", "aggro", "control"),
        opponent_model_ids=("baseline", "warm_a", "warm_b"),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control", "missing"),
        noleague_baseline_policy_id="baseline",
    )

    assert [(group.name, group.policy_ids, group.weight) for group in groups] == [
        ("heuristic_public", ("heuristic",), 0.2),
        ("heuristic_public_variant", ("aggro", "control"), 0.2),
        ("noleague_baseline", ("baseline",), 0.2),
        ("warmup_snapshot", ("warm_a", "warm_b"), 0.2),
        ("mirror", ("mirror",), pytest.approx(0.2)),
    ]


def test_build_runtime_opponent_sampling_groups_preserves_pfsp_ready_lane_order_and_recent_remainder() -> None:
    league_config = SimpleNamespace(
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=0.25,
            hard_negative_mix_fraction=0.15,
        )
    )

    groups = build_runtime_opponent_sampling_groups(
        league_config=league_config,
        pfsp_ready=True,
        reference_update=0,
        mirror_weight=0.2,
        heuristic_public_weight=0.1,
        heuristic_public_variant_weight=0.0,
        noleague_baseline_weight=0.1,
        warmup_snapshot_weight=0.3,
        opponent_candidate_ids=("champ_a", "hard_a", "recent_a"),
        opponent_hard_negative_ids=("hard_a",),
        opponent_champion_ids=("champ_a",),
        opponent_recent_ids=("recent_a",),
        opponent_heuristic_policy_ids=("heuristic",),
        opponent_model_ids=("baseline", "champ_a", "hard_a", "recent_a"),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=(),
        noleague_baseline_policy_id="baseline",
    )

    assert [(group.name, group.policy_ids, group.weight) for group in groups] == [
        ("heuristic_public", ("heuristic",), 0.1),
        ("noleague_baseline", ("baseline",), 0.1),
        ("mirror", ("mirror",), 0.2),
        ("hard_negative", ("hard_a",), 0.15),
        ("champion", ("champ_a",), 0.25),
        ("recent", ("recent_a",), pytest.approx(0.2)),
    ]


def test_runtime_opponent_sampling_plan_preserves_zero_weight_uniform_fallback() -> None:
    plan = build_runtime_opponent_sampling_plan(
        (
            RuntimeOpponentGroup(name="hard_negative", policy_ids=("hard_a",), weight=0.0),
            RuntimeOpponentGroup(name="champion", policy_ids=("champ_a",), weight=0.0),
        )
    )

    assert [group.name for group in plan.groups] == ["hard_negative", "champion"]
    assert np.array_equal(plan.probabilities, np.asarray([0.5, 0.5], dtype=np.float64))


def test_sample_runtime_opponent_group_policy_ids_preserves_variant_rng_draws() -> None:
    group = RuntimeOpponentGroup(name="heuristic_public_variant", policy_ids=("aggro", "control"), weight=1.0)

    sampled = sample_runtime_opponent_group_policy_ids(
        group=group,
        count=5,
        rng=np.random.default_rng(3),
        league_config=SimpleNamespace(pfsp_power=2.0, pfsp_epsilon_uniform=0.0),
        outcomes=_SamplingOutcomes(),
    )

    expected_indices = np.random.default_rng(3).integers(2, size=5)
    assert sampled == tuple(("aggro", "control")[int(index)] for index in expected_indices)


def test_sample_runtime_opponent_policy_ids_handles_empty_and_no_league_cases() -> None:
    common: dict[str, Any] = dict(
        rng=np.random.default_rng(1),
        league_config=None,
        pfsp_ready=False,
        reference_update=0,
        mirror_weight=0.0,
        heuristic_public_weight=0.0,
        heuristic_public_variant_weight=0.0,
        noleague_baseline_weight=0.0,
        warmup_snapshot_weight=0.0,
        opponent_candidate_ids=(),
        opponent_hard_negative_ids=(),
        opponent_champion_ids=(),
        opponent_recent_ids=(),
        opponent_heuristic_policy_ids=(),
        opponent_model_ids=(),
        outcomes=None,
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control"),
        noleague_baseline_policy_id="baseline",
    )

    empty = sample_runtime_opponent_policy_ids(count=0, league_enabled=False, **common)
    no_league = sample_runtime_opponent_policy_ids(count=3, league_enabled=False, **common)

    assert empty.policy_ids == ()
    assert empty.sampled_envs == 0
    assert no_league.policy_ids == ("mirror", "mirror", "mirror")
    assert no_league.sampled_envs == 0
    assert no_league.mirror_envs == 3


def test_sample_runtime_opponent_policy_ids_preserves_pfsp_ready_rng_order_and_counters() -> None:
    league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=0.25,
            hard_negative_mix_fraction=0.25,
        ),
    )

    result = sample_runtime_opponent_policy_ids(
        count=12,
        rng=np.random.default_rng(11),
        league_enabled=True,
        league_config=league_config,
        pfsp_ready=True,
        reference_update=0,
        mirror_weight=0.0,
        heuristic_public_weight=0.0,
        heuristic_public_variant_weight=0.0,
        noleague_baseline_weight=0.0,
        warmup_snapshot_weight=0.0,
        opponent_candidate_ids=("champ_a", "recent_a", "recent_b", "hard_a", "hard_b"),
        opponent_hard_negative_ids=("hard_a", "hard_b"),
        opponent_champion_ids=("champ_a",),
        opponent_recent_ids=("recent_a", "recent_b"),
        opponent_heuristic_policy_ids=(),
        opponent_model_ids=("champ_a", "recent_a", "recent_b", "hard_a", "hard_b"),
        outcomes=_SamplingOutcomes(),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control"),
        noleague_baseline_policy_id="baseline",
    )

    assert result.policy_ids == (
        "hard_b",
        "champ_a",
        "recent_b",
        "hard_a",
        "hard_a",
        "recent_b",
        "hard_b",
        "hard_b",
        "recent_b",
        "recent_b",
        "champ_a",
        "recent_b",
    )
    assert result.sampled_envs == 12
    assert result.champion_envs == 2
    assert result.recent_envs == 5
    assert result.hard_negative_envs == 5
    assert result.mirror_envs == 0
    assert dict(result.sampled_policy_envs) == {
        "champ_a": 2,
        "hard_a": 2,
        "hard_b": 3,
        "recent_b": 5,
    }
    assert dict(result.champion_policy_envs) == {"champ_a": 2}
    assert dict(result.hard_negative_policy_envs) == {"hard_a": 2, "hard_b": 3}
    assert dict(result.recent_policy_envs) == {"recent_b": 5}


def test_sample_runtime_opponent_policy_ids_weights_focused_hard_negatives() -> None:
    league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=1.0,
            hard_negative_focus_policy_ids=("hard_b",),
            hard_negative_focus_weight_multiplier=5.0,
        ),
    )

    result = sample_runtime_opponent_policy_ids(
        count=8,
        rng=np.random.default_rng(17),
        league_enabled=True,
        league_config=league_config,
        pfsp_ready=True,
        reference_update=0,
        mirror_weight=0.0,
        heuristic_public_weight=0.0,
        heuristic_public_variant_weight=0.0,
        noleague_baseline_weight=0.0,
        warmup_snapshot_weight=0.0,
        opponent_candidate_ids=("hard_a", "hard_b"),
        opponent_hard_negative_ids=("hard_a", "hard_b"),
        opponent_champion_ids=(),
        opponent_recent_ids=(),
        opponent_heuristic_policy_ids=(),
        opponent_model_ids=("hard_a", "hard_b"),
        outcomes=_SamplingOutcomes(),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control"),
        noleague_baseline_policy_id="baseline",
    )

    base_probabilities = np.asarray([((1.0 - 0.2) ** 2.0), ((1.0 - 0.4) ** 2.0)], dtype=np.float64)
    base_probabilities = base_probabilities / np.sum(base_probabilities)
    expected_probabilities = base_probabilities * np.array([1.0, 5.0], dtype=np.float64)
    expected_probabilities = expected_probabilities / np.sum(expected_probabilities)
    expected_rng = np.random.default_rng(17)
    expected_rng.choice(1, size=8, replace=True, p=np.array([1.0], dtype=np.float64))
    expected_indices = expected_rng.choice(2, size=8, replace=True, p=expected_probabilities)
    expected = tuple(("hard_a", "hard_b")[index] for index in expected_indices.tolist())

    assert result.policy_ids == expected
    assert result.hard_negative_envs == 8
    assert sum(dict(result.hard_negative_policy_envs).values()) == 8


def test_sample_runtime_opponent_policy_ids_applies_row_deficit_weights_to_champions() -> None:
    league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=1.0,
            hard_negative_mix_fraction=0.0,
            row_deficit_policy_weights=(("champ_b", 4.0),),
        ),
    )

    result = sample_runtime_opponent_policy_ids(
        count=8,
        rng=np.random.default_rng(19),
        league_enabled=True,
        league_config=league_config,
        pfsp_ready=True,
        reference_update=0,
        mirror_weight=0.0,
        heuristic_public_weight=0.0,
        heuristic_public_variant_weight=0.0,
        noleague_baseline_weight=0.0,
        warmup_snapshot_weight=0.0,
        opponent_candidate_ids=("champ_a", "champ_b"),
        opponent_hard_negative_ids=(),
        opponent_champion_ids=("champ_a", "champ_b"),
        opponent_recent_ids=(),
        opponent_heuristic_policy_ids=(),
        opponent_model_ids=("champ_a", "champ_b"),
        outcomes=_SamplingOutcomes(),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control"),
        noleague_baseline_policy_id="baseline",
    )

    base_probabilities = np.asarray([((1.0 - 0.5) ** 2.0), ((1.0 - 0.5) ** 2.0)], dtype=np.float64)
    base_probabilities = base_probabilities / np.sum(base_probabilities)
    expected_probabilities = base_probabilities * np.array([1.0, 4.0], dtype=np.float64)
    expected_probabilities = expected_probabilities / np.sum(expected_probabilities)
    expected_rng = np.random.default_rng(19)
    expected_rng.choice(1, size=8, replace=True, p=np.array([1.0], dtype=np.float64))
    expected_indices = expected_rng.choice(2, size=8, replace=True, p=expected_probabilities)
    expected = tuple(("champ_a", "champ_b")[index] for index in expected_indices.tolist())

    assert result.policy_ids == expected
    assert result.champion_envs == 8


def test_sample_runtime_opponent_policy_ids_preserves_pre_pfsp_mixed_bucket_accounting() -> None:
    league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        warmup=SimpleNamespace(first_updates=999),
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )

    result = sample_runtime_opponent_policy_ids(
        count=12,
        rng=np.random.default_rng(1),
        league_enabled=True,
        league_config=league_config,
        pfsp_ready=False,
        reference_update=0,
        mirror_weight=0.0,
        heuristic_public_weight=0.2,
        heuristic_public_variant_weight=0.2,
        noleague_baseline_weight=0.2,
        warmup_snapshot_weight=0.2,
        opponent_candidate_ids=("warm_a", "warm_b"),
        opponent_hard_negative_ids=(),
        opponent_champion_ids=(),
        opponent_recent_ids=(),
        opponent_heuristic_policy_ids=("heuristic", "aggro", "control"),
        opponent_model_ids=("baseline", "warm_a", "warm_b"),
        outcomes=_SamplingOutcomes(),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control"),
        noleague_baseline_policy_id="baseline",
    )

    assert result.policy_ids == (
        "baseline",
        "mirror",
        "heuristic",
        "mirror",
        "control",
        "baseline",
        "mirror",
        "baseline",
        "baseline",
        "heuristic",
        "warm_a",
        "baseline",
    )
    assert result.sampled_envs == 9
    assert result.mirror_envs == 3
    assert result.heuristic_public_envs == 2
    assert result.heuristic_public_variant_envs == 1
    assert result.noleague_baseline_envs == 5
    assert result.warmup_snapshot_envs == 1


def test_sample_runtime_opponent_policy_ids_supports_live_mirror_lane_after_pfsp_ready() -> None:
    league_config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=0.0,
            hard_negative_mix_fraction=0.0,
        ),
    )

    result = sample_runtime_opponent_policy_ids(
        count=200,
        rng=np.random.default_rng(11),
        league_enabled=True,
        league_config=league_config,
        pfsp_ready=True,
        reference_update=10,
        mirror_weight=0.4,
        heuristic_public_weight=0.2,
        heuristic_public_variant_weight=0.0,
        noleague_baseline_weight=0.0,
        warmup_snapshot_weight=0.0,
        opponent_candidate_ids=("recent_a", "recent_b"),
        opponent_hard_negative_ids=(),
        opponent_champion_ids=(),
        opponent_recent_ids=("recent_a", "recent_b"),
        opponent_heuristic_policy_ids=("heuristic",),
        opponent_model_ids=("recent_a", "recent_b"),
        outcomes=_SamplingOutcomes(),
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control"),
        noleague_baseline_policy_id="baseline",
    )

    assert len(result.policy_ids) == 200
    assert result.mirror_envs > 0
    assert result.heuristic_public_envs > 0
    assert result.recent_envs > 0
    assert result.warmup_snapshot_envs == 0
    assert result.sampled_envs == result.heuristic_public_envs + result.recent_envs


def test_sample_warmup_snapshot_policy_ids_preserves_pfsp_counters() -> None:
    league_config = SimpleNamespace(pfsp_power=2.0, pfsp_epsilon_uniform=0.0)

    result = sample_warmup_snapshot_policy_ids(
        count=6,
        rng=np.random.default_rng(3),
        opponent_candidate_ids=("warm_a", "warm_b"),
        league_config=league_config,
        outcomes=_SamplingOutcomes(),
    )

    assert result.policy_ids == ("warm_a", "warm_a", "warm_a", "warm_a", "warm_a", "warm_a")
    assert result.sampled_envs == 6
    assert result.warmup_snapshot_envs == 6
    assert result.mirror_envs == 0
    assert dict(result.sampled_policy_envs) == {"warm_a": 6}
    assert dict(result.warmup_snapshot_policy_envs) == {"warm_a": 6}
    assert (
        select_hard_negative_ids(
            candidate_ids=("policy",),
            league_config=SimpleNamespace(),
            outcomes=None,
            registry_path=None,
        )
        == ()
    )
