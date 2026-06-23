from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.league.registry import SnapshotRegistry, snapshot_weights_relpath
from weiss_rl.runtime.components.opponents import (
    configured_hard_negative_focus_policy_ids,
    configured_row_deficit_policy_weights,
    hard_negative_focus_policy_id_matches,
    hard_negative_focus_weight_multipliers,
    row_deficit_weight_multipliers,
    select_hard_negative_ids,
)


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
    assert (
        select_hard_negative_ids(
            candidate_ids=("policy",),
            league_config=SimpleNamespace(),
            outcomes=None,
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
