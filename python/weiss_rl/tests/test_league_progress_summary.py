from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.experiments.league_progress_summary import build_league_progress_summary, slugify_opponent


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_slugify_opponent_matches_collector_metric_names() -> None:
    assert slugify_opponent("B1 NoLeague baseline") == "b1_noleague_baseline"
    assert slugify_opponent("seed_c3aac2f9dc_policy_000004") == "seed_c3aac2f9dc_policy_000004"


def test_league_progress_summary_groups_fixed_and_learned_rows(tmp_path: Path) -> None:
    scalars = tmp_path / "scalars.jsonl"
    _write_jsonl(
        scalars,
        [
            {
                "update_count": 1,
                "pfsp_champion_envs": 2,
                "pfsp_hard_negative_envs": 0,
                "pfsp_champion_pool_size": 8,
                "collector_pfsp_champion_policy_envs__seed_champ_policy_000001": 2,
                "collector_pfsp_sampled_policy_envs__seed_champ_policy_000001": 2,
                "collector_outcome_vs_b1_noleague_baseline_games": 10,
                "collector_outcome_vs_b1_noleague_baseline_wins": 6,
                "collector_outcome_vs_b1_noleague_baseline_losses": 4,
                "collector_outcome_vs_b1_noleague_baseline_draws": 0,
                "collector_outcome_vs_b1_noleague_baseline_timeouts": 0,
                "collector_outcome_vs_b1_noleague_baseline_win_rate": 0.6,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_games": 5,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_wins": 2,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_losses": 3,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_draws": 0,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_timeouts": 0,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_win_rate": 0.4,
            },
            {
                "update_count": 2,
                "pfsp_champion_envs": 4,
                "pfsp_hard_negative_envs": 3,
                "pfsp_champion_pool_size": 8,
                "pfsp_hard_negative_pool_size": 1,
                "collector_pfsp_champion_policy_envs__seed_champ_policy_000001": 1,
                "collector_pfsp_champion_policy_envs__seed_champ_policy_000002": 3,
                "collector_pfsp_hard_negative_policy_envs__seed_hard_policy_000003": 3,
                "collector_pfsp_sampled_policy_envs__seed_champ_policy_000001": 1,
                "collector_pfsp_sampled_policy_envs__seed_champ_policy_000002": 3,
                "collector_pfsp_sampled_policy_envs__seed_hard_policy_000003": 3,
                "collector_outcome_vs_b1_noleague_baseline_games": 10,
                "collector_outcome_vs_b1_noleague_baseline_wins": 7,
                "collector_outcome_vs_b1_noleague_baseline_losses": 3,
                "collector_outcome_vs_b1_noleague_baseline_draws": 0,
                "collector_outcome_vs_b1_noleague_baseline_timeouts": 0,
                "collector_outcome_vs_b1_noleague_baseline_win_rate": 0.7,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_games": 5,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_wins": 3,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_losses": 2,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_draws": 0,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_timeouts": 0,
                "collector_outcome_vs_seed_c3aac2f9dc_policy_000004_win_rate": 0.6,
            },
        ],
    )

    summary = build_league_progress_summary(
        scalars_jsonl=scalars,
        learned_opponents=["seed_c3aac2f9dc_policy_000004"],
        notes="unit test",
    )

    assert summary["evidence_grade"] == "diagnostic_unpaired_training_collector"
    assert summary["exposure_totals"]["pfsp_champion_envs"] == 6.0
    assert summary["exposure_totals"]["pfsp_hard_negative_envs"] == 3.0
    assert summary["policy_exposure_totals"]["champion"] == {
        "seed_champ_policy_000001": 3.0,
        "seed_champ_policy_000002": 3.0,
    }
    assert summary["policy_exposure_totals"]["hard_negative"] == {"seed_hard_policy_000003": 3.0}
    assert summary["policy_exposure_max"]["sampled"]["seed_champ_policy_000001"] == 2.0
    assert summary["pool_size_last"]["pfsp_hard_negative_pool_size"] == 1.0
    assert summary["groups"]["fixed_baseline"]["delta_last_minus_first"] == 0.09999999999999998
    assert summary["groups"]["imported_learned"]["delta_last_minus_first"] == 0.19999999999999996
    learned = next(item for item in summary["opponents"] if item["opponent_slug"] == "seed_c3aac2f9dc_policy_000004")
    assert learned["group"] == "imported_learned"
    assert learned["total"]["wins"] == 5
    assert learned["total"]["games"] == 10


def test_league_progress_summary_classifies_seed_wrapped_hard_negatives(tmp_path: Path) -> None:
    scalars = tmp_path / "scalars.jsonl"
    _write_jsonl(
        scalars,
        [
            {
                "update_count": 1,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_games": 4,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_wins": 1,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_losses": 3,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_draws": 0,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_timeouts": 0,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_win_rate": 0.25,
            },
            {
                "update_count": 2,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_games": 4,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_wins": 3,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_losses": 1,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_draws": 0,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_timeouts": 0,
                "collector_outcome_vs_seed_outer_seed_inner_policy_000002_win_rate": 0.75,
            },
        ],
    )

    summary = build_league_progress_summary(
        scalars_jsonl=scalars,
        learned_opponents=["seed_inner_policy_000002"],
        hard_negative_opponents=["seed_inner_policy_000002"],
    )

    hard_negative = summary["groups"]["hard_negative"]
    assert hard_negative["opponent_count"] == 1
    assert hard_negative["delta_last_minus_first"] == 0.5
    opponent = summary["opponents"][0]
    assert opponent["opponent_slug"] == "seed_outer_seed_inner_policy_000002"
    assert opponent["group"] == "hard_negative"
