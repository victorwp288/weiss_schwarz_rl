from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from weiss_rl.config.sections_league import parse_league_config


def _copy_section(body: dict[str, object], key: str) -> dict[str, object]:
    return dict(cast(Mapping[str, object], body[key]))


def _league_body() -> dict[str, object]:
    return {
        "enabled": True,
        "pool": {"recent_size": 4, "champion_size": 2},
        "sampling": {
            "opponent_sampling": "pfsp",
            "pfsp_power": 1.0,
            "pfsp_epsilon_uniform": 0.05,
            "pfsp_stats_source": "online_outcomes",
            "pfsp_window_episodes": 32,
        },
        "warmup": {
            "first_updates": 10,
            "initial_window_episodes": 20,
            "ramp_target_updates": 30,
            "ramp_target_window_episodes": 40,
        },
        "promotion": {
            "enabled": True,
            "paired_seeds": 8,
            "threshold": "prob_gt_half",
            "anchor_set_v1": {"required": ["b0"], "optional_if_available": ["b1"]},
            "seed_file": "configs/seeds/promotion.txt",
            "gate": {
                "uncertainty_method": "wilson",
                "weighting": "uniform",
                "seat_swap": True,
                "folding": "mean",
                "guardrails": {
                    "max_prob_anchor_loss_below_0_45": 0.1,
                    "max_truncation_rate": 0.2,
                },
                "record_file": "promotion.jsonl",
            },
        },
    }


def test_parse_league_config_applies_existing_sampling_defaults() -> None:
    config = parse_league_config(_league_body())

    assert config.enabled is True
    assert config.pool.recent_size == 4
    assert config.pool.champion_size == 2
    assert config.pool.champion_max_age_updates == 0
    assert config.pool.seed_snapshot_import_filter == "all"
    assert config.pool.seed_snapshot_registry_json == ""
    assert config.sampling.heuristic_public_start_updates == 0
    assert config.sampling.heuristic_public_mix_fraction == pytest.approx(0.0)
    assert config.sampling.heuristic_public_mix_end_updates == -1
    assert config.sampling.heuristic_public_final_mix_fraction == pytest.approx(0.0)
    assert config.sampling.heuristic_public_variant_mix_fraction == pytest.approx(0.0)
    assert config.sampling.heuristic_public_variant_mix_end_updates == -1
    assert config.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)
    assert config.sampling.noleague_baseline_mix_end_updates == -1
    assert config.sampling.warmup_snapshot_mix_fraction == pytest.approx(0.0)
    assert config.sampling.champion_mix_fraction == pytest.approx(0.35)
    assert config.sampling.hard_negative_mix_fraction == pytest.approx(0.2)
    assert config.sampling.hard_negative_min_samples == 16
    assert config.sampling.hard_negative_max_win_rate == pytest.approx(0.45)
    assert config.sampling.hard_negative_focus_policy_ids == ()
    assert config.sampling.hard_negative_focus_weight_multiplier == pytest.approx(1.0)
    assert config.sampling.row_deficit_policy_weights == ()
    assert config.sampling.hard_negative_overlaps_champions is False
    assert config.promotion.anchor_set_v1.required == ("b0",)
    assert config.promotion.anchor_set_v1.optional_if_available == ("b1",)


def test_parse_league_config_uses_mix_fraction_as_final_default() -> None:
    body = _league_body()
    sampling = _copy_section(body, "sampling")
    sampling["heuristic_public_mix_fraction"] = 0.25
    sampling["heuristic_public_variant_mix_fraction"] = 0.125
    body["sampling"] = sampling

    config = parse_league_config(body)

    assert config.sampling.heuristic_public_final_mix_fraction == pytest.approx(0.25)
    assert config.sampling.heuristic_public_variant_final_mix_fraction == pytest.approx(0.125)


def test_parse_league_config_accepts_optional_sampling_values() -> None:
    body = _league_body()
    pool = _copy_section(body, "pool")
    pool["champion_max_age_updates"] = 100
    pool["seed_snapshot_champion_import"] = "pinned"
    pool["seed_snapshot_import_filter"] = "pinned"
    pool["seed_snapshot_registry_json"] = "runs/source/training/snapshots/registry_augmented.json"
    body["pool"] = pool
    sampling = _copy_section(body, "sampling")
    sampling.update(
        {
            "heuristic_public_reserved_envs_per_actor": 1,
            "noleague_baseline_reserved_envs_per_actor": 2,
            "noleague_baseline_mix_fraction": 0.3,
            "noleague_baseline_mix_end_updates": 50,
            "warmup_snapshot_mix_fraction": 0.4,
            "champion_mix_fraction": 0.5,
            "hard_negative_mix_fraction": 0.6,
            "hard_negative_min_samples": 7,
            "hard_negative_max_win_rate": 0.8,
            "hard_negative_focus_policy_ids": ["seed_source_policy_000002"],
            "hard_negative_focus_weight_multiplier": 4.0,
            "row_deficit_policy_weights": {
                "seed_source_policy_000004": 2.5,
                "seed_source_policy_000001": 1.5,
            },
            "hard_negative_overlaps_champions": True,
        }
    )
    body["sampling"] = sampling

    config = parse_league_config(body)

    assert config.pool.champion_max_age_updates == 100
    assert config.pool.seed_snapshot_champion_import == "pinned"
    assert config.pool.seed_snapshot_import_filter == "pinned"
    assert config.pool.seed_snapshot_registry_json == "runs/source/training/snapshots/registry_augmented.json"
    assert config.sampling.heuristic_public_reserved_envs_per_actor == 1
    assert config.sampling.noleague_baseline_reserved_envs_per_actor == 2
    assert config.sampling.noleague_baseline_mix_fraction == pytest.approx(0.3)
    assert config.sampling.hard_negative_min_samples == 7
    assert config.sampling.hard_negative_focus_policy_ids == ("seed_source_policy_000002",)
    assert config.sampling.hard_negative_focus_weight_multiplier == pytest.approx(4.0)
    assert config.sampling.row_deficit_policy_weights == (
        ("seed_source_policy_000001", pytest.approx(1.5)),
        ("seed_source_policy_000004", pytest.approx(2.5)),
    )
    assert config.sampling.hard_negative_overlaps_champions is True


def test_parse_league_config_preserves_validation_errors() -> None:
    unknown = _league_body()
    unknown["extra"] = True
    with pytest.raises(ValueError, match="league has unsupported keys: extra"):
        parse_league_config(unknown)

    bad_sampling = _league_body()
    sampling = _copy_section(bad_sampling, "sampling")
    sampling["extra"] = True
    bad_sampling["sampling"] = sampling
    with pytest.raises(ValueError, match="league.sampling has unsupported keys: extra"):
        parse_league_config(bad_sampling)

    bad_pfsp_source = _league_body()
    sampling = _copy_section(bad_pfsp_source, "sampling")
    sampling["pfsp_stats_source"] = "file"
    bad_pfsp_source["sampling"] = sampling
    with pytest.raises(ValueError, match="league.sampling.pfsp_stats_source currently only supports 'online_outcomes'"):
        parse_league_config(bad_pfsp_source)

    bad_minimum = _league_body()
    sampling = _copy_section(bad_minimum, "sampling")
    sampling["hard_negative_min_samples"] = 0
    bad_minimum["sampling"] = sampling
    with pytest.raises(ValueError, match="league.sampling.hard_negative_min_samples must be >= 1, got 0"):
        parse_league_config(bad_minimum)

    bad_focus_multiplier = _league_body()
    sampling = _copy_section(bad_focus_multiplier, "sampling")
    sampling["hard_negative_focus_weight_multiplier"] = 0.0
    bad_focus_multiplier["sampling"] = sampling
    with pytest.raises(ValueError, match="league.sampling.hard_negative_focus_weight_multiplier must be > 0"):
        parse_league_config(bad_focus_multiplier)

    bad_row_deficit_weight = _league_body()
    sampling = _copy_section(bad_row_deficit_weight, "sampling")
    sampling["row_deficit_policy_weights"] = {"policy_000004": 0.0}
    bad_row_deficit_weight["sampling"] = sampling
    with pytest.raises(ValueError, match="league.sampling.row_deficit_policy_weights.policy_000004 must be > 0"):
        parse_league_config(bad_row_deficit_weight)

    bad_seed_import = _league_body()
    pool = _copy_section(bad_seed_import, "pool")
    pool["seed_snapshot_champion_import"] = "latest"
    bad_seed_import["pool"] = pool
    with pytest.raises(
        ValueError,
        match="league.pool.seed_snapshot_champion_import must be one of: all, none, pinned, source_champions",
    ):
        parse_league_config(bad_seed_import)

    bad_seed_filter = _league_body()
    pool = _copy_section(bad_seed_filter, "pool")
    pool["seed_snapshot_import_filter"] = "latest"
    bad_seed_filter["pool"] = pool
    with pytest.raises(
        ValueError,
        match="league.pool.seed_snapshot_import_filter must be one of: all, none, pinned",
    ):
        parse_league_config(bad_seed_filter)
