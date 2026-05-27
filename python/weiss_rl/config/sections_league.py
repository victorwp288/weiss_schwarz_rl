"""League stack config section parser."""

from __future__ import annotations

from typing import Any

from .models import (
    LeagueConfig,
    LeaguePoolConfig,
    LeaguePromotionConfig,
    LeagueSamplingConfig,
    LeagueWarmupConfig,
    PromotionAnchorSetConfig,
    PromotionGateConfig,
    PromotionGateGuardrailsConfig,
)
from .parsing_utils import (
    reject_unknown_keys,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_str_list,
    require_text,
)


def parse_league_config(body: dict[str, Any]) -> LeagueConfig:
    reject_unknown_keys(body, allowed={"enabled", "pool", "sampling", "warmup", "promotion"}, context="league")
    pool = require_mapping(body["pool"], context="league.pool")
    sampling = require_mapping(body["sampling"], context="league.sampling")
    warmup = require_mapping(body["warmup"], context="league.warmup")
    promotion = require_mapping(body["promotion"], context="league.promotion")
    anchor_set = require_mapping(promotion["anchor_set_v1"], context="league.promotion.anchor_set_v1")
    gate = require_mapping(promotion["gate"], context="league.promotion.gate")
    guardrails = require_mapping(gate["guardrails"], context="league.promotion.gate.guardrails")

    reject_unknown_keys(
        pool,
        allowed={
            "recent_size",
            "champion_size",
            "champion_max_age_updates",
            "seed_snapshot_champion_import",
            "seed_snapshot_import_filter",
            "seed_snapshot_registry_json",
        },
        context="league.pool",
    )
    reject_unknown_keys(
        sampling,
        allowed={
            "opponent_sampling",
            "pfsp_power",
            "pfsp_epsilon_uniform",
            "pfsp_stats_source",
            "pfsp_window_episodes",
            "mirror_mix_fraction",
            "mirror_mix_end_updates",
            "mirror_final_mix_fraction",
            "heuristic_public_start_updates",
            "heuristic_public_mix_fraction",
            "heuristic_public_mix_end_updates",
            "heuristic_public_final_mix_fraction",
            "heuristic_public_variant_mix_fraction",
            "heuristic_public_variant_mix_end_updates",
            "heuristic_public_variant_final_mix_fraction",
            "noleague_baseline_mix_fraction",
            "noleague_baseline_mix_end_updates",
            "warmup_snapshot_mix_fraction",
            "heuristic_public_reserved_envs_per_actor",
            "noleague_baseline_reserved_envs_per_actor",
            "champion_mix_fraction",
            "hard_negative_mix_fraction",
            "hard_negative_min_samples",
            "hard_negative_max_win_rate",
            "hard_negative_focus_policy_ids",
            "hard_negative_focus_weight_multiplier",
            "row_deficit_policy_weights",
            "hard_negative_overlaps_champions",
        },
        context="league.sampling",
    )
    reject_unknown_keys(
        warmup,
        allowed={"first_updates", "initial_window_episodes", "ramp_target_updates", "ramp_target_window_episodes"},
        context="league.warmup",
    )
    reject_unknown_keys(
        promotion,
        allowed={"enabled", "paired_seeds", "threshold", "anchor_set_v1", "seed_file", "gate"},
        context="league.promotion",
    )
    reject_unknown_keys(
        anchor_set, allowed={"required", "optional_if_available"}, context="league.promotion.anchor_set_v1"
    )
    reject_unknown_keys(
        gate,
        allowed={"uncertainty_method", "weighting", "seat_swap", "folding", "guardrails", "record_file"},
        context="league.promotion.gate",
    )
    reject_unknown_keys(
        guardrails,
        allowed={"max_prob_anchor_loss_below_0_45", "max_truncation_rate"},
        context="league.promotion.gate.guardrails",
    )

    pfsp_stats_source = require_text(
        sampling["pfsp_stats_source"],
        field_name="league.sampling.pfsp_stats_source",
    )
    if pfsp_stats_source != "online_outcomes":
        raise ValueError("league.sampling.pfsp_stats_source currently only supports 'online_outcomes'")

    seed_snapshot_champion_import = require_text(
        pool.get("seed_snapshot_champion_import", "source_champions"),
        field_name="league.pool.seed_snapshot_champion_import",
    )
    if seed_snapshot_champion_import not in {"source_champions", "pinned", "all", "none"}:
        raise ValueError(
            "league.pool.seed_snapshot_champion_import must be one of: all, none, pinned, source_champions"
        )
    seed_snapshot_import_filter = require_text(
        pool.get("seed_snapshot_import_filter", "all"),
        field_name="league.pool.seed_snapshot_import_filter",
    )
    if seed_snapshot_import_filter not in {"all", "none", "pinned", "source_champions", "pinned_or_source_champions"}:
        raise ValueError(
            "league.pool.seed_snapshot_import_filter must be one of: "
            "all, none, pinned, pinned_or_source_champions, source_champions"
        )
    raw_seed_snapshot_registry_json = pool.get("seed_snapshot_registry_json", "")
    if raw_seed_snapshot_registry_json is None:
        seed_snapshot_registry_json = ""
    elif isinstance(raw_seed_snapshot_registry_json, str):
        seed_snapshot_registry_json = raw_seed_snapshot_registry_json.strip()
    else:
        raise ValueError("league.pool.seed_snapshot_registry_json must be a string when provided")
    hard_negative_focus_weight_multiplier = require_float(
        sampling.get("hard_negative_focus_weight_multiplier", 1.0),
        field_name="league.sampling.hard_negative_focus_weight_multiplier",
    )
    if hard_negative_focus_weight_multiplier <= 0.0:
        raise ValueError("league.sampling.hard_negative_focus_weight_multiplier must be > 0")
    row_deficit_policy_weights = _parse_policy_weight_map(
        sampling.get("row_deficit_policy_weights", {}),
        field_name="league.sampling.row_deficit_policy_weights",
    )

    return LeagueConfig(
        enabled=require_bool(body["enabled"], field_name="league.enabled"),
        pool=LeaguePoolConfig(
            recent_size=require_int(pool["recent_size"], field_name="league.pool.recent_size", minimum=1),
            champion_size=require_int(pool["champion_size"], field_name="league.pool.champion_size", minimum=0),
            champion_max_age_updates=require_int(
                pool.get("champion_max_age_updates", 0),
                field_name="league.pool.champion_max_age_updates",
                minimum=0,
            ),
            seed_snapshot_champion_import=seed_snapshot_champion_import,
            seed_snapshot_import_filter=seed_snapshot_import_filter,
            seed_snapshot_registry_json=seed_snapshot_registry_json,
        ),
        sampling=LeagueSamplingConfig(
            opponent_sampling=require_text(
                sampling["opponent_sampling"], field_name="league.sampling.opponent_sampling"
            ),
            pfsp_power=require_float(sampling["pfsp_power"], field_name="league.sampling.pfsp_power"),
            pfsp_epsilon_uniform=require_float(
                sampling["pfsp_epsilon_uniform"],
                field_name="league.sampling.pfsp_epsilon_uniform",
            ),
            pfsp_stats_source=pfsp_stats_source,
            pfsp_window_episodes=require_int(
                sampling["pfsp_window_episodes"],
                field_name="league.sampling.pfsp_window_episodes",
                minimum=1,
            ),
            mirror_mix_fraction=require_float(
                sampling.get("mirror_mix_fraction", 0.0),
                field_name="league.sampling.mirror_mix_fraction",
            ),
            mirror_mix_end_updates=require_int(
                sampling.get("mirror_mix_end_updates", -1),
                field_name="league.sampling.mirror_mix_end_updates",
                minimum=-1,
            ),
            mirror_final_mix_fraction=require_float(
                sampling.get("mirror_final_mix_fraction", sampling.get("mirror_mix_fraction", 0.0)),
                field_name="league.sampling.mirror_final_mix_fraction",
            ),
            heuristic_public_start_updates=require_int(
                sampling.get("heuristic_public_start_updates", 0),
                field_name="league.sampling.heuristic_public_start_updates",
                minimum=0,
            ),
            heuristic_public_mix_fraction=require_float(
                sampling.get("heuristic_public_mix_fraction", 0.0),
                field_name="league.sampling.heuristic_public_mix_fraction",
            ),
            heuristic_public_mix_end_updates=require_int(
                sampling.get("heuristic_public_mix_end_updates", -1),
                field_name="league.sampling.heuristic_public_mix_end_updates",
                minimum=-1,
            ),
            heuristic_public_final_mix_fraction=require_float(
                sampling.get(
                    "heuristic_public_final_mix_fraction",
                    sampling.get("heuristic_public_mix_fraction", 0.0),
                ),
                field_name="league.sampling.heuristic_public_final_mix_fraction",
            ),
            heuristic_public_variant_mix_fraction=require_float(
                sampling.get("heuristic_public_variant_mix_fraction", 0.0),
                field_name="league.sampling.heuristic_public_variant_mix_fraction",
            ),
            heuristic_public_variant_mix_end_updates=require_int(
                sampling.get("heuristic_public_variant_mix_end_updates", -1),
                field_name="league.sampling.heuristic_public_variant_mix_end_updates",
                minimum=-1,
            ),
            heuristic_public_variant_final_mix_fraction=require_float(
                sampling.get(
                    "heuristic_public_variant_final_mix_fraction",
                    sampling.get("heuristic_public_variant_mix_fraction", 0.0),
                ),
                field_name="league.sampling.heuristic_public_variant_final_mix_fraction",
            ),
            noleague_baseline_mix_fraction=require_float(
                sampling.get("noleague_baseline_mix_fraction", 0.0),
                field_name="league.sampling.noleague_baseline_mix_fraction",
            ),
            noleague_baseline_mix_end_updates=require_int(
                sampling.get("noleague_baseline_mix_end_updates", -1),
                field_name="league.sampling.noleague_baseline_mix_end_updates",
                minimum=-1,
            ),
            warmup_snapshot_mix_fraction=require_float(
                sampling.get("warmup_snapshot_mix_fraction", 0.0),
                field_name="league.sampling.warmup_snapshot_mix_fraction",
            ),
            heuristic_public_reserved_envs_per_actor=require_int(
                sampling.get("heuristic_public_reserved_envs_per_actor", 0),
                field_name="league.sampling.heuristic_public_reserved_envs_per_actor",
                minimum=0,
            ),
            noleague_baseline_reserved_envs_per_actor=require_int(
                sampling.get("noleague_baseline_reserved_envs_per_actor", 0),
                field_name="league.sampling.noleague_baseline_reserved_envs_per_actor",
                minimum=0,
            ),
            champion_mix_fraction=require_float(
                sampling.get("champion_mix_fraction", 0.35),
                field_name="league.sampling.champion_mix_fraction",
            ),
            hard_negative_mix_fraction=require_float(
                sampling.get("hard_negative_mix_fraction", 0.2),
                field_name="league.sampling.hard_negative_mix_fraction",
            ),
            hard_negative_min_samples=require_int(
                sampling.get("hard_negative_min_samples", 16),
                field_name="league.sampling.hard_negative_min_samples",
                minimum=1,
            ),
            hard_negative_max_win_rate=require_float(
                sampling.get("hard_negative_max_win_rate", 0.45),
                field_name="league.sampling.hard_negative_max_win_rate",
            ),
            hard_negative_focus_policy_ids=require_str_list(
                sampling.get("hard_negative_focus_policy_ids", []),
                field_name="league.sampling.hard_negative_focus_policy_ids",
            ),
            hard_negative_focus_weight_multiplier=hard_negative_focus_weight_multiplier,
            row_deficit_policy_weights=row_deficit_policy_weights,
            hard_negative_overlaps_champions=require_bool(
                sampling.get("hard_negative_overlaps_champions", False),
                field_name="league.sampling.hard_negative_overlaps_champions",
            ),
        ),
        warmup=LeagueWarmupConfig(
            first_updates=require_int(warmup["first_updates"], field_name="league.warmup.first_updates", minimum=0),
            initial_window_episodes=require_int(
                warmup["initial_window_episodes"],
                field_name="league.warmup.initial_window_episodes",
                minimum=0,
            ),
            ramp_target_updates=require_int(
                warmup["ramp_target_updates"],
                field_name="league.warmup.ramp_target_updates",
                minimum=0,
            ),
            ramp_target_window_episodes=require_int(
                warmup["ramp_target_window_episodes"],
                field_name="league.warmup.ramp_target_window_episodes",
                minimum=0,
            ),
        ),
        promotion=LeaguePromotionConfig(
            enabled=require_bool(promotion["enabled"], field_name="league.promotion.enabled"),
            paired_seeds=require_int(promotion["paired_seeds"], field_name="league.promotion.paired_seeds", minimum=1),
            threshold=require_text(promotion["threshold"], field_name="league.promotion.threshold"),
            anchor_set_v1=PromotionAnchorSetConfig(
                required=require_str_list(anchor_set["required"], field_name="league.promotion.anchor_set_v1.required"),
                optional_if_available=require_str_list(
                    anchor_set["optional_if_available"],
                    field_name="league.promotion.anchor_set_v1.optional_if_available",
                ),
            ),
            seed_file=require_text(promotion["seed_file"], field_name="league.promotion.seed_file"),
            gate=PromotionGateConfig(
                uncertainty_method=require_text(
                    gate["uncertainty_method"],
                    field_name="league.promotion.gate.uncertainty_method",
                ),
                weighting=require_text(gate["weighting"], field_name="league.promotion.gate.weighting"),
                seat_swap=require_bool(gate["seat_swap"], field_name="league.promotion.gate.seat_swap"),
                folding=require_text(gate["folding"], field_name="league.promotion.gate.folding"),
                guardrails=PromotionGateGuardrailsConfig(
                    max_prob_anchor_loss_below_0_45=require_float(
                        guardrails["max_prob_anchor_loss_below_0_45"],
                        field_name="league.promotion.gate.guardrails.max_prob_anchor_loss_below_0_45",
                    ),
                    max_truncation_rate=require_float(
                        guardrails["max_truncation_rate"],
                        field_name="league.promotion.gate.guardrails.max_truncation_rate",
                    ),
                ),
                record_file=require_text(gate["record_file"], field_name="league.promotion.gate.record_file"),
            ),
        ),
    )


def _parse_policy_weight_map(raw: Any, *, field_name: str) -> tuple[tuple[str, float], ...]:
    if raw is None:
        return ()
    if isinstance(raw, list | tuple):
        mapping = {}
        for index, item in enumerate(raw):
            if not isinstance(item, list | tuple) or len(item) != 2:
                raise ValueError(f"{field_name}[{index}] must be a [policy_id, weight] pair")
            key, value = item
            mapping[key] = value
    else:
        mapping = require_mapping(raw, context=field_name)
    parsed: dict[str, float] = {}
    for raw_key, raw_value in mapping.items():
        key = require_text(raw_key, field_name=f"{field_name} key").strip()
        if not key:
            raise ValueError(f"{field_name} keys must be non-empty")
        weight = require_float(raw_value, field_name=f"{field_name}.{key}")
        if weight <= 0.0:
            raise ValueError(f"{field_name}.{key} must be > 0")
        parsed[key] = weight
    return tuple(sorted(parsed.items()))
