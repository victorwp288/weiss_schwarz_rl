"""Opponent-pool bookkeeping helpers for the queue runtime."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from weiss_rl.league.opponent_pool import sample_opponent_snapshot_ids
from weiss_rl.runtime.components.opponent_policies import mix as _opponent_mix
from weiss_rl.runtime.components.opponent_policies import pool_selection as _opponent_pool_selection
from weiss_rl.runtime.components.opponent_policies.sampling import (
    OpponentSamplingAccumulator,
    OpponentSamplingResult,
    RuntimeOpponentGroup,
    RuntimeOpponentSamplingPlan,
    count_items,
    empty_opponent_sampling_result,
)

active_actor_heuristic_fraction = _opponent_mix.active_actor_heuristic_fraction
active_annealed_mix_fraction = _opponent_mix.active_annealed_mix_fraction
active_heuristic_public_mix_fraction = _opponent_mix.active_heuristic_public_mix_fraction
active_heuristic_public_variant_mix_fraction = _opponent_mix.active_heuristic_public_variant_mix_fraction
active_mirror_mix_fraction = _opponent_mix.active_mirror_mix_fraction
active_noleague_baseline_mix_fraction = _opponent_mix.active_noleague_baseline_mix_fraction
active_warmup_snapshot_mix_fraction = _opponent_mix.active_warmup_snapshot_mix_fraction

active_assigned_opponent_policy_ids = _opponent_pool_selection.active_assigned_opponent_policy_ids
apply_opponent_pool_diversity_floor = _opponent_pool_selection.apply_opponent_pool_diversity_floor
combine_weight_multipliers = _opponent_pool_selection.combine_weight_multipliers
configured_fixed_opponent_policy_ids = _opponent_pool_selection.configured_fixed_opponent_policy_ids
configured_hard_negative_focus_policy_ids = _opponent_pool_selection.configured_hard_negative_focus_policy_ids
configured_resident_opponent_policy_ids = _opponent_pool_selection.configured_resident_opponent_policy_ids
configured_row_deficit_policy_ids = _opponent_pool_selection.configured_row_deficit_policy_ids
configured_row_deficit_policy_weights = _opponent_pool_selection.configured_row_deficit_policy_weights
filter_timeout_heavy_opponents = _opponent_pool_selection.filter_timeout_heavy_opponents
fixed_opponent_policy_is_active = _opponent_pool_selection.fixed_opponent_policy_is_active
fixed_opponent_policy_slots = _opponent_pool_selection.fixed_opponent_policy_slots
hard_negative_focus_policy_id_matches = _opponent_pool_selection.hard_negative_focus_policy_id_matches
hard_negative_focus_weight_multipliers = _opponent_pool_selection.hard_negative_focus_weight_multipliers
promotion_gated_recent_reservoir_size = _opponent_pool_selection.promotion_gated_recent_reservoir_size
row_deficit_weight_multipliers = _opponent_pool_selection.row_deficit_weight_multipliers
select_hard_negative_ids = _opponent_pool_selection.select_hard_negative_ids


def build_runtime_opponent_sampling_groups(
    *,
    league_config: Any,
    pfsp_ready: bool,
    reference_update: int,
    mirror_weight: float,
    heuristic_public_weight: float,
    heuristic_public_variant_weight: float,
    noleague_baseline_weight: float,
    warmup_snapshot_weight: float,
    opponent_candidate_ids: Sequence[str],
    opponent_hard_negative_ids: Sequence[str],
    opponent_champion_ids: Sequence[str],
    opponent_recent_ids: Sequence[str],
    opponent_heuristic_policy_ids: Sequence[str],
    opponent_model_ids: Sequence[str],
    mirror_policy_id: str,
    heuristic_public_policy_id: str,
    heuristic_public_variant_policy_ids: Sequence[str],
    noleague_baseline_policy_id: str,
) -> tuple[RuntimeOpponentGroup, ...]:
    sampling_cfg = getattr(league_config, "sampling", league_config)
    heuristic_public_start_updates = max(
        0,
        int(getattr(sampling_cfg, "heuristic_public_start_updates", 0)),
    )
    heuristic_ids = set(str(policy_id) for policy_id in opponent_heuristic_policy_ids)
    model_ids = set(str(policy_id) for policy_id in opponent_model_ids)
    candidate_ids = tuple(str(policy_id) for policy_id in opponent_candidate_ids)
    hard_negative_ids = tuple(str(policy_id) for policy_id in opponent_hard_negative_ids)
    champion_ids = tuple(str(policy_id) for policy_id in opponent_champion_ids)
    recent_ids = tuple(str(policy_id) for policy_id in opponent_recent_ids)
    heuristic_public_weight = float(heuristic_public_weight)
    heuristic_public_variant_weight = float(heuristic_public_variant_weight)
    noleague_baseline_weight = float(noleague_baseline_weight)
    warmup_snapshot_weight = float(warmup_snapshot_weight)
    mirror_weight = max(0.0, float(mirror_weight)) if pfsp_ready else 0.0
    champion_weight = max(0.0, float(getattr(sampling_cfg, "champion_mix_fraction", 0.35)))
    hard_negative_weight = max(0.0, float(getattr(sampling_cfg, "hard_negative_mix_fraction", 0.2)))
    recent_weight = max(
        0.0,
        1.0
        - heuristic_public_weight
        - heuristic_public_variant_weight
        - noleague_baseline_weight
        - mirror_weight
        - champion_weight
        - hard_negative_weight,
    )

    groups: list[RuntimeOpponentGroup] = []
    if (
        heuristic_public_weight > 0.0
        and int(reference_update) >= heuristic_public_start_updates
        and str(heuristic_public_policy_id) in heuristic_ids
    ):
        groups.append(
            RuntimeOpponentGroup(
                name="heuristic_public",
                policy_ids=(str(heuristic_public_policy_id),),
                weight=heuristic_public_weight,
            )
        )
    heuristic_variant_policy_ids = tuple(
        str(policy_id) for policy_id in heuristic_public_variant_policy_ids if str(policy_id) in heuristic_ids
    )
    if (
        heuristic_public_variant_weight > 0.0
        and int(reference_update) >= heuristic_public_start_updates
        and heuristic_variant_policy_ids
    ):
        groups.append(
            RuntimeOpponentGroup(
                name="heuristic_public_variant",
                policy_ids=heuristic_variant_policy_ids,
                weight=heuristic_public_variant_weight,
            )
        )
    if noleague_baseline_weight > 0.0 and str(noleague_baseline_policy_id) in model_ids:
        groups.append(
            RuntimeOpponentGroup(
                name="noleague_baseline",
                policy_ids=(str(noleague_baseline_policy_id),),
                weight=noleague_baseline_weight,
            )
        )
    if not pfsp_ready and warmup_snapshot_weight > 0.0 and candidate_ids:
        groups.append(
            RuntimeOpponentGroup(
                name="warmup_snapshot",
                policy_ids=candidate_ids,
                weight=warmup_snapshot_weight,
            )
        )
    if pfsp_ready and mirror_weight > 0.0:
        groups.append(RuntimeOpponentGroup(name="mirror", policy_ids=(str(mirror_policy_id),), weight=mirror_weight))
    if pfsp_ready and hard_negative_ids:
        groups.append(
            RuntimeOpponentGroup(name="hard_negative", policy_ids=hard_negative_ids, weight=hard_negative_weight)
        )
    if pfsp_ready and champion_ids:
        groups.append(RuntimeOpponentGroup(name="champion", policy_ids=champion_ids, weight=champion_weight))
    if pfsp_ready and recent_ids:
        groups.append(RuntimeOpponentGroup(name="recent", policy_ids=recent_ids, weight=recent_weight))
    if not pfsp_ready:
        mirror_weight = max(
            0.0,
            1.0
            - heuristic_public_weight
            - heuristic_public_variant_weight
            - noleague_baseline_weight
            - warmup_snapshot_weight,
        )
        groups.append(RuntimeOpponentGroup(name="mirror", policy_ids=(str(mirror_policy_id),), weight=mirror_weight))
    elif not groups:
        groups.append(RuntimeOpponentGroup(name="recent", policy_ids=candidate_ids, weight=1.0))
    return tuple(groups)


def build_runtime_opponent_sampling_plan(groups: Sequence[RuntimeOpponentGroup]) -> RuntimeOpponentSamplingPlan:
    group_tuple = tuple(groups)
    weights = np.asarray([group.weight for group in group_tuple], dtype=np.float64)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    probabilities = weights / np.sum(weights)
    return RuntimeOpponentSamplingPlan(groups=group_tuple, probabilities=probabilities)


def sample_runtime_opponent_group_policy_ids(
    *,
    group: RuntimeOpponentGroup,
    count: int,
    rng: np.random.Generator,
    league_config: Any,
    outcomes: Any,
) -> tuple[str, ...]:
    sample_count = int(count)
    group_ids = group.policy_ids
    if group.name in {"mirror", "heuristic_public", "noleague_baseline"}:
        return tuple(group_ids[0] for _ in range(sample_count))
    if group.name == "heuristic_public_variant":
        return tuple(str(group_ids[int(index)]) for index in rng.integers(len(group_ids), size=sample_count))
    row_deficit_multipliers = row_deficit_weight_multipliers(policy_ids=group_ids, league_config=league_config)
    focus_multipliers = (
        hard_negative_focus_weight_multipliers(policy_ids=group_ids, league_config=league_config)
        if group.name == "hard_negative"
        else None
    )
    weight_multipliers = combine_weight_multipliers(focus_multipliers, row_deficit_multipliers)
    return sample_opponent_snapshot_ids(
        group_ids,
        count=sample_count,
        rng=rng,
        win_rates_by_snapshot_id={policy_id: outcomes.win_rate(policy_id) for policy_id in group_ids},
        weight_multipliers_by_snapshot_id=weight_multipliers,
        power=float(league_config.pfsp_power),
        eps_uniform=float(league_config.pfsp_epsilon_uniform),
    )


def sample_runtime_opponent_policy_ids(
    *,
    count: int,
    rng: np.random.Generator,
    league_enabled: bool,
    league_config: Any | None,
    pfsp_ready: bool,
    reference_update: int,
    mirror_weight: float,
    heuristic_public_weight: float,
    heuristic_public_variant_weight: float,
    noleague_baseline_weight: float,
    warmup_snapshot_weight: float,
    opponent_candidate_ids: Sequence[str],
    opponent_hard_negative_ids: Sequence[str],
    opponent_champion_ids: Sequence[str],
    opponent_recent_ids: Sequence[str],
    opponent_heuristic_policy_ids: Sequence[str],
    opponent_model_ids: Sequence[str],
    outcomes: Any,
    mirror_policy_id: str,
    heuristic_public_policy_id: str,
    heuristic_public_variant_policy_ids: Sequence[str],
    noleague_baseline_policy_id: str,
) -> OpponentSamplingResult:
    sample_count = int(count)
    if sample_count <= 0:
        return empty_opponent_sampling_result()
    if not bool(league_enabled):
        return OpponentSamplingResult(
            policy_ids=tuple(str(mirror_policy_id) for _ in range(sample_count)),
            mirror_envs=sample_count,
        )
    if league_config is None:
        raise AssertionError("league_config is required when league sampling is enabled")
    groups = build_runtime_opponent_sampling_groups(
        league_config=league_config,
        pfsp_ready=pfsp_ready,
        reference_update=reference_update,
        mirror_weight=mirror_weight,
        heuristic_public_weight=heuristic_public_weight,
        heuristic_public_variant_weight=heuristic_public_variant_weight,
        noleague_baseline_weight=noleague_baseline_weight,
        warmup_snapshot_weight=warmup_snapshot_weight,
        opponent_candidate_ids=opponent_candidate_ids,
        opponent_hard_negative_ids=opponent_hard_negative_ids,
        opponent_champion_ids=opponent_champion_ids,
        opponent_recent_ids=opponent_recent_ids,
        opponent_heuristic_policy_ids=opponent_heuristic_policy_ids,
        opponent_model_ids=opponent_model_ids,
        mirror_policy_id=mirror_policy_id,
        heuristic_public_policy_id=heuristic_public_policy_id,
        heuristic_public_variant_policy_ids=heuristic_public_variant_policy_ids,
        noleague_baseline_policy_id=noleague_baseline_policy_id,
    )
    plan = build_runtime_opponent_sampling_plan(groups)
    sampled_group_indices = rng.choice(len(plan.groups), size=sample_count, replace=True, p=plan.probabilities)
    accumulator = OpponentSamplingAccumulator.create(sample_count)
    for group_index, group in enumerate(plan.groups):
        positions = np.flatnonzero(sampled_group_indices == group_index)
        if positions.size == 0:
            continue
        sampled_group_ids = sample_runtime_opponent_group_policy_ids(
            group=group,
            count=int(positions.size),
            rng=rng,
            league_config=league_config,
            outcomes=outcomes,
        )
        accumulator.record(group_name=group.name, positions=positions, policy_ids=sampled_group_ids)
    return accumulator.result()


def sample_warmup_snapshot_policy_ids(
    *,
    count: int,
    rng: np.random.Generator,
    opponent_candidate_ids: Sequence[str],
    league_config: Any | None,
    outcomes: Any,
) -> OpponentSamplingResult:
    sample_count = int(count)
    candidate_ids = tuple(str(policy_id) for policy_id in opponent_candidate_ids)
    if sample_count <= 0 or not candidate_ids:
        return empty_opponent_sampling_result()
    if league_config is None:
        raise AssertionError("league_config is required for warmup snapshot sampling")
    sampled_policy_ids = sample_opponent_snapshot_ids(
        candidate_ids,
        count=sample_count,
        rng=rng,
        win_rates_by_snapshot_id={policy_id: outcomes.win_rate(policy_id) for policy_id in candidate_ids},
        power=float(league_config.pfsp_power),
        eps_uniform=float(league_config.pfsp_epsilon_uniform),
    )
    return OpponentSamplingResult(
        policy_ids=tuple(str(policy_id) for policy_id in sampled_policy_ids),
        sampled_envs=sample_count,
        warmup_snapshot_envs=sample_count,
        sampled_policy_envs=count_items(Counter(str(policy_id) for policy_id in sampled_policy_ids)),
        warmup_snapshot_policy_envs=count_items(Counter(str(policy_id) for policy_id in sampled_policy_ids)),
    )
