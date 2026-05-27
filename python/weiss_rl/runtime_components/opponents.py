"""Opponent-pool bookkeeping helpers for the queue runtime."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.core.schedules import linear_anneal_value
from weiss_rl.league.opponent_pool import sample_opponent_snapshot_ids
from weiss_rl.league.registry import SnapshotRegistry


@dataclass(frozen=True, slots=True)
class OpponentSamplingResult:
    policy_ids: tuple[str, ...]
    sampled_envs: int = 0
    mirror_envs: int = 0
    heuristic_public_envs: int = 0
    heuristic_public_variant_envs: int = 0
    noleague_baseline_envs: int = 0
    champion_envs: int = 0
    recent_envs: int = 0
    hard_negative_envs: int = 0
    warmup_snapshot_envs: int = 0
    sampled_policy_envs: tuple[tuple[str, int], ...] = ()
    heuristic_public_policy_envs: tuple[tuple[str, int], ...] = ()
    heuristic_public_variant_policy_envs: tuple[tuple[str, int], ...] = ()
    noleague_baseline_policy_envs: tuple[tuple[str, int], ...] = ()
    champion_policy_envs: tuple[tuple[str, int], ...] = ()
    recent_policy_envs: tuple[tuple[str, int], ...] = ()
    hard_negative_policy_envs: tuple[tuple[str, int], ...] = ()
    warmup_snapshot_policy_envs: tuple[tuple[str, int], ...] = ()


def empty_opponent_sampling_result() -> OpponentSamplingResult:
    return OpponentSamplingResult(policy_ids=())


def _count_items(counts: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple((str(policy_id), int(count)) for policy_id, count in sorted(counts.items()) if int(count) > 0)


def active_noleague_baseline_mix_fraction(*, league_config: Any | None, reference_update: int) -> float:
    if league_config is None:
        return 0.0
    sampling_cfg = getattr(league_config, "sampling", league_config)
    noleague_mix_fraction = max(
        0.0,
        float(getattr(sampling_cfg, "noleague_baseline_mix_fraction", 0.0)),
    )
    if noleague_mix_fraction <= 0.0:
        return 0.0
    mix_end_updates = int(getattr(sampling_cfg, "noleague_baseline_mix_end_updates", -1))
    if mix_end_updates >= 0 and int(reference_update) >= mix_end_updates:
        return 0.0
    return noleague_mix_fraction


def active_annealed_mix_fraction(
    *,
    league_config: Any | None,
    reference_update: int,
    initial_attr: str,
    final_attr: str,
    end_attr: str,
) -> float:
    if league_config is None:
        return 0.0
    sampling_cfg = getattr(league_config, "sampling", league_config)
    initial_fraction = max(0.0, float(getattr(sampling_cfg, initial_attr, 0.0)))
    final_fraction = max(0.0, float(getattr(sampling_cfg, final_attr, initial_fraction)))
    end_updates = int(getattr(sampling_cfg, end_attr, -1))
    if end_updates < 0 or initial_fraction == final_fraction:
        return initial_fraction
    current_update = max(0, int(reference_update))
    if end_updates == 0:
        return final_fraction
    if current_update >= end_updates:
        return final_fraction
    progress = float(current_update) / float(end_updates)
    return initial_fraction + (final_fraction - initial_fraction) * progress


def active_heuristic_public_mix_fraction(*, league_config: Any | None, reference_update: int) -> float:
    return active_annealed_mix_fraction(
        league_config=league_config,
        reference_update=reference_update,
        initial_attr="heuristic_public_mix_fraction",
        final_attr="heuristic_public_final_mix_fraction",
        end_attr="heuristic_public_mix_end_updates",
    )


def active_heuristic_public_variant_mix_fraction(*, league_config: Any | None, reference_update: int) -> float:
    return active_annealed_mix_fraction(
        league_config=league_config,
        reference_update=reference_update,
        initial_attr="heuristic_public_variant_mix_fraction",
        final_attr="heuristic_public_variant_final_mix_fraction",
        end_attr="heuristic_public_variant_mix_end_updates",
    )


def active_mirror_mix_fraction(*, league_config: Any | None, reference_update: int) -> float:
    return active_annealed_mix_fraction(
        league_config=league_config,
        reference_update=reference_update,
        initial_attr="mirror_mix_fraction",
        final_attr="mirror_final_mix_fraction",
        end_attr="mirror_mix_end_updates",
    )


def active_warmup_snapshot_mix_fraction(
    *,
    league_config: Any | None,
    reference_update: int,
    has_opponent_candidates: bool,
    has_opponent_models: bool,
) -> float:
    if league_config is None:
        return 0.0
    sampling_cfg = getattr(league_config, "sampling", league_config)
    warmup_fraction = max(
        0.0,
        float(getattr(sampling_cfg, "warmup_snapshot_mix_fraction", 0.0)),
    )
    if warmup_fraction <= 0.0:
        return 0.0
    if int(reference_update) >= int(league_config.warmup.first_updates):
        return 0.0
    if not has_opponent_candidates or not has_opponent_models:
        return 0.0
    return warmup_fraction


def active_actor_heuristic_fraction(
    *,
    initial_fraction: float,
    final_fraction: float | None,
    start_updates: int,
    end_updates: int,
    reference_update: int,
) -> float:
    initial = max(0.0, min(1.0, float(initial_fraction)))
    final = max(0.0, min(1.0, float(initial if final_fraction is None else final_fraction)))
    return float(
        linear_anneal_value(
            initial_value=initial,
            final_value=final,
            start_update=max(0, int(start_updates)),
            end_update=int(end_updates),
            update_count=max(0, int(reference_update)),
        )
    )


def fixed_opponent_policy_slots(
    *,
    envs_per_actor: int,
    heuristic_reserved_envs: int,
    noleague_reserved_envs: int,
    heuristic_policy_id: str,
    noleague_policy_id: str,
) -> np.ndarray | None:
    env_count = int(envs_per_actor)
    slots = np.full((env_count,), "", dtype=object)
    cursor = 0
    heuristic_count = min(int(heuristic_reserved_envs), env_count - cursor)
    if heuristic_count > 0:
        slots[cursor : cursor + heuristic_count] = heuristic_policy_id
        cursor += heuristic_count
    baseline_count = min(int(noleague_reserved_envs), env_count - cursor)
    if baseline_count > 0:
        slots[cursor : cursor + baseline_count] = noleague_policy_id
        cursor += baseline_count
    if cursor <= 0:
        return None
    return slots


def fixed_opponent_policy_is_active(
    *,
    policy_id: str,
    forced_policy_ids: Sequence[str],
    heuristic_policy_ids: Sequence[str],
    opponent_model_ids: Sequence[str],
    league_config: Any | None,
    reference_update: int,
    noleague_policy_id: str,
) -> bool:
    policy_key = str(policy_id).strip()
    if not policy_key:
        return False
    heuristic_ids = set(str(policy_id) for policy_id in heuristic_policy_ids)
    model_ids = set(str(policy_id) for policy_id in opponent_model_ids)
    forced_ids = set(str(policy_id) for policy_id in forced_policy_ids)
    if policy_key in forced_ids:
        if policy_key in heuristic_ids:
            return policy_key in heuristic_ids
        if policy_key == noleague_policy_id:
            return policy_key in model_ids
    if policy_key in heuristic_ids:
        if league_config is None:
            return False
        sampling_cfg = getattr(league_config, "sampling", league_config)
        start_updates = int(getattr(sampling_cfg, "heuristic_public_start_updates", 0))
        return int(reference_update) >= start_updates and policy_key in heuristic_ids
    if policy_key == noleague_policy_id:
        return (
            league_config is not None
            and int(reference_update) >= int(league_config.warmup.first_updates)
            and policy_key in model_ids
        )
    return policy_key in model_ids or policy_key in heuristic_ids


def active_assigned_opponent_policy_ids(
    *,
    actors: Sequence[Any],
    mirror_policy_id: str,
) -> tuple[str, ...]:
    active_policy_ids: list[str] = []
    for actor in actors:
        policy_ids = getattr(actor, "opponent_policy_id_by_env", None)
        if policy_ids is None:
            continue
        for policy_id in np.asarray(policy_ids, dtype=object).tolist():
            policy_id_text = str(policy_id).strip()
            if not policy_id_text or policy_id_text == mirror_policy_id:
                continue
            active_policy_ids.append(policy_id_text)
    return tuple(dict.fromkeys(active_policy_ids))


def configured_fixed_opponent_policy_ids(
    *,
    heuristic_reserved_envs_per_actor: int,
    noleague_baseline_reserved_envs_per_actor: int,
    heuristic_policy_id: str,
    noleague_policy_id: str,
    heuristic_policy_ids: Sequence[str],
) -> tuple[str, ...]:
    policy_ids: list[str] = []
    if int(heuristic_reserved_envs_per_actor) > 0 and str(heuristic_policy_id) in {
        str(policy_id) for policy_id in heuristic_policy_ids
    }:
        policy_ids.append(str(heuristic_policy_id))
    if int(noleague_baseline_reserved_envs_per_actor) > 0:
        policy_ids.append(str(noleague_policy_id))
    return tuple(dict.fromkeys(policy_ids))


def configured_resident_opponent_policy_ids(
    *,
    fixed_policy_ids: Sequence[str],
    heuristic_variant_mix_fraction: float,
    noleague_mix_fraction: float,
    heuristic_variant_policy_ids: Sequence[str],
    heuristic_policy_ids: Sequence[str],
    noleague_policy_id: str,
) -> tuple[str, ...]:
    policy_ids = [str(policy_id) for policy_id in fixed_policy_ids]
    heuristic_ids = {str(policy_id) for policy_id in heuristic_policy_ids}
    if float(heuristic_variant_mix_fraction) > 0.0:
        policy_ids.extend(
            str(policy_id) for policy_id in heuristic_variant_policy_ids if str(policy_id) in heuristic_ids
        )
    if float(noleague_mix_fraction) > 0.0:
        policy_ids.append(str(noleague_policy_id))
    return tuple(dict.fromkeys(policy_ids))


def promotion_gated_recent_reservoir_size(
    *,
    base_recent_size: int,
    champion_size: int,
    admitted_champion_ids: Sequence[str],
    min_recent_size: int,
) -> int:
    base_recent_size_i = max(0, int(base_recent_size))
    if base_recent_size_i <= 0:
        return 0
    if not admitted_champion_ids:
        return min(base_recent_size_i, max(1, int(champion_size)))
    return min(
        base_recent_size_i,
        max(int(min_recent_size), max(1, int(champion_size) // 2)),
    )


def filter_timeout_heavy_opponents(
    *,
    candidate_ids: Sequence[str],
    league_config: Any | None,
    outcomes: Any,
    min_samples: int,
) -> tuple[str, ...]:
    if not candidate_ids or league_config is None or not bool(league_config.promotion_gate_enabled):
        return tuple(candidate_ids)
    timeout_threshold = float(league_config.promotion.gate.guardrails.max_truncation_rate)
    kept: list[str] = []
    for policy_id in candidate_ids:
        wins, losses, draws, timeouts = outcomes.counts(policy_id)
        total = int(wins + losses + draws + timeouts)
        if total < int(min_samples):
            kept.append(policy_id)
            continue
        timeout_rate = float(timeouts) / float(total)
        if timeout_rate <= timeout_threshold:
            kept.append(policy_id)
    return tuple(kept)


def apply_opponent_pool_diversity_floor(
    *,
    candidate_ids: Sequence[str],
    filtered_candidate_ids: Sequence[str],
    minimum_floor_size: int,
) -> tuple[tuple[str, ...], int]:
    original_ids = tuple(str(policy_id) for policy_id in candidate_ids)
    filtered_ids = tuple(str(policy_id) for policy_id in filtered_candidate_ids)
    if not original_ids:
        return (), 0
    if not filtered_ids:
        return original_ids, 0
    raw_quarantined_count = max(0, len(original_ids) - len(filtered_ids))
    restored: list[str] = list(filtered_ids)
    minimum_size = min(len(original_ids), int(minimum_floor_size))
    if len(restored) < minimum_size:
        restored_set = set(restored)
        for policy_id in original_ids:
            if policy_id in restored_set:
                continue
            restored.append(policy_id)
            restored_set.add(policy_id)
            if len(restored) >= minimum_size:
                break
    return tuple(restored), raw_quarantined_count


def select_hard_negative_ids(
    *,
    candidate_ids: Sequence[str],
    league_config: Any | None,
    outcomes: Any | None,
    registry_path: Path | None,
) -> tuple[str, ...]:
    if not candidate_ids or league_config is None or outcomes is None:
        return ()
    sampling_cfg = getattr(league_config, "sampling", league_config)
    min_samples = int(getattr(sampling_cfg, "hard_negative_min_samples", 16))
    max_win_rate = float(getattr(sampling_cfg, "hard_negative_max_win_rate", 0.45))
    scored: list[tuple[float, int, str]] = []
    snapshots_by_id: dict[str, int] = {}
    if registry_path is not None and registry_path.is_file():
        registry = SnapshotRegistry.load(registry_path)
        snapshots_by_id = {snapshot.policy_id: int(snapshot.update) for snapshot in registry.snapshots}
    for policy_id in candidate_ids:
        wins, losses, draws, timeouts = outcomes.counts(policy_id)
        total = int(wins + losses + draws + timeouts)
        if total < min_samples:
            continue
        win_rate = float(outcomes.win_rate(policy_id))
        if win_rate <= max_win_rate:
            scored.append((win_rate, -int(snapshots_by_id.get(policy_id, 0)), str(policy_id)))
    scored.sort()
    selected = [policy_id for _, _, policy_id in scored]
    selected_set = set(selected)
    focus_ids = (
        *configured_hard_negative_focus_policy_ids(league_config=league_config),
        *configured_row_deficit_policy_ids(league_config=league_config),
    )
    if focus_ids:
        for policy_id in candidate_ids:
            policy_id_text = str(policy_id)
            if policy_id_text in selected_set:
                continue
            if any(hard_negative_focus_policy_id_matches(policy_id_text, focus_id) for focus_id in focus_ids):
                selected.append(policy_id_text)
                selected_set.add(policy_id_text)
    return tuple(selected)


def configured_hard_negative_focus_policy_ids(*, league_config: Any | None) -> tuple[str, ...]:
    if league_config is None:
        return ()
    sampling_cfg = getattr(league_config, "sampling", league_config)
    raw_ids = getattr(sampling_cfg, "hard_negative_focus_policy_ids", ())
    return tuple(dict.fromkeys(str(policy_id).strip() for policy_id in raw_ids if str(policy_id).strip()))


def hard_negative_focus_policy_id_matches(policy_id: str, focus_policy_id: str) -> bool:
    policy_id_text = str(policy_id).strip()
    focus_id_text = str(focus_policy_id).strip()
    if not policy_id_text or not focus_id_text:
        return False
    return policy_id_text == focus_id_text or policy_id_text.endswith(f"_{focus_id_text}")


def hard_negative_focus_weight_multipliers(
    *,
    policy_ids: Sequence[str],
    league_config: Any | None,
) -> Mapping[str, float] | None:
    focus_ids = configured_hard_negative_focus_policy_ids(league_config=league_config)
    if not focus_ids:
        return None
    sampling_cfg = getattr(league_config, "sampling", league_config)
    multiplier = float(getattr(sampling_cfg, "hard_negative_focus_weight_multiplier", 1.0))
    if multiplier == 1.0:
        return None
    multipliers: dict[str, float] = {}
    for policy_id in policy_ids:
        policy_id_text = str(policy_id)
        if any(hard_negative_focus_policy_id_matches(policy_id_text, focus_id) for focus_id in focus_ids):
            multipliers[policy_id_text] = multiplier
    return multipliers or None


def configured_row_deficit_policy_weights(*, league_config: Any | None) -> tuple[tuple[str, float], ...]:
    if league_config is None:
        return ()
    sampling_cfg = getattr(league_config, "sampling", league_config)
    raw_weights = getattr(sampling_cfg, "row_deficit_policy_weights", ())
    if isinstance(raw_weights, Mapping):
        items = raw_weights.items()
    else:
        items = raw_weights
    parsed: dict[str, float] = {}
    for raw_policy_id, raw_weight in items:
        policy_id = str(raw_policy_id).strip()
        if not policy_id:
            continue
        weight = float(raw_weight)
        if weight <= 0.0:
            continue
        parsed[policy_id] = weight
    return tuple(sorted(parsed.items()))


def configured_row_deficit_policy_ids(*, league_config: Any | None) -> tuple[str, ...]:
    return tuple(policy_id for policy_id, _ in configured_row_deficit_policy_weights(league_config=league_config))


def row_deficit_weight_multipliers(
    *,
    policy_ids: Sequence[str],
    league_config: Any | None,
) -> Mapping[str, float] | None:
    configured_weights = configured_row_deficit_policy_weights(league_config=league_config)
    if not configured_weights:
        return None
    multipliers: dict[str, float] = {}
    for policy_id in policy_ids:
        policy_id_text = str(policy_id)
        for focus_id, weight in configured_weights:
            if hard_negative_focus_policy_id_matches(policy_id_text, focus_id):
                multipliers[policy_id_text] = multipliers.get(policy_id_text, 1.0) * float(weight)
    return multipliers or None


def combine_weight_multipliers(*multipliers: Mapping[str, float] | None) -> Mapping[str, float] | None:
    combined: dict[str, float] = {}
    for raw_multiplier in multipliers:
        if raw_multiplier is None:
            continue
        for policy_id, weight in raw_multiplier.items():
            combined[str(policy_id)] = combined.get(str(policy_id), 1.0) * float(weight)
    return combined or None


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
    sampling_cfg = getattr(league_config, "sampling", league_config)
    groups: list[tuple[str, tuple[str, ...], float]] = []
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
    if (
        heuristic_public_weight > 0.0
        and int(reference_update) >= heuristic_public_start_updates
        and str(heuristic_public_policy_id) in heuristic_ids
    ):
        groups.append(("heuristic_public", (str(heuristic_public_policy_id),), heuristic_public_weight))
    heuristic_variant_policy_ids = tuple(
        str(policy_id) for policy_id in heuristic_public_variant_policy_ids if str(policy_id) in heuristic_ids
    )
    if (
        heuristic_public_variant_weight > 0.0
        and int(reference_update) >= heuristic_public_start_updates
        and heuristic_variant_policy_ids
    ):
        groups.append(("heuristic_public_variant", heuristic_variant_policy_ids, heuristic_public_variant_weight))
    if noleague_baseline_weight > 0.0 and str(noleague_baseline_policy_id) in model_ids:
        groups.append(("noleague_baseline", (str(noleague_baseline_policy_id),), noleague_baseline_weight))
    if not pfsp_ready and warmup_snapshot_weight > 0.0 and candidate_ids:
        groups.append(("warmup_snapshot", candidate_ids, warmup_snapshot_weight))
    if pfsp_ready and mirror_weight > 0.0:
        groups.append(("mirror", (str(mirror_policy_id),), mirror_weight))
    if pfsp_ready and hard_negative_ids:
        groups.append(("hard_negative", hard_negative_ids, hard_negative_weight))
    if pfsp_ready and champion_ids:
        groups.append(("champion", champion_ids, champion_weight))
    if pfsp_ready and recent_ids:
        groups.append(("recent", recent_ids, recent_weight))
    if not pfsp_ready:
        mirror_weight = max(
            0.0,
            1.0
            - heuristic_public_weight
            - heuristic_public_variant_weight
            - noleague_baseline_weight
            - warmup_snapshot_weight,
        )
        groups.append(("mirror", (str(mirror_policy_id),), mirror_weight))
    elif not groups:
        groups.append(("recent", candidate_ids, 1.0))

    weights = np.asarray([weight for _, _, weight in groups], dtype=np.float64)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)
    sampled_group_indices = rng.choice(len(groups), size=sample_count, replace=True, p=weights)
    sampled_policy_ids_list = [""] * sample_count
    mirror_count = 0
    heuristic_public_count = 0
    heuristic_public_variant_count = 0
    noleague_baseline_count = 0
    hard_negative_count = 0
    champion_count = 0
    recent_count = 0
    warmup_snapshot_count = 0
    sampled_policy_counts: Counter[str] = Counter()
    heuristic_public_policy_counts: Counter[str] = Counter()
    heuristic_public_variant_policy_counts: Counter[str] = Counter()
    noleague_baseline_policy_counts: Counter[str] = Counter()
    champion_policy_counts: Counter[str] = Counter()
    recent_policy_counts: Counter[str] = Counter()
    hard_negative_policy_counts: Counter[str] = Counter()
    warmup_snapshot_policy_counts: Counter[str] = Counter()
    for group_index, (group_name, group_ids, _) in enumerate(groups):
        positions = np.flatnonzero(sampled_group_indices == group_index)
        if positions.size == 0:
            continue
        if group_name in {"mirror", "heuristic_public", "noleague_baseline"}:
            sampled_group_ids = tuple(group_ids[0] for _ in range(int(positions.size)))
        elif group_name == "heuristic_public_variant":
            sampled_group_ids = tuple(
                str(group_ids[int(index)]) for index in rng.integers(len(group_ids), size=int(positions.size))
            )
        else:
            row_deficit_multipliers = row_deficit_weight_multipliers(policy_ids=group_ids, league_config=league_config)
            focus_multipliers = (
                hard_negative_focus_weight_multipliers(policy_ids=group_ids, league_config=league_config)
                if group_name == "hard_negative"
                else None
            )
            weight_multipliers = combine_weight_multipliers(focus_multipliers, row_deficit_multipliers)
            sampled_group_ids = sample_opponent_snapshot_ids(
                group_ids,
                count=int(positions.size),
                rng=rng,
                win_rates_by_snapshot_id={policy_id: outcomes.win_rate(policy_id) for policy_id in group_ids},
                weight_multipliers_by_snapshot_id=weight_multipliers,
                power=float(league_config.pfsp_power),
                eps_uniform=float(league_config.pfsp_epsilon_uniform),
            )
        for idx, policy_id in zip(positions.tolist(), sampled_group_ids, strict=True):
            sampled_policy_ids_list[int(idx)] = str(policy_id)
            if group_name != "mirror":
                sampled_policy_counts[str(policy_id)] += 1
        if group_name == "mirror":
            mirror_count += int(positions.size)
        elif group_name == "heuristic_public":
            heuristic_public_count += int(positions.size)
            heuristic_public_policy_counts.update(str(policy_id) for policy_id in sampled_group_ids)
        elif group_name == "heuristic_public_variant":
            heuristic_public_variant_count += int(positions.size)
            heuristic_public_variant_policy_counts.update(str(policy_id) for policy_id in sampled_group_ids)
        elif group_name == "noleague_baseline":
            noleague_baseline_count += int(positions.size)
            noleague_baseline_policy_counts.update(str(policy_id) for policy_id in sampled_group_ids)
        elif group_name == "hard_negative":
            hard_negative_count += int(positions.size)
            hard_negative_policy_counts.update(str(policy_id) for policy_id in sampled_group_ids)
        elif group_name == "champion":
            champion_count += int(positions.size)
            champion_policy_counts.update(str(policy_id) for policy_id in sampled_group_ids)
        elif group_name == "warmup_snapshot":
            warmup_snapshot_count += int(positions.size)
            warmup_snapshot_policy_counts.update(str(policy_id) for policy_id in sampled_group_ids)
        else:
            recent_count += int(positions.size)
            recent_policy_counts.update(str(policy_id) for policy_id in sampled_group_ids)
    return OpponentSamplingResult(
        policy_ids=tuple(str(policy_id) for policy_id in sampled_policy_ids_list),
        sampled_envs=sample_count - mirror_count,
        mirror_envs=mirror_count,
        heuristic_public_envs=heuristic_public_count,
        heuristic_public_variant_envs=heuristic_public_variant_count,
        noleague_baseline_envs=noleague_baseline_count,
        champion_envs=champion_count,
        recent_envs=recent_count,
        hard_negative_envs=hard_negative_count,
        warmup_snapshot_envs=warmup_snapshot_count,
        sampled_policy_envs=_count_items(sampled_policy_counts),
        heuristic_public_policy_envs=_count_items(heuristic_public_policy_counts),
        heuristic_public_variant_policy_envs=_count_items(heuristic_public_variant_policy_counts),
        noleague_baseline_policy_envs=_count_items(noleague_baseline_policy_counts),
        champion_policy_envs=_count_items(champion_policy_counts),
        recent_policy_envs=_count_items(recent_policy_counts),
        hard_negative_policy_envs=_count_items(hard_negative_policy_counts),
        warmup_snapshot_policy_envs=_count_items(warmup_snapshot_policy_counts),
    )


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
        sampled_policy_envs=_count_items(Counter(str(policy_id) for policy_id in sampled_policy_ids)),
        warmup_snapshot_policy_envs=_count_items(Counter(str(policy_id) for policy_id in sampled_policy_ids)),
    )
