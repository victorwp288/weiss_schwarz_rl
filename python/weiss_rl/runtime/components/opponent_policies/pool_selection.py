"""Pure opponent-pool selection and weighting helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.league.registry import SnapshotRegistry


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
    items: Iterable[tuple[Any, Any]]
    if isinstance(raw_weights, Mapping):
        items = tuple(raw_weights.items())
    else:
        items = tuple(raw_weights)
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


__all__ = [
    "active_assigned_opponent_policy_ids",
    "apply_opponent_pool_diversity_floor",
    "combine_weight_multipliers",
    "configured_fixed_opponent_policy_ids",
    "configured_hard_negative_focus_policy_ids",
    "configured_resident_opponent_policy_ids",
    "configured_row_deficit_policy_ids",
    "configured_row_deficit_policy_weights",
    "filter_timeout_heavy_opponents",
    "fixed_opponent_policy_is_active",
    "fixed_opponent_policy_slots",
    "hard_negative_focus_policy_id_matches",
    "hard_negative_focus_weight_multipliers",
    "promotion_gated_recent_reservoir_size",
    "row_deficit_weight_multipliers",
    "select_hard_negative_ids",
]
