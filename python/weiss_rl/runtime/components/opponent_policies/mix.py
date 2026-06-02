"""League opponent mix schedules used by runtime sampling."""

from __future__ import annotations

from typing import Any

from weiss_rl.core.schedules import linear_anneal_value


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


__all__ = [
    "active_actor_heuristic_fraction",
    "active_annealed_mix_fraction",
    "active_heuristic_public_mix_fraction",
    "active_heuristic_public_variant_mix_fraction",
    "active_mirror_mix_fraction",
    "active_noleague_baseline_mix_fraction",
    "active_warmup_snapshot_mix_fraction",
]
