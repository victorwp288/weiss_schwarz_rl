from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from weiss_rl.runtime.components.opponents import sample_runtime_opponent_policy_ids

from .runtime_opponent_sampling_test_support import OpponentSamplingOutcomes


def sampling_config(
    *,
    champion_mix_fraction: float = 0.0,
    hard_negative_mix_fraction: float = 0.0,
    warmup_first_updates: int | None = None,
    hard_negative_focus_policy_ids: tuple[str, ...] = (),
    hard_negative_focus_weight_multiplier: float = 1.0,
    row_deficit_policy_weights: tuple[tuple[str, float], ...] = (),
) -> SimpleNamespace:
    config = SimpleNamespace(
        pfsp_power=2.0,
        pfsp_epsilon_uniform=0.0,
        sampling=SimpleNamespace(
            heuristic_public_start_updates=0,
            champion_mix_fraction=champion_mix_fraction,
            hard_negative_mix_fraction=hard_negative_mix_fraction,
        ),
    )
    if warmup_first_updates is not None:
        config.warmup = SimpleNamespace(first_updates=warmup_first_updates)
    if hard_negative_focus_policy_ids:
        config.sampling.hard_negative_focus_policy_ids = hard_negative_focus_policy_ids
        config.sampling.hard_negative_focus_weight_multiplier = hard_negative_focus_weight_multiplier
    if row_deficit_policy_weights:
        config.sampling.row_deficit_policy_weights = row_deficit_policy_weights
    return config


def sample_runtime_policy_ids(
    *,
    count: int,
    rng_seed: int,
    league_enabled: bool = True,
    league_config: Any | None = None,
    pfsp_ready: bool = False,
    reference_update: int = 0,
    mirror_weight: float = 0.0,
    heuristic_public_weight: float = 0.0,
    heuristic_public_variant_weight: float = 0.0,
    noleague_baseline_weight: float = 0.0,
    warmup_snapshot_weight: float = 0.0,
    opponent_candidate_ids: tuple[str, ...] = (),
    opponent_hard_negative_ids: tuple[str, ...] = (),
    opponent_champion_ids: tuple[str, ...] = (),
    opponent_recent_ids: tuple[str, ...] = (),
    opponent_heuristic_policy_ids: tuple[str, ...] = (),
    opponent_model_ids: tuple[str, ...] = (),
    outcomes: Any | None = None,
):
    return sample_runtime_opponent_policy_ids(
        count=count,
        rng=np.random.default_rng(rng_seed),
        league_enabled=league_enabled,
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
        outcomes=OpponentSamplingOutcomes() if outcomes is None and league_enabled else outcomes,
        mirror_policy_id="mirror",
        heuristic_public_policy_id="heuristic",
        heuristic_public_variant_policy_ids=("aggro", "control"),
        noleague_baseline_policy_id="baseline",
    )
