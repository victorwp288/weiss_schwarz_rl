"""Runtime metric aggregation helpers.

These functions keep public metric names and aggregation behavior in one place
while leaving `QueueRuntime` responsible for owning mutable runtime state.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from weiss_rl.runtime.components.outcomes import parse_outcome_counter_key

MAX_COUNTER_KEYS = frozenset({"max_consecutive_main_moves"})


def runtime_counter_totals(selected: Sequence[Any]) -> dict[str, float]:
    counter_totals: dict[str, float] = {}
    for unroll in selected:
        counters = getattr(unroll, "counters", None)
        if counters is None:
            continue
        for key, value in counters.items():
            if parse_outcome_counter_key(str(key)) is not None:
                continue
            numeric_value = float(value)
            if key in MAX_COUNTER_KEYS:
                counter_totals[key] = max(counter_totals.get(key, 0.0), numeric_value)
            else:
                counter_totals[key] = counter_totals.get(key, 0.0) + numeric_value
    return counter_totals


def _metric_safe_policy_id(policy_id: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", str(policy_id).strip()).strip("_").lower()
    return sanitized or "unknown"


def runtime_outcome_metrics(selected: Sequence[Any]) -> dict[str, float]:
    outcome_totals: dict[str, dict[str, int]] = {}
    for unroll in selected:
        counters = getattr(unroll, "counters", None)
        if counters is None:
            continue
        for key, value in counters.items():
            parsed = parse_outcome_counter_key(str(key))
            if parsed is None:
                continue
            policy_id, outcome = parsed
            policy_totals = outcome_totals.setdefault(
                str(policy_id),
                {
                    "w": 0,
                    "l": 0,
                    "d": 0,
                    "t": 0,
                },
            )
            policy_totals[str(outcome)] += int(value)

    metrics: dict[str, float] = {}
    for policy_id, counts in outcome_totals.items():
        safe_policy_id = _metric_safe_policy_id(policy_id)
        wins = int(counts.get("w", 0))
        losses = int(counts.get("l", 0))
        draws = int(counts.get("d", 0))
        timeouts = int(counts.get("t", 0))
        games = int(wins + losses + draws + timeouts)
        decisive_games = int(wins + losses)
        prefix = f"collector_outcome_vs_{safe_policy_id}"
        metrics[f"{prefix}_wins"] = float(wins)
        metrics[f"{prefix}_losses"] = float(losses)
        metrics[f"{prefix}_draws"] = float(draws)
        metrics[f"{prefix}_timeouts"] = float(timeouts)
        metrics[f"{prefix}_games"] = float(games)
        metrics[f"{prefix}_win_rate"] = float(wins / games) if games > 0 else 0.0
        metrics[f"{prefix}_decisive_win_rate"] = float(wins / decisive_games) if decisive_games > 0 else 0.0
    return metrics


def build_runtime_metrics(
    *,
    selected: Sequence[Any],
    occupancy_samples: Sequence[float],
    now: float,
    runtime_start: float,
    runtime_last_metrics_time: float,
    runtime_cumulative_env_steps: int,
    last_published_snapshot_version: int,
    current_learner_update: int,
    effective_learner_update: int,
    actor_heuristic_fraction_active: float,
    mirror_mix_fraction_active: float,
    heuristic_public_mix_fraction_active: float,
    heuristic_public_variant_mix_fraction_active: float,
    warmup_snapshot_mix_fraction_active: float,
    pfsp_pool_size: int,
    pfsp_quarantined_opponents: int,
    pfsp_champion_pool_size: int,
    pfsp_recent_pool_size: int,
    pfsp_hard_negative_pool_size: int,
    pfsp_last_sampled_envs: int,
    pfsp_last_mirror_envs: int,
    pfsp_last_heuristic_public_envs: int,
    pfsp_last_heuristic_public_variant_envs: int,
    pfsp_last_noleague_baseline_envs: int,
    pfsp_last_champion_envs: int,
    pfsp_last_recent_envs: int,
    pfsp_last_hard_negative_envs: int,
    pfsp_last_warmup_snapshot_envs: int,
    pfsp_epoch: int,
) -> tuple[dict[str, float], int]:
    batch_env_steps = sum(int(unroll.obs.shape[0] * unroll.obs.shape[1]) for unroll in selected)
    elapsed = max(float(now) - float(runtime_start), 1e-6)
    elapsed_window = max(float(now) - float(runtime_last_metrics_time), 1e-6)
    next_cumulative_env_steps = int(runtime_cumulative_env_steps) + int(batch_env_steps)
    policy_lags = [
        float(int(last_published_snapshot_version) - int(unroll.behavior_policy_version)) for unroll in selected
    ]
    learner_actor_lags = [
        float(max(0, int(current_learner_update) - int(unroll.behavior_policy_version))) for unroll in selected
    ]
    occupancy = np.asarray(tuple(occupancy_samples) or (0.0,), dtype=np.float64)
    lag_array = np.asarray(policy_lags or (0.0,), dtype=np.float64)
    learner_actor_lag_array = np.asarray(learner_actor_lags or (0.0,), dtype=np.float64)
    counter_totals = runtime_counter_totals(selected)
    outcome_metrics = runtime_outcome_metrics(selected)
    tactical_rows = float(counter_totals.get("tactical_row_count", 0.0))
    packed_candidates = float(counter_totals.get("packed_candidate_count", 0.0))
    row_count_total = float(counter_totals.get("focal_row_count", 0.0) + counter_totals.get("opponent_row_count", 0.0))
    if row_count_total <= 0.0:
        row_count_total = float(counter_totals.get("total_actions", 0.0))
    if row_count_total <= 0.0:
        row_count_total = tactical_rows
    metrics = {
        "actor_env_steps_per_sec": float(batch_env_steps / elapsed_window),
        "actor_env_steps_per_sec_cumulative": float(next_cumulative_env_steps / elapsed),
        "batch_env_steps": float(batch_env_steps),
        "queue_occupancy_p50": float(np.percentile(occupancy, 50)),
        "queue_occupancy_p90": float(np.percentile(occupancy, 90)),
        "policy_version_lag_p50": float(np.percentile(lag_array, 50)),
        "policy_version_lag_p90": float(np.percentile(lag_array, 90)),
        "learner_actor_update_lag_p50": float(np.percentile(learner_actor_lag_array, 50)),
        "learner_actor_update_lag_p90": float(np.percentile(learner_actor_lag_array, 90)),
        "learner_update_for_collected_batch": float(current_learner_update),
        "last_published_snapshot_version": float(last_published_snapshot_version),
        "league_effective_update": float(effective_learner_update),
        "league_update_lag": float(max(0, int(current_learner_update) - int(effective_learner_update))),
        "actor_heuristic_fraction_active": float(actor_heuristic_fraction_active),
        "mirror_mix_fraction_active": float(mirror_mix_fraction_active),
        "heuristic_public_mix_fraction_active": float(heuristic_public_mix_fraction_active),
        "heuristic_public_variant_mix_fraction_active": float(heuristic_public_variant_mix_fraction_active),
        "warmup_snapshot_mix_fraction_active": float(warmup_snapshot_mix_fraction_active),
        "pfsp_pool_size": float(pfsp_pool_size),
        "pfsp_quarantined_opponents": float(pfsp_quarantined_opponents),
        "pfsp_champion_pool_size": float(pfsp_champion_pool_size),
        "pfsp_recent_pool_size": float(pfsp_recent_pool_size),
        "pfsp_hard_negative_pool_size": float(pfsp_hard_negative_pool_size),
        "pfsp_sampled_envs": float(counter_totals.get("pfsp_sampled_envs", pfsp_last_sampled_envs)),
        "pfsp_mirror_envs": float(counter_totals.get("pfsp_mirror_envs", pfsp_last_mirror_envs)),
        "pfsp_heuristic_public_envs": float(
            counter_totals.get("pfsp_heuristic_public_envs", pfsp_last_heuristic_public_envs)
        ),
        "pfsp_heuristic_public_variant_envs": float(
            counter_totals.get("pfsp_heuristic_public_variant_envs", pfsp_last_heuristic_public_variant_envs)
        ),
        "pfsp_noleague_baseline_envs": float(
            counter_totals.get("pfsp_noleague_baseline_envs", pfsp_last_noleague_baseline_envs)
        ),
        "pfsp_champion_envs": float(counter_totals.get("pfsp_champion_envs", pfsp_last_champion_envs)),
        "pfsp_recent_envs": float(counter_totals.get("pfsp_recent_envs", pfsp_last_recent_envs)),
        "pfsp_hard_negative_envs": float(counter_totals.get("pfsp_hard_negative_envs", pfsp_last_hard_negative_envs)),
        "pfsp_warmup_snapshot_envs": float(
            counter_totals.get("pfsp_warmup_snapshot_envs", pfsp_last_warmup_snapshot_envs)
        ),
        "pfsp_epoch": float(pfsp_epoch),
        "tactical_row_count": tactical_rows,
        "packed_candidate_count": packed_candidates,
        "copied_bytes_estimate": float(counter_totals.get("copied_bytes_estimate", 0.0)),
        "avg_legal_actions_per_row": float(packed_candidates / max(row_count_total, 1.0)),
        **{f"collector_{key}": value for key, value in counter_totals.items()},
        **outcome_metrics,
    }
    for key, value in counter_totals.items():
        if key.startswith("simulator_") and key.endswith("_ns"):
            metrics[f"timer_{key[:-3]}_ms"] = float(value / 1_000_000.0)
        elif key.startswith("simulator_python_"):
            metrics[f"timer_{key}_ms"] = float(value / 1_000_000.0)
    return metrics, next_cumulative_env_steps
