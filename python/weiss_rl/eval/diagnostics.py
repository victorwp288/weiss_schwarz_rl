"""Diagnostic reports for seat-swapped evaluation records and leakage checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from weiss_rl.core.masking import masked_log_softmax
from weiss_rl.eval.harness import EvalGameRecord
from weiss_rl.eval.payoff_folding import paired_seed_group_key, paired_seed_scores

__all__ = [
    "HiddenInfoLeakagePair",
    "build_hidden_info_leakage_diagnostics",
    "build_seat_advantage_diagnostics",
    "write_leakage_diagnostics_json",
    "write_matchup_diagnostics_json",
]

_DEFAULT_KL_MEDIAN_THRESHOLD = 0.02
_DEFAULT_KL_P95_THRESHOLD = 0.05
_DEFAULT_TV_MEDIAN_THRESHOLD = 0.05
_DEFAULT_TV_P95_THRESHOLD = 0.1


@dataclass(frozen=True, slots=True)
class HiddenInfoLeakagePair:
    public_observation_a: ArrayLike
    public_observation_b: ArrayLike
    logits_a: ArrayLike
    logits_b: ArrayLike
    legal_mask_a: ArrayLike
    legal_mask_b: ArrayLike
    pair_id: int | str | None = None


def build_seat_advantage_diagnostics(records: tuple[EvalGameRecord, ...] | list[EvalGameRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("seat diagnostics require at least one record")

    paired_seed_scores(records, scheme="S0")
    focal_policy_id, opponent_policy_id = _matchup_ids(records)
    policy_ids = (focal_policy_id, opponent_policy_id)
    policy_breakdown = {
        policy_id: {
            "games_as_seat0": 0,
            "games_as_seat1": 0,
            "wins_as_seat0": 0,
            "wins_as_seat1": 0,
            "total_wins": 0,
        }
        for policy_id in policy_ids
    }

    seat0_wins = 0
    seat1_wins = 0
    draws = 0
    truncations = 0
    engine_errors = 0

    for record in records:
        policy_breakdown[record.seat0_policy_id]["games_as_seat0"] += 1
        policy_breakdown[record.seat1_policy_id]["games_as_seat1"] += 1
        if int(record.engine_status) != 0:
            engine_errors += 1

        winner_seat = _winner_seat(record)
        if winner_seat is None:
            if record.outcome == "D":
                draws += 1
            else:
                truncations += 1
            continue

        if winner_seat == 0:
            seat0_wins += 1
            policy_breakdown[record.seat0_policy_id]["wins_as_seat0"] += 1
            policy_breakdown[record.seat0_policy_id]["total_wins"] += 1
        else:
            seat1_wins += 1
            policy_breakdown[record.seat1_policy_id]["wins_as_seat1"] += 1
            policy_breakdown[record.seat1_policy_id]["total_wins"] += 1

    decisive_games = seat0_wins + seat1_wins
    total_games = decisive_games + draws + truncations
    return {
        "focal_policy_id": focal_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "pair_count": len({paired_seed_group_key(record) for record in records}),
        "seat_results": {
            "seat0_wins": seat0_wins,
            "seat1_wins": seat1_wins,
            "draws": draws,
            "truncations": truncations,
            "engine_errors": engine_errors,
            "decisive_games": decisive_games,
            "total_games": total_games,
            "seat0_win_rate": (seat0_wins / decisive_games) if decisive_games else None,
            "seat1_win_rate": (seat1_wins / decisive_games) if decisive_games else None,
        },
        "policy_breakdown": policy_breakdown,
    }


def build_hidden_info_leakage_diagnostics(
    pairs: tuple[HiddenInfoLeakagePair, ...] | list[HiddenInfoLeakagePair],
    *,
    kl_median_threshold: float = _DEFAULT_KL_MEDIAN_THRESHOLD,
    kl_p95_threshold: float = _DEFAULT_KL_P95_THRESHOLD,
    tv_median_threshold: float = _DEFAULT_TV_MEDIAN_THRESHOLD,
    tv_p95_threshold: float = _DEFAULT_TV_P95_THRESHOLD,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("leakage diagnostics require at least one pair")

    per_pair: list[dict[str, Any]] = []
    kl_values: list[float] = []
    tv_values: list[float] = []

    for pair_index, pair in enumerate(pairs):
        pair_label = pair_index if pair.pair_id is None else pair.pair_id
        _require_public_parity(pair.public_observation_a, pair.public_observation_b, pair_label=pair_label)

        legal_mask_a = _coerce_pair_mask(pair.legal_mask_a, name="legal_mask_a")
        legal_mask_b = _coerce_pair_mask(pair.legal_mask_b, name="legal_mask_b")
        if legal_mask_a.shape != legal_mask_b.shape or not np.array_equal(legal_mask_a, legal_mask_b):
            raise ValueError(f"public parity violation for pair {pair_label}: legal masks differ")
        if not np.any(legal_mask_a):
            raise ValueError(f"leakage diagnostics require at least one legal action for pair {pair_label}")

        logits_a = _coerce_pair_logits(pair.logits_a, expected_shape=legal_mask_a.shape, name="logits_a")
        logits_b = _coerce_pair_logits(pair.logits_b, expected_shape=legal_mask_a.shape, name="logits_b")
        kl_divergence_ab, kl_divergence_ba, total_variation = _distribution_distances(
            logits_a,
            logits_b,
            legal_mask_a,
        )
        worst_kl = max(kl_divergence_ab, kl_divergence_ba)

        kl_values.append(worst_kl)
        tv_values.append(total_variation)
        per_pair.append(
            {
                "pair_id": pair_label,
                "kl_divergence": worst_kl,
                "kl_divergence_ab": kl_divergence_ab,
                "kl_divergence_ba": kl_divergence_ba,
                "total_variation": total_variation,
            }
        )

    summary: dict[str, Any] = {
        "pair_count": len(per_pair),
        "public_parity_passed": True,
        "kl_median": _percentile(kl_values, 50),
        "kl_p95": _percentile(kl_values, 95),
        "tv_median": _percentile(tv_values, 50),
        "tv_p95": _percentile(tv_values, 95),
        "max_kl_divergence": float(np.max(kl_values)),
        "max_total_variation": float(np.max(tv_values)),
    }
    thresholds: dict[str, float] = {
        "kl_median_threshold": float(kl_median_threshold),
        "kl_p95_threshold": float(kl_p95_threshold),
        "tv_median_threshold": float(tv_median_threshold),
        "tv_p95_threshold": float(tv_p95_threshold),
    }
    threshold_failures = [
        name
        for name, passed in (
            ("kl_median", summary["kl_median"] <= thresholds["kl_median_threshold"]),
            ("kl_p95", summary["kl_p95"] <= thresholds["kl_p95_threshold"]),
            ("tv_median", summary["tv_median"] <= thresholds["tv_median_threshold"]),
            ("tv_p95", summary["tv_p95"] <= thresholds["tv_p95_threshold"]),
        )
        if not passed
    ]
    summary["thresholds_passed"] = not threshold_failures
    summary["threshold_failures"] = threshold_failures

    return {
        "pairs": per_pair,
        "summary": summary,
        "thresholds": thresholds,
    }


def write_matchup_diagnostics_json(path: Path, payload: dict[str, Any]) -> None:
    _write_diagnostics_json(path, payload)


def write_leakage_diagnostics_json(path: Path, payload: dict[str, Any]) -> None:
    _write_diagnostics_json(path, payload)


def _write_diagnostics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _matchup_ids(records: tuple[EvalGameRecord, ...] | list[EvalGameRecord]) -> tuple[str, str]:
    focal_ids = {record.focal_policy_id for record in records}
    opponent_ids = {record.opponent_policy_id for record in records}
    if len(focal_ids) != 1 or len(opponent_ids) != 1:
        raise ValueError("seat diagnostics expect records for exactly one focal/opponent matchup")
    return next(iter(focal_ids)), next(iter(opponent_ids))


def _winner_seat(record: EvalGameRecord) -> int | None:
    if record.outcome == "W":
        return int(record.focal_seat)
    if record.outcome == "L":
        return 1 - int(record.focal_seat)
    return None


def _require_public_parity(
    public_observation_a: ArrayLike,
    public_observation_b: ArrayLike,
    *,
    pair_label: int | str,
) -> None:
    public_a = np.asarray(public_observation_a)
    public_b = np.asarray(public_observation_b)
    if public_a.shape != public_b.shape or not np.array_equal(public_a, public_b):
        raise ValueError(f"public parity violation for pair {pair_label}: public observations differ")


def _coerce_pair_logits(logits: ArrayLike, *, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if logits_array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return logits_array


def _coerce_pair_mask(legal_mask: ArrayLike, *, name: str) -> np.ndarray:
    legal_mask_array = np.asarray(legal_mask)
    if legal_mask_array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return legal_mask_array != 0


def _distribution_distances(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    legal_mask: np.ndarray,
) -> tuple[float, float, float]:
    stacked_logits = np.stack((logits_a, logits_b), axis=0)
    stacked_mask = np.stack((legal_mask, legal_mask), axis=0)
    log_probs = masked_log_softmax(stacked_logits, stacked_mask)
    safe_log_probs = np.where(stacked_mask, log_probs, 0.0)
    probs = np.exp(safe_log_probs.astype(np.float64, copy=False))

    kl_divergence_ab = float(np.sum(probs[0] * (safe_log_probs[0] - safe_log_probs[1]), dtype=np.float64))
    kl_divergence_ba = float(np.sum(probs[1] * (safe_log_probs[1] - safe_log_probs[0]), dtype=np.float64))
    total_variation = float(0.5 * np.sum(np.abs(probs[0] - probs[1]), dtype=np.float64))
    return kl_divergence_ab, kl_divergence_ba, total_variation


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
