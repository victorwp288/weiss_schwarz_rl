"""Artifact writers for metagame sensitivity reports."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.config.models import SensitivityCaseConfig
from weiss_rl.eval.analysis.payoff_folding import PayoffFoldScheme


@dataclass(frozen=True, slots=True)
class SensitivityCaseArtifacts:
    """Computed samples and paths for one sensitivity case."""

    case_id: str
    description: str
    scheme: PayoffFoldScheme
    case_dir: Path
    policy_ids: tuple[str, ...]
    p_mean: np.ndarray
    u_mean: np.ndarray
    nash_samples: np.ndarray
    alpharank_samples: np.ndarray
    payoff_rows: tuple[dict[str, Any], ...]


def write_case_artifacts(
    *,
    out_dir: Path,
    case_id: str,
    case_config: SensitivityCaseConfig,
    scheme: PayoffFoldScheme,
    alpharank_selection_mode: str,
    policy_ids: Sequence[str],
    p_mean: np.ndarray,
    u_mean: np.ndarray,
    payoff_rows: Sequence[dict[str, Any]],
    nash_samples: np.ndarray,
    alpharank_samples: np.ndarray,
    top_shift_limit: int,
) -> None:
    """Write payoff, Nash, AlphaRank, and case summary artifacts."""
    payoff_dir = out_dir / "payoff"
    nash_dir = out_dir / "nash"
    alpharank_dir = out_dir / "alpharank"

    write_matrix_csv(payoff_dir / "p_mean.csv", policy_ids, p_mean)
    write_matrix_json(payoff_dir / "p_mean.json", policy_ids, p_mean)
    write_matrix_csv(payoff_dir / "u_mean.csv", policy_ids, u_mean)
    write_matrix_json(payoff_dir / "u_mean.json", policy_ids, u_mean)
    write_rows_csv(payoff_dir / "matchups.csv", payoff_rows)

    nash_mean = np.mean(nash_samples, axis=0)
    nash_rows = [
        {
            "policy_id": policy_id,
            "mean_mixture": float(nash_mean[index]),
            "prob_mass_gt_0_05": float(np.mean(nash_samples[:, index] > 0.05)),
        }
        for index, policy_id in enumerate(policy_ids)
    ]
    nash_rows.sort(key=lambda row: (-as_float(row["mean_mixture"]), str(row["policy_id"])))
    write_rows_csv(nash_dir / "mixture_mean.csv", nash_rows)
    write_json(
        nash_dir / "mixture_samples.json",
        {
            "policy_ids": list(policy_ids),
            "sample_count": int(nash_samples.shape[0]),
            "values": [[float(value) for value in row] for row in nash_samples],
        },
    )
    write_json(
        nash_dir / "summary.json",
        {
            "case_id": case_id,
            "scheme": scheme,
            "top_policies_by_mean_mixture": nash_rows[:top_shift_limit],
        },
    )

    alpharank_mean = np.mean(alpharank_samples, axis=0)
    alpharank_rows = [
        {
            "policy_id": policy_id,
            "mean_stationary_mass": float(alpharank_mean[index]),
        }
        for index, policy_id in enumerate(policy_ids)
    ]
    alpharank_rows.sort(key=lambda row: (-as_float(row["mean_stationary_mass"]), str(row["policy_id"])))
    write_rows_csv(alpharank_dir / "stationary_mean.csv", alpharank_rows)
    write_json(
        alpharank_dir / "stationary_samples.json",
        {
            "policy_ids": list(policy_ids),
            "sample_count": int(alpharank_samples.shape[0]),
            "values": [[float(value) for value in row] for row in alpharank_samples],
        },
    )
    write_json(
        alpharank_dir / "summary.json",
        {
            "case_id": case_id,
            "scheme": scheme,
            "selection_mode": alpharank_selection_mode,
            "top_policies_by_stationary_mass": alpharank_rows[:top_shift_limit],
        },
    )

    write_json(
        out_dir / "summary.json",
        {
            "case_id": case_id,
            "description": case_config.description,
            "scheme": scheme,
            "policy_ids": list(policy_ids),
            "top_payoff_shifts_ready": True,
            "neutral_fallback_matchup_count": sum(1 for row in payoff_rows if not row["has_payoff_samples"]),
        },
    )


def write_delta_artifacts(
    *,
    out_dir: Path,
    summary_root: Path,
    baseline: SensitivityCaseArtifacts,
    cases: Mapping[str, SensitivityCaseArtifacts],
    top_shift_limit: int,
) -> dict[str, dict[str, str]]:
    """Write case-vs-S0 delta reports and return their summary paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict[str, str]] = {}
    baseline_nash_mean = np.mean(baseline.nash_samples, axis=0)
    baseline_alpharank_mean = np.mean(baseline.alpharank_samples, axis=0)
    baseline_payoffs = {
        (str(row["focal_policy_id"]), str(row["opponent_policy_id"])): row for row in baseline.payoff_rows
    }

    for case_id, artifacts in cases.items():
        if case_id == baseline.case_id:
            continue
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        nash_mean = np.mean(artifacts.nash_samples, axis=0)
        nash_rows = [
            {
                "policy_id": policy_id,
                "baseline_case_id": baseline.case_id,
                "case_id": case_id,
                "s0_mean_mixture": float(baseline_nash_mean[index]),
                "case_mean_mixture": float(nash_mean[index]),
                "delta_mean_mixture": stable_delta(float(nash_mean[index]), float(baseline_nash_mean[index])),
                "abs_delta_mean_mixture": abs(stable_delta(float(nash_mean[index]), float(baseline_nash_mean[index]))),
            }
            for index, policy_id in enumerate(artifacts.policy_ids)
        ]
        nash_rows.sort(key=lambda row: (-as_float(row["abs_delta_mean_mixture"]), str(row["policy_id"])))
        write_rows_csv(case_dir / "nash_sensitivity_delta_vs_s0.csv", nash_rows)

        alpharank_mean = np.mean(artifacts.alpharank_samples, axis=0)
        alpharank_rows = [
            {
                "policy_id": policy_id,
                "baseline_case_id": baseline.case_id,
                "case_id": case_id,
                "s0_mean_stationary_mass": float(baseline_alpharank_mean[index]),
                "case_mean_stationary_mass": float(alpharank_mean[index]),
                "delta_mean_stationary_mass": stable_delta(
                    float(alpharank_mean[index]),
                    float(baseline_alpharank_mean[index]),
                ),
                "abs_delta_mean_stationary_mass": abs(
                    stable_delta(float(alpharank_mean[index]), float(baseline_alpharank_mean[index]))
                ),
            }
            for index, policy_id in enumerate(artifacts.policy_ids)
        ]
        alpharank_rows.sort(key=lambda row: (-as_float(row["abs_delta_mean_stationary_mass"]), str(row["policy_id"])))
        write_rows_csv(case_dir / "alpharank_sensitivity_delta_vs_s0.csv", alpharank_rows)

        payoff_rows: list[dict[str, Any]] = []
        for row in artifacts.payoff_rows:
            focal_policy_id = str(row["focal_policy_id"])
            opponent_policy_id = str(row["opponent_policy_id"])
            if focal_policy_id == opponent_policy_id:
                continue
            baseline_row = baseline_payoffs[(focal_policy_id, opponent_policy_id)]
            delta = stable_delta(float(row["p_ij_mean"]), float(baseline_row["p_ij_mean"]))
            payoff_rows.append(
                {
                    "baseline_case_id": baseline.case_id,
                    "case_id": case_id,
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                    "s0_p_ij_mean": float(baseline_row["p_ij_mean"]),
                    "case_p_ij_mean": float(row["p_ij_mean"]),
                    "delta_p_ij_mean": delta,
                    "abs_delta_p_ij_mean": abs(delta),
                    "s0_has_payoff_samples": bool(baseline_row["has_payoff_samples"]),
                    "case_has_payoff_samples": bool(row["has_payoff_samples"]),
                }
            )
        payoff_rows.sort(
            key=lambda row: (
                -as_float(row["abs_delta_p_ij_mean"]),
                str(row["focal_policy_id"]),
                str(row["opponent_policy_id"]),
            )
        )
        write_rows_csv(case_dir / "largest_matchup_pij_shifts.csv", payoff_rows)
        write_json(
            case_dir / "summary.json",
            {
                "case_id": case_id,
                "baseline_case_id": baseline.case_id,
                "top_nash_mixture_deltas": nash_rows[:top_shift_limit],
                "top_alpharank_mass_deltas": alpharank_rows[:top_shift_limit],
                "top_matchup_pij_shifts": payoff_rows[:top_shift_limit],
            },
        )
        payload[case_id] = {
            "nash_sensitivity_delta_vs_s0": relative_to(
                case_dir / "nash_sensitivity_delta_vs_s0.csv",
                root=summary_root,
            ),
            "alpharank_sensitivity_delta_vs_s0": relative_to(
                case_dir / "alpharank_sensitivity_delta_vs_s0.csv",
                root=summary_root,
            ),
            "largest_matchup_pij_shifts": relative_to(case_dir / "largest_matchup_pij_shifts.csv", root=summary_root),
            "summary_json": relative_to(case_dir / "summary.json", root=summary_root),
        }
    return payload


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, policy_ids: Sequence[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for policy_id, row in zip(policy_ids, values.tolist(), strict=True):
            writer.writerow([policy_id, *row])


def write_matrix_json(path: Path, policy_ids: Sequence[str], values: np.ndarray) -> None:
    write_json(
        path,
        {
            "policy_ids": list(policy_ids),
            "values": [[float(value) for value in row] for row in values.tolist()],
        },
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_to(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def as_float(value: Any) -> float:
    return float(value)


def stable_delta(value: float, baseline: float, *, tol: float = 1.0e-12) -> float:
    delta = value - baseline
    if abs(delta) <= tol:
        return 0.0
    return float(delta)


__all__ = [
    "SensitivityCaseArtifacts",
    "relative_to",
    "stable_delta",
    "write_case_artifacts",
    "write_delta_artifacts",
    "write_json",
]
