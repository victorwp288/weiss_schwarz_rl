"""Sensitivity reporting over final-eval metagame artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, stable_hash64
from weiss_rl.config.models import MetagameConfig, SensitivityCaseConfig, SensitivityConfig
from weiss_rl.eval.export import load_eval_game_records
from weiss_rl.eval.harness import EvalGameRecord
from weiss_rl.eval.payoff_folding import PayoffFoldScheme, paired_seed_scores
from weiss_rl.eval.uncertainty import bayesian_bootstrap_posterior_samples
from weiss_rl.metagame.alpharank import compute_stationary_distribution
from weiss_rl.metagame.nash import solve_zero_sum_mixture

_SUPPORT_PROBABILITY_THRESHOLD = 0.05
_TOP_SHIFT_LIMIT = 10
_SUPPORTED_SENSITIVITY_CASES = frozenset({"S0", "S1", "S2"})
_SUPPORTED_NASH_IMPL = "weiss_rl_nash_lp_v1"
_SUPPORTED_NASH_TIE_BREAK = "deterministic_secondary_lp_by_policy_id"

__all__ = [
    "build_sensitivity_report",
]


@dataclass(frozen=True, slots=True)
class FinalEvalMatchup:
    focal_policy_id: str
    opponent_policy_id: str
    focal_policy_index: int
    opponent_policy_index: int
    episodes_path: Path
    records: tuple[EvalGameRecord, ...]


@dataclass(frozen=True, slots=True)
class FinalEvalContext:
    policy_ids: tuple[str, ...]
    matchups: tuple[FinalEvalMatchup, ...]


@dataclass(frozen=True, slots=True)
class CaseArtifacts:
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


def build_sensitivity_report(
    *,
    final_eval_dir: Path,
    out_dir: Path,
    metagame_config: MetagameConfig,
    sensitivity_config: SensitivityConfig,
) -> dict[str, Any]:
    context = _load_final_eval_context(final_eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_supported_nash_config(metagame_config)

    if "S0" not in sensitivity_config.cases:
        raise ValueError("sensitivity config must define S0 for delta baselines")

    case_artifacts: dict[str, CaseArtifacts] = {}
    case_paths: dict[str, dict[str, str]] = {}
    for case_id, case_config in sensitivity_config.cases.items():
        scheme = _resolve_scheme(case_id=case_id, case_config=case_config)
        artifacts = _build_case_artifacts(
            context=context,
            case_id=case_id,
            case_config=case_config,
            scheme=scheme,
            metagame_config=metagame_config,
            out_dir=out_dir / case_id,
        )
        case_artifacts[case_id] = artifacts
        case_paths[case_id] = {
            "summary_json": _relative_to(artifacts.case_dir / "summary.json", root=out_dir),
            "payoff_matchups_csv": _relative_to(artifacts.case_dir / "payoff" / "matchups.csv", root=out_dir),
            "nash_mixture_csv": _relative_to(artifacts.case_dir / "nash" / "mixture_mean.csv", root=out_dir),
            "alpharank_stationary_csv": _relative_to(
                artifacts.case_dir / "alpharank" / "stationary_mean.csv",
                root=out_dir,
            ),
        }

    delta_paths = _write_delta_artifacts(
        out_dir=out_dir / "deltas",
        summary_root=out_dir,
        baseline=case_artifacts["S0"],
        cases=case_artifacts,
    )
    payload = {
        "final_eval_dir": final_eval_dir.as_posix(),
        "out_dir": out_dir.as_posix(),
        "policy_ids": list(context.policy_ids),
        "sample_count": int(metagame_config.sampling_m),
        "alpharank_selection_mode": _alpharank_selection_mode(metagame_config),
        "required_outputs": list(sensitivity_config.report.required_outputs),
        "cases": case_paths,
        "deltas": delta_paths,
    }
    _write_json(out_dir / "summary.json", payload)
    return payload


def _load_final_eval_context(final_eval_dir: Path) -> FinalEvalContext:
    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_ids = tuple(_require_str_list(payload.get("policy_ids"), field_name="policy_ids"))
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("final_eval policy_ids must be unique")
    policy_index = {policy_id: index for index, policy_id in enumerate(policy_ids)}
    raw_matchups = payload.get("matchups")
    if not isinstance(raw_matchups, list):
        raise ValueError("final_eval summary must include a matchups list")

    matchups: list[FinalEvalMatchup] = []
    canonical_keys: set[tuple[int, int]] = set()
    for item in raw_matchups:
        if not isinstance(item, dict):
            raise ValueError("final_eval matchups must contain objects")
        focal_policy_id = _require_matchup_policy_id(
            item,
            field_name="focal_policy_id",
            policy_index=policy_index,
        )
        opponent_policy_id = _require_matchup_policy_id(
            item,
            field_name="opponent_policy_id",
            policy_index=policy_index,
        )
        focal_index = _require_matchup_policy_index(
            item,
            index_field="focal_policy_index",
            policy_id=focal_policy_id,
            policy_index=policy_index,
        )
        opponent_index = _require_matchup_policy_index(
            item,
            index_field="opponent_policy_index",
            policy_id=opponent_policy_id,
            policy_index=policy_index,
        )
        if focal_index > opponent_index:
            raise ValueError("final_eval matchups must be canonical with focal_policy_index <= opponent_policy_index")
        canonical_key = (focal_index, opponent_index)
        if canonical_key in canonical_keys:
            raise ValueError(
                "final_eval summary contains duplicate canonical matchup: "
                f"({focal_policy_id!r}, {opponent_policy_id!r})"
            )
        canonical_keys.add(canonical_key)
        episodes_path = _resolve_final_eval_episodes_path(
            final_eval_dir=final_eval_dir,
            value=item.get("episodes_path"),
            expected_relative_path=_expected_final_eval_episodes_path(
                focal_policy_index=focal_index,
                opponent_policy_index=opponent_index,
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
            ),
        )
        records = load_eval_game_records(episodes_path)
        _validate_matchup_records(
            records=records,
            episodes_path=episodes_path,
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
        )
        matchups.append(
            FinalEvalMatchup(
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                focal_policy_index=focal_index,
                opponent_policy_index=opponent_index,
                episodes_path=episodes_path,
                records=records,
            )
        )
    expected_matchups = (len(policy_ids) * (len(policy_ids) + 1)) // 2
    if len(canonical_keys) != expected_matchups:
        raise ValueError(
            "final_eval summary is missing canonical matchups: "
            f"expected {expected_matchups}, found {len(canonical_keys)}"
        )
    return FinalEvalContext(policy_ids=policy_ids, matchups=tuple(matchups))


def _resolve_scheme(*, case_id: str, case_config: SensitivityCaseConfig) -> PayoffFoldScheme:
    normalized = case_id.strip().upper()
    if normalized not in _SUPPORTED_SENSITIVITY_CASES:
        raise ValueError(f"unsupported sensitivity case: {case_id!r}")
    _validate_supported_case_config(case_id=normalized, case_config=case_config)
    return normalized  # type: ignore[return-value]


def _build_case_artifacts(
    *,
    context: FinalEvalContext,
    case_id: str,
    case_config: SensitivityCaseConfig,
    scheme: PayoffFoldScheme,
    metagame_config: MetagameConfig,
    out_dir: Path,
) -> CaseArtifacts:
    policy_ids = context.policy_ids
    policy_count = len(policy_ids)
    sample_count = int(metagame_config.sampling_m)
    p_mean = np.full((policy_count, policy_count), 0.5, dtype=np.float64)
    p_samples = np.full((sample_count, policy_count, policy_count), 0.5, dtype=np.float64)
    payoff_rows: list[dict[str, Any]] = []

    for matchup in context.matchups:
        scores = paired_seed_scores(matchup.records, scheme=scheme)
        observed_pairs = _observed_pair_count(matchup.records)
        if scores:
            seed = _bootstrap_seed(
                kind="metagame_sensitivity_payoff_v1",
                focal_policy_id=matchup.focal_policy_id,
                opponent_policy_id=matchup.opponent_policy_id,
            )
            samples = np.asarray(
                bayesian_bootstrap_posterior_samples(scores, sample_count=sample_count, seed=seed),
                dtype=np.float64,
            )
            mean = float(np.mean(scores))
            paired_seed_count = len(scores)
            has_payoff_samples = True
        else:
            samples = np.full((sample_count,), 0.5, dtype=np.float64)
            mean = 0.5
            paired_seed_count = 0
            has_payoff_samples = False
        focal_index = matchup.focal_policy_index
        opponent_index = matchup.opponent_policy_index
        p_mean[focal_index, opponent_index] = mean
        p_mean[opponent_index, focal_index] = 1.0 - mean if focal_index != opponent_index else 0.5
        p_samples[:, focal_index, opponent_index] = samples
        p_samples[:, opponent_index, focal_index] = 1.0 - samples if focal_index != opponent_index else 0.5
        payoff_rows.append(
            {
                "case_id": case_id,
                "scheme": scheme,
                "focal_policy_id": matchup.focal_policy_id,
                "opponent_policy_id": matchup.opponent_policy_id,
                "p_ij_mean": mean,
                "p_ji_mean": 1.0 - mean if focal_index != opponent_index else 0.5,
                "utility_ij_mean": (2.0 * mean) - 1.0,
                "utility_ji_mean": 1.0 - (2.0 * mean),
                "observed_paired_seed_count": observed_pairs,
                "paired_seed_count": paired_seed_count,
                "excluded_paired_seed_count": observed_pairs - paired_seed_count,
                "has_payoff_samples": has_payoff_samples,
            }
        )

    np.fill_diagonal(p_mean, 0.5)
    for sample_index in range(sample_count):
        np.fill_diagonal(p_samples[sample_index], 0.5)
    u_mean = (2.0 * p_mean) - 1.0
    np.fill_diagonal(u_mean, 0.0)

    nash_samples = np.zeros((sample_count, policy_count), dtype=np.float64)
    alpharank_samples = np.zeros((sample_count, policy_count), dtype=np.float64)
    for sample_index in range(sample_count):
        utility = (2.0 * p_samples[sample_index]) - 1.0
        np.fill_diagonal(utility, 0.0)
        nash_samples[sample_index] = solve_zero_sum_mixture(
            utility,
            policy_ids=policy_ids,
            backend=metagame_config.nash.backend,
            value_tolerance=metagame_config.nash.value_tolerance,
        ).mixture
        alpharank_samples[sample_index] = compute_stationary_distribution(
            utility,
            m=metagame_config.alpharank.m,
            alpha=metagame_config.alpharank.alpha,
            local_selection=metagame_config.alpharank.local_selection,
            use_inf_alpha=metagame_config.alpharank.use_inf_alpha,
            inf_alpha_eps=metagame_config.alpharank.inf_alpha_eps,
        ).stationary

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_case_artifacts(
        out_dir=out_dir,
        case_id=case_id,
        case_config=case_config,
        scheme=scheme,
        alpharank_selection_mode=_alpharank_selection_mode(metagame_config),
        policy_ids=policy_ids,
        p_mean=p_mean,
        u_mean=u_mean,
        payoff_rows=payoff_rows,
        nash_samples=nash_samples,
        alpharank_samples=alpharank_samples,
    )
    return CaseArtifacts(
        case_id=case_id,
        description=case_config.description,
        scheme=scheme,
        case_dir=out_dir,
        policy_ids=policy_ids,
        p_mean=p_mean,
        u_mean=u_mean,
        nash_samples=nash_samples,
        alpharank_samples=alpharank_samples,
        payoff_rows=tuple(payoff_rows),
    )


def _write_case_artifacts(
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
) -> None:
    payoff_dir = out_dir / "payoff"
    nash_dir = out_dir / "nash"
    alpharank_dir = out_dir / "alpharank"

    _write_matrix_csv(payoff_dir / "p_mean.csv", policy_ids, p_mean)
    _write_matrix_json(payoff_dir / "p_mean.json", policy_ids, p_mean)
    _write_matrix_csv(payoff_dir / "u_mean.csv", policy_ids, u_mean)
    _write_matrix_json(payoff_dir / "u_mean.json", policy_ids, u_mean)
    _write_rows_csv(payoff_dir / "matchups.csv", payoff_rows)

    nash_mean = np.mean(nash_samples, axis=0)
    nash_rows = [
        {
            "policy_id": policy_id,
            "mean_mixture": float(nash_mean[index]),
            "prob_mass_gt_0_05": float(np.mean(nash_samples[:, index] > _SUPPORT_PROBABILITY_THRESHOLD)),
        }
        for index, policy_id in enumerate(policy_ids)
    ]
    nash_rows.sort(key=lambda row: (-_as_float(row["mean_mixture"]), str(row["policy_id"])))
    _write_rows_csv(nash_dir / "mixture_mean.csv", nash_rows)
    _write_json(
        nash_dir / "mixture_samples.json",
        {
            "policy_ids": list(policy_ids),
            "sample_count": int(nash_samples.shape[0]),
            "values": [[float(value) for value in row] for row in nash_samples],
        },
    )
    _write_json(
        nash_dir / "summary.json",
        {
            "case_id": case_id,
            "scheme": scheme,
            "top_policies_by_mean_mixture": nash_rows[:_TOP_SHIFT_LIMIT],
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
    alpharank_rows.sort(key=lambda row: (-_as_float(row["mean_stationary_mass"]), str(row["policy_id"])))
    _write_rows_csv(alpharank_dir / "stationary_mean.csv", alpharank_rows)
    _write_json(
        alpharank_dir / "stationary_samples.json",
        {
            "policy_ids": list(policy_ids),
            "sample_count": int(alpharank_samples.shape[0]),
            "values": [[float(value) for value in row] for row in alpharank_samples],
        },
    )
    _write_json(
        alpharank_dir / "summary.json",
        {
            "case_id": case_id,
            "scheme": scheme,
            "selection_mode": alpharank_selection_mode,
            "top_policies_by_stationary_mass": alpharank_rows[:_TOP_SHIFT_LIMIT],
        },
    )

    _write_json(
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


def _write_delta_artifacts(
    *,
    out_dir: Path,
    summary_root: Path,
    baseline: CaseArtifacts,
    cases: Mapping[str, CaseArtifacts],
) -> dict[str, dict[str, str]]:
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
                "delta_mean_mixture": _stable_delta(float(nash_mean[index]), float(baseline_nash_mean[index])),
                "abs_delta_mean_mixture": abs(_stable_delta(float(nash_mean[index]), float(baseline_nash_mean[index]))),
            }
            for index, policy_id in enumerate(artifacts.policy_ids)
        ]
        nash_rows.sort(key=lambda row: (-_as_float(row["abs_delta_mean_mixture"]), str(row["policy_id"])))
        _write_rows_csv(case_dir / "nash_sensitivity_delta_vs_s0.csv", nash_rows)

        alpharank_mean = np.mean(artifacts.alpharank_samples, axis=0)
        alpharank_rows = [
            {
                "policy_id": policy_id,
                "baseline_case_id": baseline.case_id,
                "case_id": case_id,
                "s0_mean_stationary_mass": float(baseline_alpharank_mean[index]),
                "case_mean_stationary_mass": float(alpharank_mean[index]),
                "delta_mean_stationary_mass": _stable_delta(
                    float(alpharank_mean[index]),
                    float(baseline_alpharank_mean[index]),
                ),
                "abs_delta_mean_stationary_mass": abs(
                    _stable_delta(float(alpharank_mean[index]), float(baseline_alpharank_mean[index]))
                ),
            }
            for index, policy_id in enumerate(artifacts.policy_ids)
        ]
        alpharank_rows.sort(key=lambda row: (-_as_float(row["abs_delta_mean_stationary_mass"]), str(row["policy_id"])))
        _write_rows_csv(case_dir / "alpharank_sensitivity_delta_vs_s0.csv", alpharank_rows)

        payoff_rows: list[dict[str, Any]] = []
        for row in artifacts.payoff_rows:
            focal_policy_id = str(row["focal_policy_id"])
            opponent_policy_id = str(row["opponent_policy_id"])
            if focal_policy_id == opponent_policy_id:
                continue
            baseline_row = baseline_payoffs[(focal_policy_id, opponent_policy_id)]
            delta = _stable_delta(float(row["p_ij_mean"]), float(baseline_row["p_ij_mean"]))
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
                -_as_float(row["abs_delta_p_ij_mean"]),
                str(row["focal_policy_id"]),
                str(row["opponent_policy_id"]),
            )
        )
        _write_rows_csv(case_dir / "largest_matchup_pij_shifts.csv", payoff_rows)
        _write_json(
            case_dir / "summary.json",
            {
                "case_id": case_id,
                "baseline_case_id": baseline.case_id,
                "top_nash_mixture_deltas": nash_rows[:_TOP_SHIFT_LIMIT],
                "top_alpharank_mass_deltas": alpharank_rows[:_TOP_SHIFT_LIMIT],
                "top_matchup_pij_shifts": payoff_rows[:_TOP_SHIFT_LIMIT],
            },
        )
        payload[case_id] = {
            "nash_sensitivity_delta_vs_s0": _relative_to(
                case_dir / "nash_sensitivity_delta_vs_s0.csv",
                root=summary_root,
            ),
            "alpharank_sensitivity_delta_vs_s0": _relative_to(
                case_dir / "alpharank_sensitivity_delta_vs_s0.csv",
                root=summary_root,
            ),
            "largest_matchup_pij_shifts": _relative_to(case_dir / "largest_matchup_pij_shifts.csv", root=summary_root),
            "summary_json": _relative_to(case_dir / "summary.json", root=summary_root),
        }
    return payload


def _validate_supported_nash_config(metagame_config: MetagameConfig) -> None:
    nash_config = metagame_config.nash
    if nash_config.impl != _SUPPORTED_NASH_IMPL:
        raise ValueError(f"unsupported metagame.nash.impl for sensitivity reporting: {nash_config.impl!r}")
    if nash_config.threads != 1:
        raise ValueError(f"sensitivity reporting requires metagame.nash.threads=1, got {nash_config.threads}")
    if nash_config.tie_break != _SUPPORTED_NASH_TIE_BREAK:
        raise ValueError(
            "sensitivity reporting requires metagame.nash.tie_break="
            f"{_SUPPORTED_NASH_TIE_BREAK!r}, got {nash_config.tie_break!r}"
        )


def _alpharank_selection_mode(metagame_config: MetagameConfig) -> str:
    return _alpharank_selection_mode_from_bool(metagame_config.alpharank.local_selection)


def _alpharank_selection_mode_from_bool(local_selection: bool) -> str:
    return "local" if local_selection else "global"


def _validate_supported_case_config(*, case_id: str, case_config: SensitivityCaseConfig) -> None:
    _require_case_float(
        case_id=case_id,
        field_name="draw_score",
        value=case_config.draw_score,
        expected=0.5,
    )

    if case_id in {"S0", "S1"}:
        if case_config.truncation_score is None:
            raise ValueError(f"{case_id} must set truncation_score=0.5")
        _require_case_float(
            case_id=case_id,
            field_name="truncation_score",
            value=case_config.truncation_score,
            expected=0.5,
        )
        if case_config.truncation_handling is not None:
            raise ValueError(f"{case_id} must not set truncation_handling, got {case_config.truncation_handling!r}")
        return

    if case_config.truncation_score is not None:
        raise ValueError(f"{case_id} must not set truncation_score, got {case_config.truncation_score}")
    if case_config.truncation_handling != "exclude_from_payoff_aggregation":
        raise ValueError(f"{case_id} must set truncation_handling='exclude_from_payoff_aggregation'")


def _require_case_float(*, case_id: str, field_name: str, value: float, expected: float, tol: float = 1.0e-12) -> None:
    if abs(float(value) - expected) > tol:
        raise ValueError(f"{case_id} must set {field_name}={expected}, got {value}")


def _require_matchup_policy_id(
    item: Mapping[str, Any],
    *,
    field_name: str,
    policy_index: Mapping[str, int],
) -> str:
    value = item.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"final_eval matchup {field_name} must be a non-empty string")
    if value not in policy_index:
        raise ValueError(f"final_eval matchup {field_name}={value!r} is missing from policy_ids")
    return value


def _require_matchup_policy_index(
    item: Mapping[str, Any],
    *,
    index_field: str,
    policy_id: str,
    policy_index: Mapping[str, int],
) -> int:
    if index_field not in item:
        raise ValueError(f"final_eval matchup missing {index_field}")
    raw_value = item[index_field]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"final_eval matchup {index_field} must be an integer")
    index = int(raw_value)
    expected_index = policy_index[policy_id]
    if index != expected_index:
        raise ValueError(
            f"final_eval matchup {index_field}={index} does not match policy_ids position "
            f"for {policy_id!r} (expected {expected_index})"
        )
    return index


def _expected_final_eval_episodes_path(
    *,
    focal_policy_index: int,
    opponent_policy_index: int,
    focal_policy_id: str,
    opponent_policy_id: str,
) -> Path:
    matchup_dir = (
        f"{focal_policy_index:02d}_{_slug(focal_policy_id)}__vs__"
        f"{opponent_policy_index:02d}_{_slug(opponent_policy_id)}"
    )
    return Path("matchups") / matchup_dir / "episodes.jsonl"


def _slug(value: str) -> str:
    parts = [
        "".join(char.lower() for char in chunk if char.isalnum())
        for chunk in str(value).replace("-", " ").replace("_", " ").split()
    ]
    slug = "_".join(part for part in parts if part)
    return slug or "policy"


def _resolve_final_eval_episodes_path(*, final_eval_dir: Path, value: Any, expected_relative_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("final_eval matchup episodes_path must be a non-empty string")
    raw_path = Path(value)
    if raw_path.is_absolute() or value.startswith(("/", "\\")):
        raise ValueError(
            f"final_eval matchup episodes_path must be relative to the final_eval root, got absolute path: {value!r}"
        )
    resolved_root = final_eval_dir.resolve()
    resolved_path = (final_eval_dir / raw_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"final_eval matchup episodes_path resolves outside the final_eval root: {value!r}") from exc
    expected = expected_relative_path.as_posix()
    if value != expected:
        raise ValueError(
            "final_eval matchup episodes_path must equal canonical final_eval artifact path: "
            f"expected {expected!r}, got {value!r}"
        )
    return resolved_path


def _validate_matchup_records(
    *,
    records: Sequence[EvalGameRecord],
    episodes_path: Path,
    focal_policy_id: str,
    opponent_policy_id: str,
) -> None:
    focal_ids = {record.focal_policy_id for record in records}
    opponent_ids = {record.opponent_policy_id for record in records}
    if len(focal_ids) != 1 or len(opponent_ids) != 1:
        raise ValueError(
            "final_eval matchup episodes must contain exactly one focal_policy_id and "
            f"one opponent_policy_id: {episodes_path}"
        )
    actual_focal_policy_id = next(iter(focal_ids))
    actual_opponent_policy_id = next(iter(opponent_ids))
    if actual_focal_policy_id != focal_policy_id or actual_opponent_policy_id != opponent_policy_id:
        raise ValueError(
            "final_eval matchup episodes do not match summary metadata for "
            f"{episodes_path}: summary expects ({focal_policy_id!r}, {opponent_policy_id!r}), "
            f"loaded ({actual_focal_policy_id!r}, {actual_opponent_policy_id!r})"
        )


def _observed_pair_count(records: Sequence[EvalGameRecord]) -> int:
    return len({record.pair_index for record in records})


def _as_float(value: Any) -> float:
    return float(value)


def _stable_delta(value: float, baseline: float, *, tol: float = 1.0e-12) -> float:
    delta = value - baseline
    if abs(delta) <= tol:
        return 0.0
    return float(delta)


def _bootstrap_seed(*, kind: str, focal_policy_id: str, opponent_policy_id: str) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": kind,
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
            }
        )
    )


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(path: Path, policy_ids: Sequence[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for policy_id, row in zip(policy_ids, values.tolist(), strict=True):
            writer.writerow([policy_id, *row])


def _write_matrix_json(path: Path, policy_ids: Sequence[str], values: np.ndarray) -> None:
    _write_json(
        path,
        {
            "policy_ids": list(policy_ids),
            "values": [[float(value) for value in row] for row in values.tolist()],
        },
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_to(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _require_str_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return [str(item) for item in value]
