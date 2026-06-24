"""Sensitivity reporting over final-eval metagame artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, stable_hash64
from weiss_rl.config.models import MetagameConfig, SensitivityCaseConfig, SensitivityConfig
from weiss_rl.eval.analysis.payoff_folding import PayoffFoldScheme, paired_seed_scores
from weiss_rl.eval.analysis.uncertainty import bayesian_bootstrap_posterior_samples
from weiss_rl.metagame.alpharank import compute_stationary_distribution
from weiss_rl.metagame.nash import solve_zero_sum_mixture
from weiss_rl.metagame.sensitivity_inputs import FinalEvalContext, load_final_eval_context, observed_pair_count
from weiss_rl.metagame.sensitivity_outputs import (
    SensitivityCaseArtifacts,
    relative_to,
    write_case_artifacts,
    write_delta_artifacts,
    write_json,
)

_TOP_SHIFT_LIMIT = 10
_SUPPORTED_SENSITIVITY_CASES = frozenset({"S0", "S1", "S2"})
_SUPPORTED_NASH_IMPL = "weiss_rl_nash_lp_v1"
_SUPPORTED_NASH_TIE_BREAK = "deterministic_secondary_lp_by_policy_id"

__all__ = [
    "build_sensitivity_report",
]


def build_sensitivity_report(
    *,
    final_eval_dir: Path,
    out_dir: Path,
    metagame_config: MetagameConfig,
    sensitivity_config: SensitivityConfig,
) -> dict[str, Any]:
    context = load_final_eval_context(final_eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_supported_nash_config(metagame_config)

    if "S0" not in sensitivity_config.cases:
        raise ValueError("sensitivity config must define S0 for delta baselines")

    case_artifacts: dict[str, SensitivityCaseArtifacts] = {}
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
            "summary_json": relative_to(artifacts.case_dir / "summary.json", root=out_dir),
            "payoff_matchups_csv": relative_to(artifacts.case_dir / "payoff" / "matchups.csv", root=out_dir),
            "nash_mixture_csv": relative_to(artifacts.case_dir / "nash" / "mixture_mean.csv", root=out_dir),
            "alpharank_stationary_csv": relative_to(
                artifacts.case_dir / "alpharank" / "stationary_mean.csv",
                root=out_dir,
            ),
        }

    delta_paths = write_delta_artifacts(
        out_dir=out_dir / "deltas",
        summary_root=out_dir,
        baseline=case_artifacts["S0"],
        cases=case_artifacts,
        top_shift_limit=_TOP_SHIFT_LIMIT,
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
    write_json(out_dir / "summary.json", payload)
    return payload


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
) -> SensitivityCaseArtifacts:
    policy_ids = context.policy_ids
    policy_count = len(policy_ids)
    sample_count = int(metagame_config.sampling_m)
    p_mean = np.full((policy_count, policy_count), 0.5, dtype=np.float64)
    p_samples = np.full((sample_count, policy_count, policy_count), 0.5, dtype=np.float64)
    payoff_rows: list[dict[str, Any]] = []

    for matchup in context.matchups:
        scores = paired_seed_scores(matchup.records, scheme=scheme)
        observed_pairs = observed_pair_count(matchup.records)
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
    write_case_artifacts(
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
        top_shift_limit=_TOP_SHIFT_LIMIT,
    )
    return SensitivityCaseArtifacts(
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
