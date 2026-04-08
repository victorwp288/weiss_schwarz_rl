from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from weiss_rl.eval import EvalGameRecord
from weiss_rl.metagame.uncertainty import (
    bayesian_bootstrap_summary,
    paired_seed_uncertainty_summary,
    posterior_samples,
    write_posterior_samples,
    write_uncertainty_summary_json,
    write_uncertainty_artifacts,
)

_CONFIG_HASH256 = "ab" * 32
_SPEC_HASH256 = "cd" * 32
OutcomeToken = Literal["W", "L", "D", "T"]


def _pair(pair_index: int, outcome_a: OutcomeToken, outcome_b: OutcomeToken) -> list[EvalGameRecord]:
    episode_seed = pair_index + 100
    return [
        _record(pair_index, 0, outcome_a, episode_seed=episode_seed),
        _record(pair_index, 1, outcome_b, episode_seed=episode_seed),
    ]


def _record(
    pair_index: int,
    swap_index: int,
    outcome: OutcomeToken,
    *,
    episode_seed: int | None = None,
    focal_policy_id: str = "champion",
    opponent_policy_id: str = "baseline",
) -> EvalGameRecord:
    normalized_swap_index = int(swap_index)
    if normalized_swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1

    episode_seed_value = pair_index if episode_seed is None else episode_seed
    episode_index = pair_index * 2 + normalized_swap_index
    episode_key64 = episode_index + 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=normalized_swap_index,
        episode_index=episode_index,
        episode_seed=episode_seed_value,
        episode_key=f"{episode_key64:064x}",
        episode_key64=episode_key64,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id,
        seat1_policy_id=seat1_policy_id,
        focal_seat=focal_seat,
        outcome=outcome,
        terminated=outcome != "T",
        truncated=outcome == "T",
        engine_status=0,
    )


def test_bayesian_bootstrap_summary_reports_mean_and_probs() -> None:
    summary = bayesian_bootstrap_summary([0.25, 0.5, 1.0], sample_count=8, ci_level=0.8, seed=123)

    assert summary.mean == pytest.approx(0.5833333333333334)
    assert summary.ci_low == pytest.approx(0.40789420874755616)
    assert summary.ci_high == pytest.approx(0.709689696309074)
    assert summary.ci_half_width == pytest.approx(0.1508977437807589)
    assert summary.prob_gt_half == pytest.approx(0.375)
    assert summary.prob_lt_half == pytest.approx(0.625)


def test_paired_seed_uncertainty_summary_computes_from_records() -> None:
    records = [*_pair(0, "W", "L"), *_pair(1, "W", "W")]
    summary = paired_seed_uncertainty_summary(records, scheme="S0", sample_count=8, seed=42)

    assert summary.paired_seed_count == 2
    assert summary.mean == pytest.approx(0.75)
    assert summary.sample_count == 8


def test_paired_seed_uncertainty_summary_splits_reused_pair_index_by_episode_seed() -> None:
    records = [
        *_pair(0, "W", "L"),
        _record(0, 0, "W", episode_seed=250),
        _record(0, 1, "W", episode_seed=250),
    ]

    summary = paired_seed_uncertainty_summary(records, scheme="S0", sample_count=8, seed=42)

    assert summary.paired_seed_count == 2
    assert summary.mean == pytest.approx(0.75)
    assert summary.sample_count == 8


def test_write_posterior_samples_and_summary_json_directly(tmp_path: Path) -> None:
    samples = posterior_samples([0.0, 1.0], sample_count=4, seed=7)
    summary = bayesian_bootstrap_summary([0.0, 1.0], sample_count=4, seed=7)

    samples_path = tmp_path / "posterior_samples.npz"
    summary_path = tmp_path / "uncertainty_summary.json"

    write_posterior_samples(samples_path, samples)
    write_uncertainty_summary_json(summary_path, summary)

    loaded_npz = np.load(samples_path)
    assert "posterior_samples" in loaded_npz
    assert loaded_npz["posterior_samples"].tolist() == samples.tolist()

    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["mean"] == pytest.approx(0.5)
    assert payload["paired_seed_count"] == 2
    assert payload["sample_count"] == 4


def test_write_uncertainty_artifacts_creates_npz_and_json(tmp_path: Path) -> None:
    samples = posterior_samples([0.0, 1.0], sample_count=4, seed=7)
    summary = bayesian_bootstrap_summary([0.0, 1.0], sample_count=4, seed=7)

    samples_path = tmp_path / "payoff_posterior_samples.npz"
    summary_path = tmp_path / "payoff_uncertainty_summary.json"
    write_uncertainty_artifacts(samples_path, summary_path, summary=summary, samples=samples)

    loaded = np.load(samples_path)
    assert "posterior_samples" in loaded
    assert loaded["posterior_samples"].shape == (4,)
    assert loaded["posterior_samples"].tolist() == samples.tolist()

    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["mean"] == pytest.approx(0.5)
    assert payload["paired_seed_count"] == 2
    assert payload["sample_count"] == 4
