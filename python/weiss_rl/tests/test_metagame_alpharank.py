from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from weiss_rl.metagame import compute_alpharank_stationary, write_stationary_mean_csv, write_alpharank_artifacts


def _assert_csv_rows(path: Path, expected_rows: list[list[str]]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert rows == expected_rows


def test_compute_alpharank_stationary_basic() -> None:
    # Simple 2x2 payoff matrix where policy 0 dominates
    p_mean = np.array(
        [
            [0.5, 0.6],
            [0.4, 0.5],
        ],
        dtype=np.float64,
    )
    policy_ids = ["policy0", "policy1"]

    stationary = compute_alpharank_stationary(
        p_mean,
        policy_ids=policy_ids,
        m=50,
        alpha=100,
        local_selection=True,
        use_inf_alpha=False,
        inf_alpha_eps=0.01,
    )

    assert stationary.shape == (2,)
    assert np.sum(stationary) == pytest.approx(1.0, abs=1e-12)
    assert np.all(stationary >= 0)
    # Policy 0 should have higher probability
    assert stationary[0] > stationary[1]


def test_compute_alpharank_stationary_uniform() -> None:
    # Symmetric payoff matrix should lead to uniform distribution
    p_mean = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    policy_ids = ["policy0", "policy1"]

    stationary = compute_alpharank_stationary(p_mean, policy_ids=policy_ids)

    assert stationary.shape == (2,)
    assert np.sum(stationary) == pytest.approx(1.0, abs=1e-12)
    assert stationary.tolist() == pytest.approx([0.5, 0.5], abs=1e-8)


def test_write_stationary_mean_csv() -> None:
    stationary = np.array([0.3, 0.7])
    policy_ids = ["policy_a", "policy_b"]
    path = Path("test_stationary.csv")

    try:
        write_stationary_mean_csv(path, policy_ids, stationary)

        _assert_csv_rows(
            path,
            [
                ["policy_id", "stationary_probability"],
                ["policy_a", "0.3"],
                ["policy_b", "0.7"],
            ],
        )
    finally:
        path.unlink(missing_ok=True)


def test_write_alpharank_artifacts() -> None:
    stationary = np.array([0.3, 0.7])
    policy_ids = ["policy_a", "policy_b"]
    path = Path("test_alpharank_stationary.csv")

    try:
        write_alpharank_artifacts(path, stationary, policy_ids)

        _assert_csv_rows(
            path,
            [
                ["policy_id", "stationary_probability"],
                ["policy_a", "0.3"],
                ["policy_b", "0.7"],
            ],
        )
    finally:
        path.unlink(missing_ok=True)


def test_compute_alpharank_stationary_invalid_input() -> None:
    # Test error cases
    with pytest.raises(ValueError, match="p_mean must be a square matrix"):
        compute_alpharank_stationary(np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="p_mean must not contain NaN values"):
        compute_alpharank_stationary(np.array([[np.nan, 0.5], [0.5, 0.5]]))

    with pytest.raises(ValueError, match="policy_ids length must match p_mean dimensions"):
        compute_alpharank_stationary(np.eye(2), policy_ids=["a", "b", "c"])