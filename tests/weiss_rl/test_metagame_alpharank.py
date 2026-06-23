from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from weiss_rl.metagame import compute_alpharank_stationary, write_alpharank_artifacts, write_stationary_mean_csv


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return list(reader)


def _fixation_probability_2x2(*, resident_advantage: float, m: int, alpha: int) -> float:
    delta = -resident_advantage
    if delta == 0.0:
        return 1.0 / float(m)
    exponents = -(float(alpha) * delta) * np.arange(m, dtype=np.float64)
    max_exponent = float(np.max(exponents))
    denom = float(np.sum(np.exp(exponents - max_exponent), dtype=np.float64))
    return float(np.exp(-(max_exponent + np.log(denom))))


def test_compute_alpharank_stationary_basic() -> None:
    p_mean = np.array(
        [
            [0.5, 0.6],
            [0.4, 0.5],
        ],
        dtype=np.float64,
    )

    stationary = compute_alpharank_stationary(
        p_mean,
        policy_ids=["policy0", "policy1"],
        m=50,
        alpha=100,
        local_selection=True,
        use_inf_alpha=False,
        inf_alpha_eps=0.01,
    )

    assert stationary.shape == (2,)
    assert float(np.sum(stationary, dtype=np.float64)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(stationary >= 0.0)
    assert stationary[0] > stationary[1]


def test_equal_margin_cycle_has_uniform_stationary_distribution() -> None:
    p_mean = np.array(
        [
            [0.5, 0.6, 0.4],
            [0.4, 0.5, 0.6],
            [0.6, 0.4, 0.5],
        ],
        dtype=np.float64,
    )

    stationary = compute_alpharank_stationary(p_mean, m=50, alpha=100)

    assert stationary.tolist() == pytest.approx([1.0 / 3.0] * 3, abs=1e-12)


def test_policy_order_permutation_only_permutates_outputs() -> None:
    p_mean = np.array(
        [
            [0.5, 0.7, 0.2],
            [0.3, 0.5, 0.8],
            [0.8, 0.2, 0.5],
        ],
        dtype=np.float64,
    )
    permutation = np.array([2, 0, 1], dtype=np.int64)
    permuted_p_mean = p_mean[np.ix_(permutation, permutation)]

    stationary = compute_alpharank_stationary(p_mean, m=30, alpha=15)
    permuted_stationary = compute_alpharank_stationary(permuted_p_mean, m=30, alpha=15)

    assert permuted_stationary.tolist() == pytest.approx(stationary[permutation].tolist(), abs=1e-12)


def test_compute_alpharank_stationary_global_selection_basic() -> None:
    p_mean = np.array(
        [
            [0.7, 0.55, 0.45],
            [0.45, 0.6, 0.65],
            [0.55, 0.35, 0.5],
        ],
        dtype=np.float64,
    )

    stationary = compute_alpharank_stationary(p_mean, m=20, alpha=10, local_selection=False)

    assert stationary.shape == (3,)
    assert float(np.sum(stationary, dtype=np.float64)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(stationary >= 0.0)


def test_global_selection_uses_self_play_terms_and_can_differ_from_local_selection() -> None:
    p_mean = np.array(
        [
            [0.9, 0.6],
            [0.6, 0.1],
        ],
        dtype=np.float64,
    )

    local_stationary = compute_alpharank_stationary(p_mean, m=25, alpha=8, local_selection=True)
    global_stationary = compute_alpharank_stationary(p_mean, m=25, alpha=8, local_selection=False)

    assert local_stationary.tolist() == pytest.approx([0.5, 0.5], abs=1e-12)
    assert global_stationary[0] > global_stationary[1]
    assert global_stationary.tolist() != pytest.approx(local_stationary.tolist(), abs=1e-6)


def test_global_selection_permutation_only_permutates_outputs() -> None:
    p_mean = np.array(
        [
            [0.8, 0.5, 0.2],
            [0.4, 0.6, 0.9],
            [0.7, 0.1, 0.3],
        ],
        dtype=np.float64,
    )
    permutation = np.array([1, 2, 0], dtype=np.int64)
    permuted_p_mean = p_mean[np.ix_(permutation, permutation)]

    stationary = compute_alpharank_stationary(p_mean, m=12, alpha=5, local_selection=False)
    permuted_stationary = compute_alpharank_stationary(
        permuted_p_mean,
        m=12,
        alpha=5,
        local_selection=False,
    )

    assert permuted_stationary.tolist() == pytest.approx(stationary[permutation].tolist(), abs=1e-12)


def test_compute_alpharank_stationary_matches_known_two_policy_reference() -> None:
    p_mean = np.array(
        [
            [0.5, 0.6],
            [0.4, 0.5],
        ],
        dtype=np.float64,
    )
    m = 5
    alpha = 2

    stationary = compute_alpharank_stationary(p_mean, m=m, alpha=alpha)

    rho_01 = _fixation_probability_2x2(resident_advantage=0.2, m=m, alpha=alpha)
    rho_10 = _fixation_probability_2x2(resident_advantage=-0.2, m=m, alpha=alpha)
    expected = np.array(
        [rho_10 / (rho_01 + rho_10), rho_01 / (rho_01 + rho_10)],
        dtype=np.float64,
    )

    assert stationary.tolist() == pytest.approx(expected.tolist(), abs=1e-12)


def test_use_inf_alpha_uses_limiting_fixation_behavior() -> None:
    p_mean = np.array(
        [
            [0.5, 0.7],
            [0.3, 0.5],
        ],
        dtype=np.float64,
    )

    stationary = compute_alpharank_stationary(p_mean, m=7, alpha=1, use_inf_alpha=True, inf_alpha_eps=0.0)

    assert stationary.tolist() == pytest.approx([1.0, 0.0], abs=1e-12)


def test_write_stationary_mean_csv_sorts_descending_with_policy_id_tie_break(tmp_path: Path) -> None:
    path = tmp_path / "stationary_mean.csv"
    stationary = np.array([0.5, 0.5, 0.2], dtype=np.float64)
    policy_ids = ["policy_b", "policy_a", "policy_c"]

    write_stationary_mean_csv(path, policy_ids, stationary)

    assert _read_csv_rows(path) == [
        ["rank", "policy_id", "stationary_probability"],
        ["1", "policy_a", "0.5"],
        ["2", "policy_b", "0.5"],
        ["3", "policy_c", "0.2"],
    ]


def test_write_alpharank_artifacts_writes_ranked_csv(tmp_path: Path) -> None:
    path = tmp_path / "alpharank" / "stationary_mean.csv"
    stationary = np.array([0.3, 0.7], dtype=np.float64)
    policy_ids = ["policy_a", "policy_b"]

    write_alpharank_artifacts(path, stationary, policy_ids)

    assert _read_csv_rows(path) == [
        ["rank", "policy_id", "stationary_probability"],
        ["1", "policy_b", "0.7"],
        ["2", "policy_a", "0.3"],
    ]


def test_compute_alpharank_stationary_invalid_input() -> None:
    with pytest.raises(ValueError, match="p_mean must be a square matrix"):
        compute_alpharank_stationary(np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="p_mean must contain only finite values"):
        compute_alpharank_stationary(np.array([[np.nan, 0.5], [0.5, 0.5]]))

    with pytest.raises(ValueError, match="policy_ids length must match p_mean dimensions"):
        compute_alpharank_stationary(np.eye(2), policy_ids=["a", "b", "c"])

    with pytest.raises(ValueError, match="m must be >= 1"):
        compute_alpharank_stationary(np.eye(2), m=0)

    with pytest.raises(ValueError, match="inf_alpha_eps must be >= 0"):
        compute_alpharank_stationary(np.eye(2), use_inf_alpha=True, inf_alpha_eps=-1.0)
