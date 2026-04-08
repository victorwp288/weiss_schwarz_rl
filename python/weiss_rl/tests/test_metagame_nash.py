from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from weiss_rl.metagame import NashSolverReport, solve_nash_mixture, write_nash_artifacts


def _assert_csv_rows(path: Path, expected_rows: list[list[str]]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    assert rows == expected_rows


def test_solve_nash_mixture_returns_uniform_rps_equilibrium() -> None:
    p_mean = np.array(
        [
            [0.5, 0.75, 0.25],
            [0.25, 0.5, 0.75],
            [0.75, 0.25, 0.5],
        ],
        dtype=np.float64,
    )
    policy_ids = ["rock", "paper", "scissors"]

    mixture, report = solve_nash_mixture(
        p_mean,
        policy_ids=policy_ids,
        value_tolerance=1e-9,
        tie_break="lowest_policy_id",
        threads=1,
    )

    assert report.success is True
    assert report.solver == "linprog"
    assert report.backend == "highs"
    assert report.tie_break == "lowest_policy_id"
    assert report.threads == 1
    assert report.policy_ids == tuple(policy_ids)
    assert report.value == pytest.approx(0.5, abs=1e-8)
    assert report.actual_game_value == pytest.approx(0.5, abs=1e-8)

    assert mixture.shape == (3,)
    assert mixture.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-8)


def test_solve_nash_mixture_deterministic_tie_break_by_policy_id() -> None:
    p_mean = np.full((2, 2), 0.5, dtype=np.float64)
    policy_ids = ["zulu", "alpha"]

    mixture, report = solve_nash_mixture(
        p_mean,
        policy_ids=policy_ids,
        value_tolerance=1e-9,
        tie_break="lowest_policy_id",
        threads=1,
    )

    assert report.success is True
    assert report.policy_ids == tuple(policy_ids)
    assert mixture.tolist() == pytest.approx([0.0, 1.0], abs=1e-8)
    assert report.message


def test_write_nash_artifacts_writes_csv_and_json(tmp_path: Path) -> None:
    mixture = np.array([0.2, 0.8], dtype=np.float64)
    policy_ids = ["champion", "baseline"]
    report = NashSolverReport(
        solver="linprog",
        backend="highs",
        status=0,
        success=True,
        message="optimal",
        value=0.75,
        actual_game_value=0.75,
        mixture=(0.2, 0.8),
        policy_ids=tuple(policy_ids),
        threads=1,
        tie_break="lowest_policy_id",
        value_tolerance=1e-9,
        bias_scale=1e-12,
        max_inequality_violation=0.0,
        max_equality_violation=0.0,
    )

    csv_path = tmp_path / "mixture_mean.csv"
    json_path = tmp_path / "solver_report.json"
    write_nash_artifacts(csv_path, json_path, mixture, report, policy_ids)

    _assert_csv_rows(csv_path, [["policy_id", "mixture"], ["champion", "0.2"], ["baseline", "0.8"]])

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["solver"] == "linprog"
    assert payload["backend"] == "highs"
    assert payload["mixture"] == [0.2, 0.8]
    assert payload["policy_ids"] == policy_ids
