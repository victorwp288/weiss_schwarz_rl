from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

from weiss_rl.config import load_stack_config
from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import run_final_eval
from weiss_rl.metagame import build_sensitivity_report
from weiss_rl.tests.test_final_eval import _CONFIG_HASH256, _FakeMatrixRunner, _RUN_ID256, _SPEC_HASH256

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_final_eval_fixture(tmp_path: Path) -> Path:
    final_eval_dir = tmp_path / "final_eval"
    policies = ["policy_gamma", "policy_alpha", "policy_beta"]
    outcomes: dict[tuple[str, str, int, int], Literal["W", "L", "D", "T"]] = {
        ("policy_gamma", "policy_gamma", 0, 0): "D",
        ("policy_gamma", "policy_gamma", 0, 1): "D",
        ("policy_gamma", "policy_gamma", 1, 0): "D",
        ("policy_gamma", "policy_gamma", 1, 1): "D",
        ("policy_gamma", "policy_alpha", 0, 0): "W",
        ("policy_gamma", "policy_alpha", 0, 1): "T",
        ("policy_gamma", "policy_alpha", 1, 0): "W",
        ("policy_gamma", "policy_alpha", 1, 1): "L",
        ("policy_gamma", "policy_beta", 0, 0): "L",
        ("policy_gamma", "policy_beta", 0, 1): "L",
        ("policy_gamma", "policy_beta", 1, 0): "L",
        ("policy_gamma", "policy_beta", 1, 1): "W",
        ("policy_alpha", "policy_alpha", 0, 0): "D",
        ("policy_alpha", "policy_alpha", 0, 1): "D",
        ("policy_alpha", "policy_alpha", 1, 0): "D",
        ("policy_alpha", "policy_alpha", 1, 1): "D",
        ("policy_alpha", "policy_beta", 0, 0): "W",
        ("policy_alpha", "policy_beta", 0, 1): "W",
        ("policy_alpha", "policy_beta", 1, 0): "W",
        ("policy_alpha", "policy_beta", 1, 1): "L",
        ("policy_beta", "policy_beta", 0, 0): "D",
        ("policy_beta", "policy_beta", 0, 1): "D",
        ("policy_beta", "policy_beta", 1, 0): "D",
        ("policy_beta", "policy_beta", 1, 1): "D",
    }
    run_final_eval(
        output_dir=final_eval_dir,
        runner=_FakeMatrixRunner(outcomes),
        policy_ids=policies,
        paired_seeds=[11, 22],
        stage1_paired_seeds=2,
        max_paired_seeds=2,
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        sample_count=8,
    )
    return final_eval_dir


def _write_final_eval_summary(final_eval_dir: Path, payload: dict[str, Any]) -> None:
    (final_eval_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_build_sensitivity_report_writes_case_artifacts_and_deltas(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(tmp_path)
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.metagame is not None
    assert stack.config.sensitivity is not None

    out_dir = final_eval_dir / "sensitivity"
    payload = build_sensitivity_report(
        final_eval_dir=final_eval_dir,
        out_dir=out_dir,
        metagame_config=stack.config.metagame,
        sensitivity_config=stack.config.sensitivity,
    )

    assert payload["policy_ids"] == ["policy_gamma", "policy_alpha", "policy_beta"]
    assert sorted(payload["cases"]) == ["S0", "S1", "S2"]
    assert sorted(payload["deltas"]) == ["S1", "S2"]
    assert payload["deltas"]["S1"]["summary_json"] == "deltas/S1/summary.json"
    assert payload["deltas"]["S2"]["largest_matchup_pij_shifts"] == "deltas/S2/largest_matchup_pij_shifts.csv"
    assert (out_dir / "S0" / "payoff" / "matchups.csv").is_file()
    assert (out_dir / "S2" / "nash" / "mixture_mean.csv").is_file()
    assert (out_dir / "deltas" / "S2" / "summary.json").is_file()

    with (out_dir / "deltas" / "S1" / "nash_sensitivity_delta_vs_s0.csv").open("r", encoding="utf-8") as handle:
        s1_nash_rows = list(csv.DictReader(handle))
    assert {row["policy_id"] for row in s1_nash_rows} == {"policy_gamma", "policy_alpha", "policy_beta"}
    assert all(float(row["abs_delta_mean_mixture"]) == 0.0 for row in s1_nash_rows)

    with (out_dir / "deltas" / "S2" / "nash_sensitivity_delta_vs_s0.csv").open("r", encoding="utf-8") as handle:
        s2_nash_rows = list(csv.DictReader(handle))
    with (out_dir / "deltas" / "S2" / "alpharank_sensitivity_delta_vs_s0.csv").open(
        "r", encoding="utf-8"
    ) as handle:
        s2_alpharank_rows = list(csv.DictReader(handle))
    assert {row["policy_id"] for row in s2_nash_rows} == {"policy_gamma", "policy_alpha", "policy_beta"}
    assert {row["policy_id"] for row in s2_alpharank_rows} == {"policy_gamma", "policy_alpha", "policy_beta"}
    assert any(float(row["abs_delta_mean_mixture"]) > 0.0 for row in s2_nash_rows)
    assert any(float(row["abs_delta_mean_stationary_mass"]) > 0.0 for row in s2_alpharank_rows)

    with (out_dir / "deltas" / "S2" / "largest_matchup_pij_shifts.csv").open("r", encoding="utf-8") as handle:
        payoff_rows = list(csv.DictReader(handle))
    assert payoff_rows[0]["focal_policy_id"] == "policy_gamma"
    assert payoff_rows[0]["opponent_policy_id"] == "policy_alpha"
    assert float(payoff_rows[0]["delta_p_ij_mean"]) > 0.0

    summary = json.loads((out_dir / "deltas" / "S2" / "summary.json").read_text(encoding="utf-8"))
    assert summary["top_matchup_pij_shifts"][0]["focal_policy_id"] == "policy_gamma"
    assert summary["top_nash_mixture_deltas"]
    assert summary["top_alpharank_mass_deltas"]


def test_build_sensitivity_report_rejects_unsupported_case_config(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(tmp_path)
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.metagame is not None
    assert stack.config.sensitivity is not None

    bad_sensitivity = replace(
        stack.config.sensitivity,
        cases={
            **stack.config.sensitivity.cases,
            "S1": replace(stack.config.sensitivity.cases["S1"], draw_score=0.25),
        },
    )

    with pytest.raises(ValueError, match=r"S1 must set draw_score=0\.5"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=stack.config.metagame,
            sensitivity_config=bad_sensitivity,
        )


def test_build_sensitivity_report_rejects_unsupported_nash_contract(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(tmp_path)
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.metagame is not None
    assert stack.config.sensitivity is not None

    bad_metagame = replace(
        stack.config.metagame,
        nash=replace(stack.config.metagame.nash, threads=2),
    )

    with pytest.raises(ValueError, match=r"metagame\.nash\.threads=1"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=bad_metagame,
            sensitivity_config=stack.config.sensitivity,
        )


def test_build_sensitivity_report_rejects_mismatched_final_eval_policy_index(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(tmp_path)
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.metagame is not None
    assert stack.config.sensitivity is not None

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["matchups"][0]["focal_policy_index"] = 2
    _write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=r"focal_policy_index=2 does not match policy_ids position"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=stack.config.metagame,
            sensitivity_config=stack.config.sensitivity,
        )


def test_build_sensitivity_report_rejects_duplicate_final_eval_matchup(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(tmp_path)
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.metagame is not None
    assert stack.config.sensitivity is not None

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["matchups"][-1] = dict(payload["matchups"][0])
    _write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=r"duplicate canonical matchup"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=stack.config.metagame,
            sensitivity_config=stack.config.sensitivity,
        )


def test_build_sensitivity_report_rejects_mismatched_matchup_episodes_path(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(tmp_path)
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.metagame is not None
    assert stack.config.sensitivity is not None

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    target = payload["matchups"][0]
    source = next(
        matchup
        for matchup in payload["matchups"][1:]
        if (
            matchup["focal_policy_id"],
            matchup["opponent_policy_id"],
        )
        != (
            target["focal_policy_id"],
            target["opponent_policy_id"],
        )
    )
    target["episodes_path"] = source["episodes_path"]
    _write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=r"episodes do not match summary metadata"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=stack.config.metagame,
            sensitivity_config=stack.config.sensitivity,
        )


@pytest.mark.parametrize(
    ("episodes_path", "message"),
    [
        ("../outside.jsonl", r"resolves outside the final_eval root"),
        ("/tmp/outside.jsonl", r"must be relative to the final_eval root"),
    ],
)
def test_build_sensitivity_report_rejects_unsafe_matchup_episodes_path(
    tmp_path: Path, episodes_path: str, message: str
) -> None:
    final_eval_dir = _write_final_eval_fixture(tmp_path)
    stack = load_stack_config(REPO_ROOT / "configs/rl_stack_locked.yaml")
    assert stack.config.metagame is not None
    assert stack.config.sensitivity is not None

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["matchups"][0]["episodes_path"] = episodes_path
    _write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=message):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=stack.config.metagame,
            sensitivity_config=stack.config.sensitivity,
        )


def test_metagame_entrypoint_writes_sensitivity_tree(tmp_path: Path) -> None:
    final_eval_dir = _write_final_eval_fixture(tmp_path)
    out_dir = tmp_path / "custom_sensitivity"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "metagame.py"),
            "--stack-config",
            str(REPO_ROOT / "configs" / "rl_stack_locked.yaml"),
            "--final-eval-dir",
            str(final_eval_dir),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Sensitivity summary JSON" in result.stdout
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "S2" / "alpharank" / "stationary_mean.csv").is_file()
    assert (out_dir / "deltas" / "S2" / "largest_matchup_pij_shifts.csv").is_file()
