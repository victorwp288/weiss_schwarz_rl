from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

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
