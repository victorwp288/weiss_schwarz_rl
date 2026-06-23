from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from weiss_rl.config import load_study_config
from weiss_rl.metagame import build_sensitivity_report

from .metagame_sensitivity_test_support import STUDY_CONFIG_PATH, write_final_eval_fixture


def test_build_sensitivity_report_writes_case_artifacts_and_deltas(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    out_dir = final_eval_dir / "sensitivity"
    payload = build_sensitivity_report(
        final_eval_dir=final_eval_dir,
        out_dir=out_dir,
        metagame_config=study.metagame,
        sensitivity_config=study.sensitivity,
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
    with (out_dir / "deltas" / "S2" / "alpharank_sensitivity_delta_vs_s0.csv").open("r", encoding="utf-8") as handle:
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


def test_build_sensitivity_report_supports_global_selection_alpharank(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    global_metagame = replace(
        study.metagame,
        alpharank=replace(study.metagame.alpharank, local_selection=False),
    )

    out_dir = final_eval_dir / "sensitivity_global"
    payload = build_sensitivity_report(
        final_eval_dir=final_eval_dir,
        out_dir=out_dir,
        metagame_config=global_metagame,
        sensitivity_config=study.sensitivity,
    )

    assert payload["alpharank_selection_mode"] == "global"
    summary = json.loads((out_dir / "S0" / "alpharank" / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection_mode"] == "global"
    assert (out_dir / "S2" / "alpharank" / "stationary_mean.csv").is_file()
