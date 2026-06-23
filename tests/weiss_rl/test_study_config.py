from __future__ import annotations

from pathlib import Path

from weiss_rl.config import load_study_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_study_config_reads_metagame_and_sensitivity_defaults() -> None:
    study = load_study_config(_repo_root() / "configs" / "study" / "metagame_sensitivity.yaml")

    assert study.schema_version == 1
    assert study.metagame.sampling_m == 1000
    assert study.metagame.nash.backend == "scipy_linprog_highs"
    assert study.sensitivity.cases["S0"].truncation_score == 0.5
    assert study.sensitivity.report.required_outputs == (
        "nash_sensitivity_delta_vs_s0",
        "alpharank_sensitivity_delta_vs_s0",
        "largest_matchup_pij_shifts",
    )
