from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from weiss_rl.config import load_study_config
from weiss_rl.metagame import build_sensitivity_report

from .metagame_sensitivity_test_support import STUDY_CONFIG_PATH, write_final_eval_fixture


def test_build_sensitivity_report_rejects_unsupported_case_config(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    bad_sensitivity = replace(
        study.sensitivity,
        cases={
            **study.sensitivity.cases,
            "S1": replace(study.sensitivity.cases["S1"], draw_score=0.25),
        },
    )

    with pytest.raises(ValueError, match=r"S1 must set draw_score=0\.5"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=study.metagame,
            sensitivity_config=bad_sensitivity,
        )


def test_build_sensitivity_report_rejects_unsupported_nash_contract(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    bad_metagame = replace(
        study.metagame,
        nash=replace(study.metagame.nash, threads=2),
    )

    with pytest.raises(ValueError, match=r"metagame\.nash\.threads=1"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=bad_metagame,
            sensitivity_config=study.sensitivity,
        )
