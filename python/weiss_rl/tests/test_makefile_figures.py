from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_make_figures_target_requires_run_dir() -> None:
    result = subprocess.run(
        ["make", "figures"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert 'Usage: make figures RUN_DIR=runs/<run_dir> [FIG_ID=seat_bias] [FORMATS="pdf png"]' in message


def test_make_figures_target_forwards_run_dir_fig_id_and_formats() -> None:
    result = subprocess.run(
        ["make", "-n", "figures", "RUN_DIR=runs/synthetic", "FIG_ID=seat_bias", "FORMATS=pdf png"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "python/scripts/make_figures.py" in result.stdout
    assert '--run-dir "runs/synthetic"' in result.stdout
    assert '--fig-id "seat_bias"' in result.stdout
    assert "--format pdf --format png" in result.stdout
    assert "--out" not in result.stdout
