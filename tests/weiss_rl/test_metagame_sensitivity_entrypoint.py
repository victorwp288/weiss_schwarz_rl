from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .metagame_sensitivity_test_support import REPO_ROOT, STUDY_CONFIG_PATH, write_final_eval_fixture


def test_metagame_entrypoint_writes_sensitivity_tree(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    out_dir = tmp_path / "custom_sensitivity"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.metagame.metagame_entrypoint",
            "--study-config",
            str(STUDY_CONFIG_PATH),
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
