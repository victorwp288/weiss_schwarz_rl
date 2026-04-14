from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_thesis_run_wrapper_dry_run_writes_plan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    stack_config = repo_root / "configs" / "stack.yaml"
    stack_config.write_text("components: []\nconfig: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "thesis_run.py"),
            "--repo-root",
            str(repo_root),
            "--stack-config",
            str(stack_config),
            "--run-label",
            "demo_run",
            "--dry-run",
            "--compare-run-dir",
            str(repo_root / "runs" / "baseline_a"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary_path = repo_root / "runs" / "_wrapper_plans" / "demo_run.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "planned"
    assert len(payload["steps"]) == 3
    assert payload["steps"][0]["command"][1] == "python/scripts/train.py"
    assert payload["steps"][1]["command"][1] == "python/scripts/eval.py"
    assert payload["steps"][2]["command"][1] == "python/scripts/compare_runs.py"
