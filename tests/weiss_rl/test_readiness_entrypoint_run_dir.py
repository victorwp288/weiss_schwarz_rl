from __future__ import annotations

import json
from pathlib import Path

from .readiness_entrypoint_test_support import run_readiness_entrypoint, write_readiness_run_dir_fixture


def test_paper_readiness_entrypoint_accepts_run_dir_and_writes_run_summary(tmp_path: Path) -> None:
    run_dir = write_readiness_run_dir_fixture(tmp_path)
    readiness_json = run_dir / "paper_readiness_summary.json"

    result = run_readiness_entrypoint("--run-dir", str(run_dir), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["scope"] == "run_dir"
    assert payload["passed"] is True
    assert payload["run_directory_audit"]["passed"] is True
    assert payload["manifest_contract"]["passed"] is True
    assert payload["final_eval_artifact_contract"]["passed"] is True
