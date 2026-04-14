from __future__ import annotations

from pathlib import Path

from weiss_rl.eval import build_paper_readiness_summary
from weiss_rl.eval.paper_readiness_fixture import write_paper_readiness_run_fixture


def test_write_paper_readiness_run_fixture_satisfies_readiness_contract(tmp_path: Path) -> None:
    run_dir = write_paper_readiness_run_fixture(tmp_path / "run_ready")

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is True
    assert payload["run_directory_audit"]["passed"] is True
    assert payload["manifest_contract"]["passed"] is True
    assert payload["final_eval_artifact_contract"]["passed"] is True
