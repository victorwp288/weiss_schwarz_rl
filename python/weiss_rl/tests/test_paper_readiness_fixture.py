from __future__ import annotations

from pathlib import Path

from weiss_rl.artifact_contract import (
    CANONICAL_RUN_ROOT_FILES,
    CANONICAL_RUN_TREE_DIRS,
    REQUIRED_SENSITIVITY_CASE_IDS,
    build_run_directory_audit,
    required_run_artifact_specs,
)
from weiss_rl.eval import build_paper_readiness_summary
from weiss_rl.eval.paper_readiness_fixture import write_paper_readiness_run_fixture


def test_artifact_contract_exports_canonical_doc_contract() -> None:
    assert Path("manifest.json") in CANONICAL_RUN_ROOT_FILES
    assert Path("paper_readiness_summary.json") in CANONICAL_RUN_ROOT_FILES
    assert Path("eval/final_eval") in CANONICAL_RUN_TREE_DIRS
    assert Path("figures/paper") in CANONICAL_RUN_TREE_DIRS


def test_required_run_artifact_specs_keep_thesis_facing_ids() -> None:
    specs = required_run_artifact_specs()
    artifact_ids = [spec.artifact_id for spec in specs]
    specs_by_id = {spec.artifact_id: spec for spec in specs}

    assert artifact_ids[:3] == ["run_manifest", "spec_bundle_json", "spec_hash_txt"]
    assert "final_eval_artifact_hashes" in artifact_ids
    assert "diagnostics_replay_verification" in artifact_ids
    assert specs_by_id["final_eval_payoff_matrix_export"].paths == (Path("eval/final_eval/payoff_matrices/p_mean.csv"),)
    assert specs_by_id["final_eval_payoff_matrix_export"].compatibility_paths == (
        Path("eval/final_eval/matrices/mean.csv"),
    )
    for case_id in REQUIRED_SENSITIVITY_CASE_IDS:
        assert f"sensitivity_{case_id.lower()}_summary" in artifact_ids
        assert specs_by_id[f"sensitivity_{case_id.lower()}_summary"].compatibility_paths == (
            Path(f"eval/final_eval/sensitivity/{case_id}/summary.json"),
        )


def test_write_paper_readiness_run_fixture_satisfies_readiness_contract(tmp_path: Path) -> None:
    run_dir = write_paper_readiness_run_fixture(tmp_path / "run_ready")

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is True
    assert payload["run_directory_audit"]["passed"] is True
    assert payload["manifest_contract"]["passed"] is True
    assert payload["final_eval_artifact_contract"]["passed"] is True


def test_build_run_directory_audit_reports_missing_contract_artifacts(tmp_path: Path) -> None:
    run_dir = write_paper_readiness_run_fixture(tmp_path / "run_ready")
    (run_dir / "eval" / "diagnostics" / "replay_verification.json").unlink()

    audit = build_run_directory_audit(run_dir)

    assert audit["passed"] is False
    assert "diagnostics_replay_verification" in audit["missing_artifacts"]
    missing_check = audit["artifacts"]["diagnostics_replay_verification"]
    assert missing_check["expected_paths"] == ["eval/diagnostics/replay_verification.json"]
