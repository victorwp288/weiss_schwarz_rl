from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.eval import build_paper_readiness_summary
from weiss_rl.eval.policies.set import RANDOM_LEGAL_POLICY_ID

from tests.weiss_rl.paper_readiness_test_support import write_json, write_run_dir_fixture


def test_build_paper_readiness_summary_audits_run_directory_artifacts(tmp_path: Path) -> None:
    run_dir = write_run_dir_fixture(tmp_path)

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["scope"] == "run_dir"
    assert payload["passed"] is True
    assert payload["alarms"] == []
    assert payload["section_plan"]["final_eval_guardrails"]["title"] == "Final-eval guardrails"
    assert payload["run_directory_audit"]["passed"] is True
    assert payload["run_directory_audit"]["artifact_groups"]["diagnostics"]["title"] == "Diagnostics"
    assert payload["manifest_contract"]["passed"] is True
    assert payload["final_eval_artifact_contract"]["passed"] is True
    assert payload["final_eval_guardrails"]["passed"] is True
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_id"] == "policy_000300"


def test_build_paper_readiness_summary_accepts_interpolation_provenance_instead_of_training_metrics(
    tmp_path: Path,
) -> None:
    run_dir = write_run_dir_fixture(tmp_path)
    (run_dir / "training" / "logs" / "training_metrics.jsonl").unlink()
    write_json(
        run_dir / "eval" / "diagnostics" / "checkpoint_interpolation_summary.json",
        {
            "first_checkpoint": "runs/source_a/training/checkpoints/checkpoint_10.pt",
            "second_checkpoint": "runs/source_b/training/checkpoints/checkpoint_5.pt",
            "second_weight": 0.15,
        },
    )

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is True
    assert payload["run_directory_audit"]["artifacts"]["training_metrics"]["resolved_path"] == (
        "eval/diagnostics/checkpoint_interpolation_summary.json"
    )


def test_build_paper_readiness_summary_accepts_documented_unresolved_manifest_policy_selection(tmp_path: Path) -> None:
    run_dir = write_run_dir_fixture(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policy_set_selection"] = []
    manifest["policy_set_selection_details"] = {
        "status": "unresolved",
        "version": "deterministic_v1",
        "final_policy_set_size": 10,
        "source_paths": {
            "snapshot_registry_json": None,
            "dev_eval_summaries_json": None,
        },
        "missing_inputs": ["snapshot_registry_json", "dev_eval_summaries_json"],
        "reason": "deterministic final policy set inputs were not provided",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is False
    assert "manifest_contract" in payload["alarms"]
    assert payload["manifest_contract"]["passed"] is False
    assert payload["manifest_contract"]["fields"]["policy_set_selection"]["passed"] is False


def test_build_paper_readiness_summary_reports_out_of_range_matchup_indices(tmp_path: Path) -> None:
    run_dir = write_run_dir_fixture(tmp_path)
    summary_path = run_dir / "eval" / "final_eval" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["matchups"][0]["focal_policy_index"] = len(summary["policy_ids"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_paper_readiness_summary(run_dir=run_dir)
    contract = payload["final_eval_artifact_contract"]

    assert payload["passed"] is False
    assert payload["alarms"] == ["final_eval_artifact_contract", "final_eval_guardrails"]
    assert contract["passed"] is False
    assert contract["reason"] == "invalid_matchup_index"
    assert contract["reference_failures"] == [
        "matchups[0].focal_policy_index=2 is out of range for policy_ids with length 2"
    ]


def test_build_paper_readiness_summary_reports_negative_matchup_indices(tmp_path: Path) -> None:
    run_dir = write_run_dir_fixture(tmp_path)
    summary_path = run_dir / "eval" / "final_eval" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["matchups"][0]["opponent_policy_index"] = -1
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_paper_readiness_summary(run_dir=run_dir)
    contract = payload["final_eval_artifact_contract"]

    assert payload["passed"] is False
    assert payload["alarms"] == ["final_eval_artifact_contract", "final_eval_guardrails"]
    assert contract["passed"] is False
    assert contract["reason"] == "invalid_matchup_index"
    assert contract["reference_failures"] == [
        "matchups[0].opponent_policy_index=-1 is out of range for policy_ids with length 2"
    ]


def test_build_paper_readiness_summary_flags_run_directory_gaps(tmp_path: Path) -> None:
    run_dir = write_run_dir_fixture(tmp_path)
    (run_dir / "eval" / "diagnostics" / "replay_verification.json").unlink()
    write_json(
        run_dir / "eval" / "final_eval" / "policy_set.json",
        {"policy_ids": ["policy_000300", RANDOM_LEGAL_POLICY_ID]},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["policy_set_selection"] = []
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_paper_readiness_summary(run_dir=run_dir)

    assert payload["passed"] is False
    assert payload["alarms"] == [
        "run_directory_audit",
        "manifest_contract",
        "final_eval_artifact_contract",
    ]
    assert "diagnostics_replay_verification" in payload["run_directory_audit"]["missing_artifacts"]
    assert payload["manifest_contract"]["fields"]["policy_set_selection"]["passed"] is False
    assert payload["final_eval_artifact_contract"]["policy_set"]["passed"] is False
