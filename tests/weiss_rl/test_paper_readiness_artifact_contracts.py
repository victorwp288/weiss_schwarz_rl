from __future__ import annotations

import json
from pathlib import Path

import weiss_rl.eval.readiness.contracts as paper_readiness_contracts
import weiss_rl.eval.readiness.final_eval_summary as paper_readiness_final_eval_summary
from weiss_rl.eval.policies.set import RANDOM_LEGAL_POLICY_ID

from .paper_readiness_test_support import write_json, write_run_dir_fixture


def test_paper_readiness_final_eval_summary_selects_canonical_unordered_matchups() -> None:
    policies = [RANDOM_LEGAL_POLICY_ID, "policy_000300", "policy_000400"]
    raw_matchups = [
        {
            "focal_policy_id": "policy_000300",
            "opponent_policy_id": RANDOM_LEGAL_POLICY_ID,
            "focal_policy_index": 1,
            "opponent_policy_index": 0,
            "diagnostics_path": "reciprocal.json",
        },
        {
            "focal_policy_id": RANDOM_LEGAL_POLICY_ID,
            "opponent_policy_id": "policy_000300",
            "focal_policy_index": 0,
            "opponent_policy_index": 1,
            "diagnostics_path": "canonical.json",
        },
        {
            "focal_policy_id": "policy_000400",
            "opponent_policy_id": "policy_000400",
            "focal_policy_index": 2,
            "opponent_policy_index": 2,
            "diagnostics_path": "self.json",
        },
    ]

    selected = paper_readiness_final_eval_summary.canonical_unordered_matchups(raw_matchups, policy_ids=policies)

    assert [matchup["diagnostics_path"] for matchup in selected] == ["canonical.json", "self.json"]


def test_paper_readiness_final_eval_summary_reads_reader_section_keys() -> None:
    payload = {
        "summary_sections": [
            {"key": "metadata.selection", "question": "Which policies?"},
            {"key": "matrices", "question": "What payoffs?"},
        ]
    }

    assert paper_readiness_final_eval_summary.summary_section_keys(payload) == ["metadata.selection", "matrices"]
    assert paper_readiness_final_eval_summary.summary_section_keys({}) == []


def test_final_eval_artifact_contract_reports_summary_section_keys(tmp_path: Path) -> None:
    run_dir = write_run_dir_fixture(tmp_path)
    final_eval_dir = run_dir / "eval" / "final_eval"
    summary_path = final_eval_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["summary_sections"] = [{"key": "metadata.selection"}, {"key": "matrices"}]
    write_json(summary_path, summary)

    contract = paper_readiness_contracts.build_final_eval_artifact_contract(final_eval_dir)

    assert contract["summary_section_keys"] == ["metadata.selection", "matrices"]


def test_paper_readiness_contract_catalog_keeps_thesis_artifact_surfaces() -> None:
    specs = {spec.artifact_id: spec for spec in paper_readiness_contracts.required_run_artifact_specs()}

    assert specs["training_metrics"].paths == (
        Path("training/logs/training_metrics.jsonl"),
        Path("eval/diagnostics/checkpoint_interpolation_summary.json"),
    )
    assert specs["sensitivity_summary"].paths == (
        Path("eval/metagame/summary.json"),
        Path("eval/final_eval/sensitivity/summary.json"),
    )
    assert specs["sensitivity_s0_nash_mixture"].paths == (
        Path("eval/metagame/S0/nash/mixture_mean.csv"),
        Path("eval/final_eval/sensitivity/S0/nash/mixture_mean.csv"),
    )
    assert specs["paper_figures_pdf"].glob == "figures/paper/*.pdf"
    assert specs["paper_figures_png"].glob == "figures/paper/*.png"


def test_paper_readiness_contract_catalog_groups_artifacts_by_reader_question() -> None:
    specs = paper_readiness_contracts.required_run_artifact_specs()
    group_payload = paper_readiness_contracts.required_run_artifact_group_payload(specs)

    assert tuple(group_payload) == (
        "run_root",
        "training",
        "final_eval",
        "diagnostics",
        "sensitivity",
        "figures",
    )
    assert group_payload["run_root"]["title"] == "Run identity"
    assert group_payload["final_eval"]["artifact_count"] == 6
    assert group_payload["sensitivity"]["artifact_count"] == 13
