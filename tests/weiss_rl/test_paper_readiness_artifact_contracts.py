from __future__ import annotations

from pathlib import Path

import weiss_rl.eval.readiness.contracts as paper_readiness_contracts
import weiss_rl.eval.readiness.final_eval_summary as paper_readiness_final_eval_summary
from weiss_rl.eval.policies.set import RANDOM_LEGAL_POLICY_ID


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
