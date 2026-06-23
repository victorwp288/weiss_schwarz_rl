from __future__ import annotations

import weiss_rl.eval.paper_readiness as paper_readiness
import weiss_rl.eval.readiness.check_entrypoint as paper_readiness_check_entrypoint
import weiss_rl.eval.readiness.contracts as paper_readiness_contracts
import weiss_rl.eval.readiness.final_eval_summary as paper_readiness_final_eval_summary
import weiss_rl.eval.readiness.guardrails as paper_readiness_guardrails


def test_eval_package_root_does_not_export_paper_readiness_module_aliases() -> None:
    import weiss_rl.eval as eval_package

    for name in (
        "paper_readiness_check_cli",
        "paper_readiness_check_entrypoint",
        "paper_readiness_check_reporting",
        "paper_readiness_check_runtime",
        "paper_readiness_contracts",
        "paper_readiness_fields",
        "paper_readiness_final_eval_summary",
        "paper_readiness_fixture",
        "paper_readiness_fixture_cli",
        "paper_readiness_fixture_entrypoint",
        "paper_readiness_fixture_writer",
        "paper_readiness_guardrails",
    ):
        assert not hasattr(eval_package, name)


def test_paper_readiness_facade_keeps_contract_helpers_aliased() -> None:
    assert paper_readiness.RequiredArtifactSpec is paper_readiness_contracts.RequiredArtifactSpec
    assert paper_readiness._build_run_directory_audit is paper_readiness_contracts.build_run_directory_audit
    assert paper_readiness._build_manifest_contract is paper_readiness_contracts.build_manifest_contract
    assert paper_readiness._build_final_eval_artifact_contract is (
        paper_readiness_contracts.build_final_eval_artifact_contract
    )


def test_paper_readiness_guardrails_keep_final_eval_summary_helpers_aliased() -> None:
    assert paper_readiness_guardrails.policy_ids is paper_readiness_final_eval_summary.policy_ids
    assert paper_readiness_guardrails.matchups is paper_readiness_final_eval_summary.matchups
    assert paper_readiness_guardrails.matchup_policy_index is paper_readiness_final_eval_summary.matchup_policy_index
    assert paper_readiness_guardrails.matrix_cell is paper_readiness_final_eval_summary.matrix_cell
    assert paper_readiness_guardrails.posterior_samples is paper_readiness_final_eval_summary.posterior_samples


def test_paper_readiness_check_entrypoint_exposes_only_cli_boundary() -> None:
    assert hasattr(paper_readiness_check_entrypoint, "main")
    assert not hasattr(paper_readiness_check_entrypoint, "_closed_interval")
    assert not hasattr(paper_readiness_check_entrypoint, "_default_readiness_json")
    assert not hasattr(paper_readiness_check_entrypoint, "_format_alarm")
    assert not hasattr(paper_readiness_check_entrypoint, "_format_alarm_detail")
