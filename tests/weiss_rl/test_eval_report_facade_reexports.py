from __future__ import annotations


def test_eval_report_facade_reexports_split_module_owners() -> None:
    import weiss_rl.workflows.eval_support.eval_policy_final_set_resolution as eval_policy_final_set_resolution
    import weiss_rl.workflows.eval_support.eval_policy_manifest_selection as eval_policy_manifest_selection
    import weiss_rl.workflows.eval_support.eval_policy_selection as eval_policy_selection
    import weiss_rl.workflows.eval_support.eval_policy_selection_results as eval_policy_selection_results
    import weiss_rl.workflows.eval_support.eval_report_io as eval_report_io
    import weiss_rl.workflows.eval_support.eval_report_scaffolding as eval_report_scaffolding
    import weiss_rl.workflows.eval_support.eval_report_update_payloads as eval_report_update_payloads
    import weiss_rl.workflows.eval_support.eval_report_updates as eval_report_updates
    import weiss_rl.workflows.eval_support.eval_reports as eval_reports

    assert (
        eval_reports._authoritative_manifest_policy_selection
        is eval_policy_manifest_selection._authoritative_manifest_policy_selection
    )
    assert eval_reports._effective_manifest_git_commit is eval_policy_manifest_selection._effective_manifest_git_commit
    assert (
        eval_reports._persist_policy_selection_in_manifest
        is eval_policy_manifest_selection._persist_policy_selection_in_manifest
    )
    assert eval_reports._policy_selection_mode is eval_policy_manifest_selection._policy_selection_mode
    assert (
        eval_reports._resolve_selection_inputs_from_manifest
        is eval_policy_manifest_selection._resolve_selection_inputs_from_manifest
    )
    assert (
        eval_reports._run_summary_marks_canonical_eval_completed
        is eval_policy_manifest_selection._run_summary_marks_canonical_eval_completed
    )
    assert eval_reports._default_dev_eval_summaries_path is (
        eval_policy_final_set_resolution._default_dev_eval_summaries_path
    )
    assert eval_reports._explicit_policy_selection is eval_policy_selection_results._explicit_policy_selection
    assert (
        eval_reports._manifest_policy_selection_fallback
        is eval_policy_selection_results._manifest_policy_selection_fallback
    )
    assert eval_reports.RunLevelReportUpdateInputs is eval_report_update_payloads.RunLevelReportUpdateInputs
    assert eval_reports.build_run_summary_update_fields is eval_report_update_payloads.build_run_summary_update_fields
    assert (
        eval_reports.build_determinism_report_update_fields
        is eval_report_update_payloads.build_determinism_report_update_fields
    )
    assert eval_reports._load_json_object is eval_report_io._load_json_object
    assert eval_reports._expected_sha256 is eval_report_io._expected_sha256
    assert eval_reports._load_run_summary_or_default is eval_report_scaffolding._load_run_summary_or_default
    assert eval_reports._ensure_run_level_report_scaffolding is (
        eval_report_scaffolding._ensure_run_level_report_scaffolding
    )
    assert eval_reports._resolve_policy_ids_for_run is eval_policy_selection._resolve_policy_ids_for_run
    assert eval_policy_selection._explicit_policy_selection is eval_policy_selection_results._explicit_policy_selection
    assert (
        eval_policy_selection._manifest_policy_selection_fallback
        is eval_policy_selection_results._manifest_policy_selection_fallback
    )
    assert eval_reports._persist_policy_selection_in_manifest is (
        eval_policy_selection._persist_policy_selection_in_manifest
    )
    assert eval_reports._update_run_level_reports is eval_report_updates._update_run_level_reports
