from __future__ import annotations


def test_eval_entrypoint_reexports_workflow_and_report_owners() -> None:
    import weiss_rl.workflows.eval_entrypoint as eval_entrypoint
    import weiss_rl.workflows.eval_entrypoint_support.export_groups as eval_entrypoint_export_groups
    import weiss_rl.workflows.eval_entrypoint_support.exports as eval_entrypoint_exports
    import weiss_rl.workflows.eval_entrypoint_support.external_exports as eval_entrypoint_external_exports
    import weiss_rl.workflows.eval_entrypoint_support.main as eval_entrypoint_main
    import weiss_rl.workflows.eval_entrypoint_support.report_exports as eval_entrypoint_report_exports
    import weiss_rl.workflows.eval_entrypoint_support.runtime as eval_entrypoint_runtime
    import weiss_rl.workflows.eval_entrypoint_support.workflow_exports as eval_entrypoint_workflow_exports
    import weiss_rl.workflows.eval_support.eval_parser as eval_parser
    from weiss_rl.workflows import eval_entrypoint as eval_script

    assert eval_script is eval_entrypoint
    assert eval_entrypoint.EVAL_ENTRYPOINT_EXPORTS is eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS
    assert eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS is eval_entrypoint_export_groups.EVAL_ENTRYPOINT_EXPORTS
    assert eval_entrypoint_exports.EVAL_REPORT_HELPER_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_REPORT_HELPER_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_WORKFLOW_COMPAT_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_WORKFLOW_COMPAT_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_ADDITIONAL_COMPAT_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_ADDITIONAL_COMPAT_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS == [
        *eval_entrypoint_export_groups.EVAL_REPORT_HELPER_EXPORTS,
        *eval_entrypoint_export_groups.EVAL_WORKFLOW_COMPAT_EXPORTS,
    ]
    assert set(eval_entrypoint_exports.__all__) == set(
        [
            *eval_entrypoint_export_groups.EVAL_ENTRYPOINT_EXPORTS,
            *eval_entrypoint_export_groups.EVAL_ADDITIONAL_COMPAT_EXPORTS,
            "EVAL_ADDITIONAL_COMPAT_EXPORTS",
            "EVAL_REPORT_HELPER_EXPORTS",
            "EVAL_WORKFLOW_COMPAT_EXPORTS",
        ]
    )
    assert eval_entrypoint.run_eval_entrypoint_main is eval_entrypoint_main.run_eval_entrypoint_main
    assert eval_entrypoint.run_eval_entrypoint is eval_entrypoint_runtime.run_eval_entrypoint
    assert (
        eval_entrypoint.run_eval_entrypoint_canonical_pipeline
        is eval_entrypoint_runtime.run_eval_entrypoint_canonical_pipeline
    )
    assert eval_script.build_eval_parser is eval_parser.build_eval_parser
    assert eval_entrypoint_exports.build_eval_parser is eval_entrypoint_workflow_exports.build_eval_parser
    assert eval_entrypoint_exports.resolve_eval_policies is eval_entrypoint_external_exports.resolve_eval_policies
    assert eval_entrypoint_exports._resolve_policy_ids_for_run is (
        eval_entrypoint_report_exports._resolve_policy_ids_for_run
    )
    assert eval_entrypoint_exports.TensorBoardLogger is eval_entrypoint_external_exports.TensorBoardLogger
    assert eval_entrypoint_exports.CanonicalEvalDependencies is (
        eval_entrypoint_workflow_exports.CanonicalEvalDependencies
    )
