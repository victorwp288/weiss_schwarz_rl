from __future__ import annotations


def test_canonical_eval_reexports_split_module_owners() -> None:
    import weiss_rl.workflows.canonical_eval.cli_messages as eval_canonical_cli_messages
    import weiss_rl.workflows.canonical_eval.dependencies as eval_canonical_dependencies
    import weiss_rl.workflows.canonical_eval.entrypoint_adapter as eval_canonical_entrypoint_adapter
    import weiss_rl.workflows.canonical_eval.entrypoint_request as eval_canonical_entrypoint_request
    import weiss_rl.workflows.canonical_eval.figure_outputs as eval_canonical_figure_outputs
    import weiss_rl.workflows.canonical_eval.final_eval as eval_canonical_final_eval
    import weiss_rl.workflows.canonical_eval.metagame_outputs as eval_canonical_metagame_outputs
    import weiss_rl.workflows.canonical_eval.output_bundle as eval_canonical_output_bundle
    import weiss_rl.workflows.canonical_eval.outputs as eval_canonical_outputs
    import weiss_rl.workflows.canonical_eval.phases as eval_canonical_phases
    import weiss_rl.workflows.canonical_eval.policy_runtime as eval_canonical_policy_runtime
    import weiss_rl.workflows.canonical_eval.publisher as eval_canonical_publisher
    import weiss_rl.workflows.canonical_eval.readiness_outputs as eval_canonical_readiness_outputs
    import weiss_rl.workflows.canonical_eval.report_publication as eval_canonical_report_publication
    import weiss_rl.workflows.canonical_eval.runtime as eval_canonical_runtime
    import weiss_rl.workflows.canonical_eval.seed_budget as eval_canonical_seed_budget
    import weiss_rl.workflows.canonical_eval.setup as eval_canonical_setup
    import weiss_rl.workflows.canonical_eval.state as eval_canonical_state
    import weiss_rl.workflows.canonical_eval.supplemental_outputs as eval_canonical_supplemental_outputs
    import weiss_rl.workflows.canonical_eval.tensorboard_publication as eval_canonical_tensorboard_publication
    import weiss_rl.workflows.eval_canonical as eval_canonical
    from weiss_rl.workflows import eval_entrypoint as eval_script

    assert eval_script.run_canonical_eval_pipeline is eval_canonical.run_canonical_eval_pipeline
    assert (
        eval_script.run_canonical_eval_entrypoint_pipeline
        is eval_canonical_entrypoint_adapter.run_canonical_eval_entrypoint_pipeline
    )
    assert "CanonicalEvalEntrypointRequest" in eval_canonical_entrypoint_request.__all__
    assert "canonical_eval_entrypoint_request" in eval_canonical_entrypoint_request.__all__
    assert eval_script.CanonicalEvalDependencies is eval_canonical.CanonicalEvalDependencies
    assert eval_script.CanonicalEvalDependencies is eval_canonical_dependencies.CanonicalEvalDependencies
    assert eval_canonical.CanonicalEvalDependencies is eval_canonical_dependencies.CanonicalEvalDependencies
    assert eval_canonical.CanonicalEvalRunState is eval_canonical_state.CanonicalEvalRunState
    assert eval_canonical.CanonicalEvalRuntimeState is eval_canonical_state.CanonicalEvalRuntimeState
    assert eval_canonical_phases.CanonicalEvalRunState is eval_canonical_state.CanonicalEvalRunState
    assert eval_canonical_phases.CanonicalEvalRuntimeState is eval_canonical_state.CanonicalEvalRuntimeState
    assert eval_canonical.prepare_canonical_eval_run_state is eval_canonical_setup.prepare_canonical_eval_run_state
    assert eval_canonical_phases.prepare_canonical_eval_run_state is (
        eval_canonical_setup.prepare_canonical_eval_run_state
    )
    assert eval_canonical.resolve_canonical_eval_runtime_state is (
        eval_canonical_runtime.resolve_canonical_eval_runtime_state
    )
    assert eval_canonical_phases.resolve_canonical_eval_runtime_state is (
        eval_canonical_runtime.resolve_canonical_eval_runtime_state
    )
    assert eval_canonical.write_canonical_eval_outputs is eval_canonical_outputs.write_canonical_eval_outputs
    assert eval_canonical_phases.write_canonical_eval_outputs is (eval_canonical_outputs.write_canonical_eval_outputs)
    assert "render_canonical_eval_output_messages" in eval_canonical_cli_messages.__all__
    assert "build_canonical_figure_outputs" in eval_canonical_figure_outputs.__all__
    assert "run_canonical_final_eval_output" in eval_canonical_final_eval.__all__
    assert "build_canonical_metagame_output" in eval_canonical_metagame_outputs.__all__
    assert "build_canonical_readiness_output" in eval_canonical_readiness_outputs.__all__
    assert "build_canonical_supplemental_outputs" in eval_canonical_supplemental_outputs.__all__
    assert "build_canonical_eval_output_bundle" in eval_canonical_output_bundle.__all__
    assert "publish_canonical_eval_outputs" in eval_canonical_publisher.__all__
    assert "publish_canonical_eval_run_reports" in eval_canonical_report_publication.__all__
    assert "publish_canonical_eval_tensorboard_summaries" in eval_canonical_tensorboard_publication.__all__
    assert "resolve_canonical_eval_policy_runtime" in eval_canonical_policy_runtime.__all__
    assert "resolve_recommended_focal_policy_id" in eval_canonical_policy_runtime.__all__
    assert "resolve_canonical_eval_seed_budget" in eval_canonical_seed_budget.__all__
