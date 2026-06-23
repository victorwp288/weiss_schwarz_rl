from __future__ import annotations


def test_targeted_confirm_entrypoint_exposes_only_cli_main() -> None:
    from weiss_rl.eval.final import worker_runtime as final_eval_worker_runtime
    from weiss_rl.eval.targeted_confirm import core as targeted_confirm_core
    from weiss_rl.eval.targeted_confirm import entrypoint as targeted_confirm_entrypoint

    assert targeted_confirm_entrypoint.main is targeted_confirm_core.main
    assert targeted_confirm_core.run_final_eval_worker is final_eval_worker_runtime.run_final_eval_worker
    retired_helper_exports = {
        "DEFAULT_OPPONENTS",
        "FAST_LOOP_EXACT_PAIRED_SEEDS",
        "MAIN_LEAGUE_FULL13_OPPONENTS",
        "MAIN_LEAGUE_SENTINEL_OPPONENTS",
        "OPPONENT_SETS",
        "TargetedConfirmPlan",
        "_god_search_payload_from_args",
        "_require_exact_opponent_panel",
        "_require_fast_loop_gate",
        "_resolve_opponents",
        "_resolve_paired_seed_file",
        "_run_targeted_jobs",
        "_targeted_eval_job",
        "_targeted_worker",
        "_validate_fast_loop_eval_request",
        "build_arg_parser",
        "build_targeted_confirm_jobs",
        "build_targeted_confirm_summary",
        "parse_args",
        "prepare_targeted_confirm_plan",
        "validate_targeted_confirm_request",
        "write_targeted_confirm_summary",
    }

    assert not any(hasattr(targeted_confirm_entrypoint, name) for name in retired_helper_exports)


def test_targeted_confirm_core_does_not_reexport_plan_helpers() -> None:
    from weiss_rl.eval.targeted_confirm import core as targeted_confirm_core

    retired_plan_exports = {
        "DEFAULT_OPPONENTS",
        "FAST_LOOP_EXACT_PAIRED_SEEDS",
        "MAIN_LEAGUE_FULL13_OPPONENTS",
        "MAIN_LEAGUE_SENTINEL_OPPONENTS",
        "OPPONENT_SETS",
        "TargetedConfirmPlan",
        "_god_search_payload_from_args",
        "_require_exact_opponent_panel",
        "_require_fast_loop_gate",
        "_resolve_opponents",
        "_resolve_paired_seed_file",
        "_targeted_eval_job",
        "_validate_fast_loop_eval_request",
        "build_arg_parser",
        "build_targeted_confirm_jobs",
        "build_targeted_confirm_summary",
        "parse_args",
        "prepare_targeted_confirm_plan",
        "validate_targeted_confirm_request",
        "write_targeted_confirm_summary",
    }

    assert not any(hasattr(targeted_confirm_core, name) for name in retired_plan_exports)


def test_targeted_confirm_plan_exposes_only_parser_helpers() -> None:
    from weiss_rl.eval.targeted_confirm import plan as targeted_confirm_plan

    retired_helper_exports = {
        "DEFAULT_OPPONENTS",
        "FAST_LOOP_EXACT_PAIRED_SEEDS",
        "MAIN_LEAGUE_FULL13_OPPONENTS",
        "MAIN_LEAGUE_SENTINEL_OPPONENTS",
        "OPPONENT_SETS",
        "TargetedConfirmPlan",
        "_god_search_payload_from_args",
        "_require_exact_opponent_panel",
        "_require_fast_loop_gate",
        "_resolve_opponents",
        "_resolve_paired_seed_file",
        "_targeted_eval_job",
        "_validate_fast_loop_eval_request",
        "build_targeted_confirm_jobs",
        "build_targeted_confirm_summary",
        "prepare_targeted_confirm_plan",
        "validate_targeted_confirm_request",
        "write_targeted_confirm_summary",
    }

    assert targeted_confirm_plan.__all__ == ["build_arg_parser", "parse_args"]
    assert not any(hasattr(targeted_confirm_plan, name) for name in retired_helper_exports)


def test_eval_root_does_not_export_targeted_confirm_module_aliases() -> None:
    import weiss_rl.eval as eval_package

    assert not hasattr(eval_package, "targeted_confirm_core")
    assert not hasattr(eval_package, "targeted_confirm_entrypoint")
    assert not hasattr(eval_package, "targeted_confirm_jobs")
    assert not hasattr(eval_package, "targeted_confirm_opponents")
    assert not hasattr(eval_package, "targeted_confirm_plan")
    assert not hasattr(eval_package, "targeted_confirm_summary")
