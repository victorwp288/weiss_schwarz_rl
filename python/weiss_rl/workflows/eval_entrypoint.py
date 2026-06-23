from __future__ import annotations

# ruff: noqa: F401
import argparse
from pathlib import Path
from typing import Any

from weiss_rl.workflows.eval_entrypoint_support.exports import (
    EVAL_ENTRYPOINT_EXPORTS,
    ArtifactLayout,
    CanonicalEvalDependencies,
    EvalDispatchDependencies,
    EvalStartup,
    EvalStartupDependencies,
    EvalValidatedArgs,
    SimulatorEvalRunner,
    TensorBoardLogger,
    _effective_manifest_git_commit,
    _ensure_run_level_report_scaffolding,
    _expected_sha256,
    _load_determinism_report_or_default,
    _load_environment_or_default,
    _load_json_object,
    _load_run_summary_or_default,
    _normalize_git_commit,
    _normalize_sha256,
    _persist_policy_selection_in_manifest,
    _require_matching_hash,
    _require_positive_int,
    _resolve_policy_ids_for_run,
    _resolve_run_label,
    _update_run_level_reports,
    _write_json,
    assert_spec_bundle_contract,
    build_canonical_eval_dependencies,
    build_eval_dispatch_dependencies,
    build_eval_parser,
    build_eval_startup_dependencies,
    build_matchup_export,
    build_paper_readiness_summary,
    build_seat_advantage_diagnostics,
    build_sensitivity_report,
    compute_config_hash256,
    load_dev_eval_summaries,
    load_eval_game_records,
    load_stack_config,
    load_study_config,
    load_verified_simulator_contract,
    parse_seed_file,
    prepare_eval_startup,
    print_startup_banner,
    public_demo_spec_bundle,
    public_demo_spec_hash256,
    public_demo_stop_rules,
    recommend_focal_policy_id,
    render_paper_figures,
    resolve_eval_policies,
    resolve_final_policy_set,
    run_canonical_eval_entrypoint_pipeline,
    run_canonical_eval_pipeline,
    run_eval_dispatch,
    run_final_eval,
    run_public_demo_eval_mode,
    run_public_demo_final_eval,
    run_summary_only_eval_mode,
    tensorboard_unavailable_reason,
    validate_eval_args,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
    write_paper_readiness_json,
)
from weiss_rl.workflows.eval_entrypoint_support.main import run_eval_entrypoint_main
from weiss_rl.workflows.eval_entrypoint_support.runtime import (
    build_eval_entrypoint_canonical_dependencies,
    build_eval_entrypoint_dispatch_dependencies,
    build_eval_entrypoint_startup_dependencies,
    run_eval_entrypoint,
    run_eval_entrypoint_canonical_pipeline,
)

__all__ = [
    *EVAL_ENTRYPOINT_EXPORTS,
    "_run_canonical_eval_pipeline",
    "main",
]


def _canonical_eval_dependencies() -> CanonicalEvalDependencies:
    return build_eval_entrypoint_canonical_dependencies(globals())


def _eval_dispatch_dependencies() -> EvalDispatchDependencies:
    return build_eval_entrypoint_dispatch_dependencies(globals())


def _eval_startup_dependencies() -> EvalStartupDependencies:
    return build_eval_entrypoint_startup_dependencies(globals())


def _run_canonical_eval_pipeline(
    *,
    parser: argparse.ArgumentParser,
    stack: Any,
    run_dir: Path,
    final_eval_dir: Path | None,
    policy_ids: list[str],
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    b1_baseline_run_dir: Path | None,
    bootstrap_samples: int,
    paired_seed_limit: int | None,
    stage1_paired_seeds: int | None,
    max_paired_seeds: int | None,
    skip_metagame: bool,
    study_config_path: Path | None,
    skip_figures: bool,
    skip_readiness: bool,
    git_commit_override: str,
) -> int:
    return run_eval_entrypoint_canonical_pipeline(
        entrypoint_globals=globals(),
        parser=parser,
        stack=stack,
        run_dir=run_dir,
        final_eval_dir=final_eval_dir,
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        b1_baseline_run_dir=b1_baseline_run_dir,
        bootstrap_samples=bootstrap_samples,
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        skip_metagame=skip_metagame,
        study_config_path=study_config_path,
        skip_figures=skip_figures,
        skip_readiness=skip_readiness,
        git_commit_override=git_commit_override,
    )


def main() -> None:
    run_eval_entrypoint(entrypoint_globals=globals())


if __name__ == "__main__":
    main()
