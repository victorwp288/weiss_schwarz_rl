from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from weiss_rl.workflows.canonical_eval.dependencies import CanonicalEvalDependencies


def build_canonical_eval_dependencies(entrypoint_globals: Mapping[str, Any]) -> CanonicalEvalDependencies:
    return CanonicalEvalDependencies(
        artifact_layout_cls=entrypoint_globals["ArtifactLayout"],
        tensorboard_logger_cls=entrypoint_globals["TensorBoardLogger"],
        tensorboard_unavailable_reason_fn=entrypoint_globals["tensorboard_unavailable_reason"],
        parse_seed_file_fn=entrypoint_globals["parse_seed_file"],
        load_study_config_fn=entrypoint_globals["load_study_config"],
        load_json_object_fn=entrypoint_globals["_load_json_object"],
        effective_manifest_git_commit_fn=entrypoint_globals["_effective_manifest_git_commit"],
        normalize_git_commit_fn=entrypoint_globals["_normalize_git_commit"],
        resolve_policy_ids_for_run_fn=entrypoint_globals["_resolve_policy_ids_for_run"],
        persist_policy_selection_in_manifest_fn=entrypoint_globals["_persist_policy_selection_in_manifest"],
        load_verified_simulator_contract_fn=entrypoint_globals["load_verified_simulator_contract"],
        resolve_eval_policies_fn=entrypoint_globals["resolve_eval_policies"],
        simulator_eval_runner_cls=entrypoint_globals["SimulatorEvalRunner"],
        recommend_focal_policy_id_fn=entrypoint_globals["recommend_focal_policy_id"],
        load_dev_eval_summaries_fn=entrypoint_globals["load_dev_eval_summaries"],
        run_final_eval_fn=entrypoint_globals["run_final_eval"],
        build_sensitivity_report_fn=entrypoint_globals["build_sensitivity_report"],
        render_paper_figures_fn=entrypoint_globals["render_paper_figures"],
        ensure_run_level_report_scaffolding_fn=entrypoint_globals["_ensure_run_level_report_scaffolding"],
        build_paper_readiness_summary_fn=entrypoint_globals["build_paper_readiness_summary"],
        write_paper_readiness_json_fn=entrypoint_globals["write_paper_readiness_json"],
        update_run_level_reports_fn=entrypoint_globals["_update_run_level_reports"],
    )
