from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.artifacts.reproducibility import parse_seed_file
from weiss_rl.config import load_study_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger, tensorboard_unavailable_reason
from weiss_rl.eval import (
    build_paper_readiness_summary,
    load_dev_eval_summaries,
    run_final_eval,
    write_paper_readiness_json,
)
from weiss_rl.eval.policies.set import recommend_focal_policy_id
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
from weiss_rl.metagame import build_sensitivity_report
from weiss_rl.plotting.paper_figures import render_paper_figures
from weiss_rl.workflows.eval_support.eval_reports import (
    _effective_manifest_git_commit,
    _ensure_run_level_report_scaffolding,
    _load_json_object,
    _normalize_git_commit,
    _persist_policy_selection_in_manifest,
    _resolve_policy_ids_for_run,
    _update_run_level_reports,
)


@dataclass(frozen=True)
class CanonicalEvalDependencies:
    artifact_layout_cls: Any = ArtifactLayout
    tensorboard_logger_cls: Any = TensorBoardLogger
    tensorboard_unavailable_reason_fn: Any = tensorboard_unavailable_reason
    parse_seed_file_fn: Any = parse_seed_file
    load_study_config_fn: Any = load_study_config
    load_json_object_fn: Any = _load_json_object
    effective_manifest_git_commit_fn: Any = _effective_manifest_git_commit
    normalize_git_commit_fn: Any = _normalize_git_commit
    resolve_policy_ids_for_run_fn: Any = _resolve_policy_ids_for_run
    persist_policy_selection_in_manifest_fn: Any = _persist_policy_selection_in_manifest
    load_verified_simulator_contract_fn: Any = load_verified_simulator_contract
    resolve_eval_policies_fn: Any = resolve_eval_policies
    simulator_eval_runner_cls: Any = SimulatorEvalRunner
    recommend_focal_policy_id_fn: Any = recommend_focal_policy_id
    load_dev_eval_summaries_fn: Any = load_dev_eval_summaries
    run_final_eval_fn: Any = run_final_eval
    build_sensitivity_report_fn: Any = build_sensitivity_report
    render_paper_figures_fn: Any = render_paper_figures
    ensure_run_level_report_scaffolding_fn: Any = _ensure_run_level_report_scaffolding
    build_paper_readiness_summary_fn: Any = build_paper_readiness_summary
    write_paper_readiness_json_fn: Any = write_paper_readiness_json
    update_run_level_reports_fn: Any = _update_run_level_reports


__all__ = ["CanonicalEvalDependencies"]
