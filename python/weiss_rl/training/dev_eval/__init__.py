"""Periodic dev-eval support.

The canonical training entrypoint imports concrete behavior from the named
modules in this package. This package root remains as a small compatibility
surface for existing tests and retained scripts.
"""

from __future__ import annotations

from weiss_rl.training.dev_eval.common import (
    DevEvalTrainingPaths,
    json_relative_path,
    load_json_object,
    periodic_dev_eval_summaries_path,
    resolve_repo_path,
    stall_monitor_state_path,
    write_json,
)
from weiss_rl.training.dev_eval.model_clone import clone_cpu_eval_model
from weiss_rl.training.dev_eval.plan import (
    PERIODIC_DEV_EVAL_PLAN,
    PeriodicDevEvalPlanStep,
    periodic_dev_eval_plan_payload,
)
from weiss_rl.training.dev_eval.runtime_contracts import (
    evaluation_config_or_raise,
    legal_ids_for_env_row,
    should_run_periodic_dev_eval,
    validate_periodic_dev_eval_contract,
)
from weiss_rl.training.dev_eval.seed_schedule import (
    periodic_dev_eval_bootstrap_seed,
    periodic_dev_eval_rng_seed,
    periodic_dev_eval_schedule,
    periodic_dev_eval_seed_usage_payload,
    promotion_gate_bootstrap_seed,
    promotion_gate_rng_seed,
    resolve_periodic_dev_eval_seed_file,
)
from weiss_rl.training.dev_eval.summary_state import (
    persist_periodic_dev_eval_summary,
    summary_rate,
    update_stall_monitor,
)

__all__ = [
    "DevEvalTrainingPaths",
    "PERIODIC_DEV_EVAL_PLAN",
    "PeriodicDevEvalPlanStep",
    "clone_cpu_eval_model",
    "evaluation_config_or_raise",
    "json_relative_path",
    "legal_ids_for_env_row",
    "load_json_object",
    "periodic_dev_eval_bootstrap_seed",
    "periodic_dev_eval_plan_payload",
    "periodic_dev_eval_rng_seed",
    "periodic_dev_eval_schedule",
    "periodic_dev_eval_seed_usage_payload",
    "periodic_dev_eval_summaries_path",
    "persist_periodic_dev_eval_summary",
    "promotion_gate_bootstrap_seed",
    "promotion_gate_rng_seed",
    "resolve_periodic_dev_eval_seed_file",
    "resolve_repo_path",
    "should_run_periodic_dev_eval",
    "stall_monitor_state_path",
    "summary_rate",
    "update_stall_monitor",
    "validate_periodic_dev_eval_contract",
    "write_json",
]
