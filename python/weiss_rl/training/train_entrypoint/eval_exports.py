"""Evaluation, environment, and promotion exports for the entrypoint facade."""

from __future__ import annotations

# ruff: noqa: F401
from collections.abc import Mapping
from typing import Any

from weiss_rl.envs.decision_env import DecisionBoundaryBatch as _DecisionBoundaryBatch
from weiss_rl.eval.harness import ScheduledGame as _ScheduledGame
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.training.dev_eval import (
    clone_cpu_eval_model,
    evaluation_config_or_raise,
    json_relative_path,
    legal_ids_for_env_row,
    periodic_dev_eval_bootstrap_seed,
    periodic_dev_eval_rng_seed,
    periodic_dev_eval_schedule,
    periodic_dev_eval_summaries_path,
    persist_periodic_dev_eval_summary,
    promotion_gate_bootstrap_seed,
    promotion_gate_rng_seed,
    resolve_periodic_dev_eval_seed_file,
    resolve_repo_path,
    should_run_periodic_dev_eval,
    stall_monitor_state_path,
    validate_periodic_dev_eval_contract,
    write_json,
)
from weiss_rl.training.dev_eval import (
    update_stall_monitor as _update_stall_monitor_impl,
)
from weiss_rl.training.dev_eval.opponents import periodic_dev_eval_opponents as periodic_dev_eval_opponents
from weiss_rl.training.dev_eval.runner import PeriodicDevEvalRunner
from weiss_rl.training.environments import (
    build_ids_eval_env,
    build_training_env,
    env_pool_config,
    spec_dimensions,
)
from weiss_rl.training.periodic_dev_eval_run import run_periodic_dev_eval as run_periodic_dev_eval
from weiss_rl.training.policy_selection import (
    load_dev_eval_summaries,
    load_snapshot_registry,
    policy_set_selection,
    resolve_policy_set_selection,
    selection_requires_dev_eval_summaries,
    selection_requires_snapshot_registry,
)
from weiss_rl.training.promotion import (
    build_heuristic_public_policy,
    find_noleague_baseline_snapshot,
    promotion_anchor_policy_id_candidates,
    resolve_promotion_anchor_policy_ids,
    resolve_symbolic_promotion_anchor_policy_id,
    slug_policy_id,
    snapshot_meta_by_policy_id,
)
from weiss_rl.training.promotion_gate_execution import run_snapshot_promotion_gate as run_snapshot_promotion_gate
from weiss_rl.training.promotion_gate_runner import PromotionGateRunner

_EVAL_EXPORT_NAMES = (
    "_DecisionBoundaryBatch",
    "_ScheduledGame",
    "HeuristicPublicPolicy",
    "clone_cpu_eval_model",
    "evaluation_config_or_raise",
    "json_relative_path",
    "legal_ids_for_env_row",
    "periodic_dev_eval_bootstrap_seed",
    "periodic_dev_eval_rng_seed",
    "periodic_dev_eval_schedule",
    "periodic_dev_eval_summaries_path",
    "persist_periodic_dev_eval_summary",
    "promotion_gate_bootstrap_seed",
    "promotion_gate_rng_seed",
    "resolve_periodic_dev_eval_seed_file",
    "resolve_repo_path",
    "should_run_periodic_dev_eval",
    "stall_monitor_state_path",
    "validate_periodic_dev_eval_contract",
    "write_json",
    "_update_stall_monitor_impl",
    "periodic_dev_eval_opponents",
    "PeriodicDevEvalRunner",
    "build_ids_eval_env",
    "build_training_env",
    "env_pool_config",
    "spec_dimensions",
    "run_periodic_dev_eval",
    "load_dev_eval_summaries",
    "load_snapshot_registry",
    "policy_set_selection",
    "resolve_policy_set_selection",
    "selection_requires_dev_eval_summaries",
    "selection_requires_snapshot_registry",
    "build_heuristic_public_policy",
    "find_noleague_baseline_snapshot",
    "promotion_anchor_policy_id_candidates",
    "resolve_promotion_anchor_policy_ids",
    "resolve_symbolic_promotion_anchor_policy_id",
    "slug_policy_id",
    "snapshot_meta_by_policy_id",
    "run_snapshot_promotion_gate",
    "PromotionGateRunner",
)

EVAL_COMPAT_EXPORTS: Mapping[str, Any] = {
    **{name: globals()[name] for name in _EVAL_EXPORT_NAMES},
    "_DecisionBoundaryBatch": _DecisionBoundaryBatch,
    "_ScheduledGame": _ScheduledGame,
    "PeriodicDevEvalRunner": PeriodicDevEvalRunner,
    "resolve_policy_set_selection": resolve_policy_set_selection,
    "PromotionGateRunner": PromotionGateRunner,
}

__all__ = ["EVAL_COMPAT_EXPORTS", *_EVAL_EXPORT_NAMES]
