from __future__ import annotations

import inspect
import json
import multiprocessing as mp
import re
import shutil
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from weiss_rl.action_catalog import ActionCatalog
from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.autoscale import (
    ResolvedTrainingTopology,
    ScalingRequest,
    hardware_profile_from_name,
    resolve_training_topology,
    validate_ddp_world_size,
)
from weiss_rl.cli_banner import print_startup_banner
from weiss_rl.config import (
    StackConfig,
    apply_stack_overrides,
    canonical_config_dict,
    compute_config_hash256,
    load_stack_config,
    parse_override_tokens,
)
from weiss_rl.distributed import (
    DistributedContext,
    all_reduce_float,
    average_gradients,
    broadcast_object,
    destroy_process_group_if_initialized,
    distributed_context_from_env,
    init_process_group_if_needed,
    rank_seed,
    resolve_distributed_learner_device,
    shard_env_count,
)
from weiss_rl.distributed import (
    barrier as distributed_barrier,
)
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.eval import (
    DevEvalPolicySummary,
    EvalGameRecord,
    PayoffFoldScheme,
    build_matchup_export,
    build_seat_advantage_diagnostics,
    build_seat_swapped_schedule,
    record_completed_game,
    run_seat_swapped_matchup,
    write_episodes_jsonl,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
)
from weiss_rl.eval.harness import ScheduledGame
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policy_set import (
    HEURISTIC_PUBLIC_POLICY_ID,
    heuristic_public_profile_name_for_policy_id,
    select_final_policy_set_deterministic_v1,
)
from weiss_rl.eval.simulator_runner import _resolve_eval_device
from weiss_rl.league import (
    PromotionGateResult,
    resolve_promotion_gate_seed_file,
    run_promotion_gate,
)
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SNAPSHOT_WEIGHTS_FILENAME,
    SnapshotMeta,
    SnapshotRegistry,
)
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.ppo_lite_learner import PpoLiteLearner
from weiss_rl.manifest import (
    RunArtifacts,
    RunManifest,
    build_seed_file_manifest,
    default_run_dir_name,
    write_run_artifacts,
)
from weiss_rl.masking import assert_strictly_increasing_legal_ids
from weiss_rl.model import PolicyValueModel, build_policy_value_model
from weiss_rl.repro import (
    compute_run_id64,
    compute_run_id256,
    parse_seed_file,
)
from weiss_rl.residual_policy import (
    FrozenStoredLogitResidual,
    TrainableLiveFrozenB1Residual,
    load_frozen_stored_logit_residual,
)
from weiss_rl.runtime import QueueRuntime, QueueRuntimeMode, build_runtime_config, resolve_actor_device_layout
from weiss_rl.simulator_contract import SimulatorContract, load_verified_simulator_contract
from weiss_rl.spec import assert_spec_bundle_contract
from weiss_rl.tensorboard_logger import TensorBoardLogger, tensorboard_unavailable_reason
from weiss_rl.toy_public_demo import (
    PUBLIC_DEMO_MODE,
    public_demo_simulator_info,
    public_demo_spec_bundle,
    public_demo_spec_hash256,
    stage_public_demo_run,
)
from weiss_rl.training.anchor_resolution import (
    promotion_anchor_policy_id_candidates as _promotion_anchor_policy_id_candidates_impl,
)
from weiss_rl.training.anchor_resolution import (
    resolve_periodic_dev_eval_opponent_specs as _resolve_periodic_dev_eval_opponent_specs_impl,
)
from weiss_rl.training.anchor_resolution import (
    resolve_promotion_anchor_policy_ids as _resolve_promotion_anchor_policy_ids_impl,
)
from weiss_rl.training.anchor_resolution import (
    resolve_promotion_gate_anchor_specs as _resolve_promotion_gate_anchor_specs_impl,
)
from weiss_rl.training.anchor_resolution import (
    resolve_symbolic_promotion_anchor_policy_id as _resolve_symbolic_promotion_anchor_policy_id_impl,
)
from weiss_rl.training.anchor_resolution import (
    slug_policy_id as _slug_policy_id_impl,
)
from weiss_rl.training.anchor_resolution import (
    snapshot_meta_by_policy_id as _snapshot_meta_by_policy_id_impl,
)
from weiss_rl.training.anchor_resolution import (
    true_local_recent_snapshot_ids as _true_local_recent_snapshot_ids_impl,
)
from weiss_rl.training.batch_building import IMPALA_ALGORITHMS as _IMPALA_ALGORITHMS
from weiss_rl.training.batch_building import PPO_ALGORITHMS as _PPO_ALGORITHMS
from weiss_rl.training.batch_building import MinimalRollout
from weiss_rl.training.batch_building import bootstrap_values as _bootstrap_values_impl
from weiss_rl.training.batch_building import build_learner_batch as _build_learner_batch_impl
from weiss_rl.training.batch_building import collect_training_batch as _collect_training_batch_impl
from weiss_rl.training.batch_building import collect_training_batch_prefetch as _collect_training_batch_prefetch_impl
from weiss_rl.training.batch_building import torch_num_threads_scope as _torch_num_threads_scope_impl
from weiss_rl.training.bootstrap import apply_training_flag_overrides as _apply_training_flag_overrides_impl
from weiss_rl.training.bootstrap import build_train_arg_parser as _build_train_arg_parser
from weiss_rl.training.bootstrap import expected_sha256 as _expected_sha256_impl
from weiss_rl.training.bootstrap import format_loaded_stack_config_message as _format_loaded_stack_config_message_impl
from weiss_rl.training.bootstrap import (
    format_manifest_scaffold_only_message as _format_manifest_scaffold_only_message_impl,
)
from weiss_rl.training.bootstrap import format_manifest_written_message as _format_manifest_written_message_impl
from weiss_rl.training.bootstrap import (
    format_public_demo_disclaimer_message as _format_public_demo_disclaimer_message_impl,
)
from weiss_rl.training.bootstrap import format_public_demo_staged_message as _format_public_demo_staged_message_impl
from weiss_rl.training.bootstrap import format_resume_run_dir_message as _format_resume_run_dir_message_impl
from weiss_rl.training.bootstrap import format_spec_bundle_status_message as _format_spec_bundle_status_message_impl
from weiss_rl.training.bootstrap import format_tensorboard_disabled_message as _format_tensorboard_disabled_message_impl
from weiss_rl.training.bootstrap import manifest_scaffold_only_reason as _manifest_scaffold_only_reason_impl
from weiss_rl.training.bootstrap import normalize_sha256 as _normalize_sha256_impl
from weiss_rl.training.bootstrap import require_matching_hash as _require_matching_hash_impl
from weiss_rl.training.bootstrap import require_positive_int as _require_positive_int_impl
from weiss_rl.training.bootstrap import require_positive_optional_float as _require_positive_optional_float_impl
from weiss_rl.training.bootstrap import resolve_device as _resolve_device_impl
from weiss_rl.training.bootstrap import resolve_run_label as _resolve_run_label_impl
from weiss_rl.training.bootstrap import resolve_runtime_profile as _resolve_runtime_profile_impl
from weiss_rl.training.bootstrap import resolve_seed as _resolve_seed_impl
from weiss_rl.training.bootstrap import (
    runtime_training_prerequisite_failure as _runtime_training_prerequisite_failure_impl,
)
from weiss_rl.training.checkpoint_guard import checkpoint_guard_rollback_plan as _checkpoint_guard_rollback_plan
from weiss_rl.training.checkpoint_guard import (
    format_checkpoint_guard_final_selection_message as _format_checkpoint_guard_final_selection_message_impl,
)
from weiss_rl.training.checkpoint_guard import (
    format_checkpoint_guard_rollback_message as _format_checkpoint_guard_rollback_message_impl,
)
from weiss_rl.training.checkpoints import (
    build_checkpoint_record as _build_checkpoint_record,
)
from weiss_rl.training.checkpoints import (
    load_checkpoint_tracker as _load_checkpoint_tracker,
)
from weiss_rl.training.checkpoints import (
    publish_best_checkpoint_from_dev_eval as _publish_best_checkpoint_from_dev_eval_impl,
)
from weiss_rl.training.checkpoints import (
    publish_checkpoint_aliases as _publish_checkpoint_aliases_impl,
)
from weiss_rl.training.checkpoints import (
    relative_path_text as _checkpoint_relative_path_text,
)
from weiss_rl.training.checkpoints import (
    seed_checkpoint_tracker_from_resume_best as _seed_checkpoint_tracker_from_resume_best,
)
from weiss_rl.training.checkpoints import (
    write_checkpoint_tracker as _write_checkpoint_tracker,
)
from weiss_rl.training.confirmatory_eval import (
    build_confirmatory_dev_eval_plan as _build_confirmatory_dev_eval_plan_impl,
)
from weiss_rl.training.confirmatory_eval import (
    format_confirmatory_dev_eval_message as _format_confirmatory_dev_eval_message_impl,
)
from weiss_rl.training.curriculum_guards import (
    apply_stall_monitor_to_dev_eval_summary as _apply_stall_monitor_to_dev_eval_summary_impl,
)
from weiss_rl.training.curriculum_guards import (
    early_cutoff_metric_updates as _early_cutoff_metric_updates_impl,
)
from weiss_rl.training.curriculum_guards import (
    format_early_cutoff_triggered_message as _format_early_cutoff_triggered_message_impl,
)
from weiss_rl.training.curriculum_guards import (
    format_stall_monitor_warning as _format_stall_monitor_warning_impl,
)
from weiss_rl.training.curriculum_guards import (
    format_training_stopped_by_early_cutoff_message as _format_training_stopped_by_early_cutoff_message_impl,
)
from weiss_rl.training.curriculum_guards import (
    update_early_cutoff as _update_early_cutoff_impl,
)
from weiss_rl.training.curriculum_guards import (
    update_stall_monitor as _update_stall_monitor_impl,
)
from weiss_rl.training.dev_eval_metrics import (
    checkpoint_candidate_metric as _checkpoint_candidate_metric,  # noqa: F401
)
from weiss_rl.training.dev_eval_metrics import (
    confirmatory_dev_eval_request as _confirmatory_dev_eval_request,  # noqa: F401
)
from weiss_rl.training.dev_eval_metrics import (
    dev_eval_aggregate_score as _dev_eval_aggregate_score,
)
from weiss_rl.training.dev_eval_metrics import (
    dev_eval_confidence_stats as _dev_eval_confidence_stats,
)
from weiss_rl.training.dev_eval_metrics import (
    dev_eval_ineligibility_reasons as _dev_eval_ineligibility_reasons,  # noqa: F401
)
from weiss_rl.training.dev_eval_metrics import (
    dev_eval_is_authoritative as _dev_eval_is_authoritative,
)
from weiss_rl.training.dev_eval_metrics import (
    dev_eval_surface as _dev_eval_surface,
)
from weiss_rl.training.dev_eval_metrics import (
    should_promote_best_checkpoint as _should_promote_best_checkpoint,  # noqa: F401
)
from weiss_rl.training.dev_eval_metrics import (
    weighted_dev_eval_aggregate as _weighted_dev_eval_aggregate,  # noqa: F401
)
from weiss_rl.training.dev_eval_runner import PendingPeriodicDevEval, PendingPromotionGate
from weiss_rl.training.dev_eval_runner import (
    PeriodicDevEvalRunner as _TrainingPeriodicDevEvalRunner,
)
from weiss_rl.training.eval_artifacts import (
    append_b2_disagreement_audit_request as _append_b2_disagreement_audit_request_impl,
)
from weiss_rl.training.eval_artifacts import (
    append_checkpoint_guard_event as _append_checkpoint_guard_event_impl,
)
from weiss_rl.training.eval_artifacts import (
    b2_disagreement_audit_requests_path as _b2_disagreement_audit_requests_path_impl,
)
from weiss_rl.training.eval_artifacts import (
    build_periodic_dev_eval_checkpoint_summary as _build_periodic_dev_eval_checkpoint_summary_impl,
)
from weiss_rl.training.eval_artifacts import (
    build_periodic_dev_eval_matchup_context_payload as _build_periodic_dev_eval_matchup_context_payload_impl,
)
from weiss_rl.training.eval_artifacts import (
    build_periodic_dev_eval_matchup_runtime_payload as _build_periodic_dev_eval_matchup_runtime_payload_impl,
)
from weiss_rl.training.eval_artifacts import (
    build_periodic_dev_eval_seed_usage_payload as _build_periodic_dev_eval_seed_usage_payload_impl,
)
from weiss_rl.training.eval_artifacts import (
    build_periodic_dev_eval_summary_record as _build_periodic_dev_eval_summary_record_impl,
)
from weiss_rl.training.eval_artifacts import (
    checkpoint_guard_log_path as _checkpoint_guard_log_path_impl,
)
from weiss_rl.training.eval_artifacts import (
    collate_periodic_dev_eval_seed_block_matchup as _collate_periodic_dev_eval_seed_block_matchup_impl,
)
from weiss_rl.training.eval_artifacts import (
    dev_eval_has_confidence_only_block as _dev_eval_has_confidence_only_block_impl,
)
from weiss_rl.training.eval_artifacts import (
    format_b2_disagreement_audit_request_message as _format_b2_disagreement_audit_request_message_impl,
)
from weiss_rl.training.eval_artifacts import (
    format_periodic_dev_eval_console_message as _format_periodic_dev_eval_console_message_impl,
)
from weiss_rl.training.eval_artifacts import (
    format_periodic_dev_eval_scheduled_message as _format_periodic_dev_eval_scheduled_message_impl,
)
from weiss_rl.training.eval_artifacts import (
    group_periodic_dev_eval_seed_block_results as _group_periodic_dev_eval_seed_block_results_impl,
)
from weiss_rl.training.eval_artifacts import (
    maybe_log_structured_mainmove_guard as _maybe_log_structured_mainmove_guard_impl,
)
from weiss_rl.training.eval_artifacts import (
    maybe_request_b2_disagreement_audit as _maybe_request_b2_disagreement_audit_impl,
)
from weiss_rl.training.eval_artifacts import (
    periodic_dev_eval_fast_screens_path as _periodic_dev_eval_fast_screens_path_impl,
)
from weiss_rl.training.eval_artifacts import (
    periodic_dev_eval_matchup_dir as _periodic_dev_eval_matchup_dir_impl,
)
from weiss_rl.training.eval_artifacts import (
    periodic_dev_eval_summaries_path as _periodic_dev_eval_summaries_path_impl,
)
from weiss_rl.training.eval_artifacts import (
    persist_periodic_dev_eval_fast_screen as _persist_periodic_dev_eval_fast_screen_impl,
)
from weiss_rl.training.eval_artifacts import (
    persist_periodic_dev_eval_result as _persist_periodic_dev_eval_result_impl,
)
from weiss_rl.training.eval_artifacts import (
    persist_periodic_dev_eval_summary as _persist_periodic_dev_eval_summary_impl,
)
from weiss_rl.training.eval_artifacts import (
    promotion_gate_records_by_anchor_index as _promotion_gate_records_by_anchor_index_impl,
)
from weiss_rl.training.eval_artifacts import (
    sum_periodic_dev_eval_counter_payloads as _sum_periodic_dev_eval_counter_payloads_impl,
)
from weiss_rl.training.eval_model_cache import get_cached_eval_model as _get_cached_eval_model_impl
from weiss_rl.training.eval_model_cache import remember_eval_model as _remember_eval_model_impl
from weiss_rl.training.eval_schedule import (
    AsyncPeriodicDevEvalRequest,
    AsyncPromotionGateRequest,
    PeriodicDevEvalOpponentSpec,
    PeriodicDevEvalSeedBlockJob,
    PromotionGateSeedBlockJob,
)
from weiss_rl.training.eval_schedule import (
    build_async_periodic_dev_eval_request as _build_async_periodic_dev_eval_request,
)
from weiss_rl.training.eval_schedule import (
    build_async_promotion_gate_request as _build_async_promotion_gate_request,
)
from weiss_rl.training.eval_schedule import (
    build_periodic_dev_eval_seed_block_jobs as _build_periodic_dev_eval_seed_block_jobs_impl,
)
from weiss_rl.training.eval_schedule import (
    build_promotion_gate_seed_block_jobs as _build_promotion_gate_seed_block_jobs_impl,
)
from weiss_rl.training.eval_schedule import (
    evaluation_config_or_raise as _evaluation_config_or_raise_impl,
)
from weiss_rl.training.eval_schedule import (
    format_league_eval_warmup_gate_message as _format_league_eval_warmup_gate_message_impl,
)
from weiss_rl.training.eval_schedule import (
    is_noleague_baseline_role as _is_noleague_baseline_role_impl,
)
from weiss_rl.training.eval_schedule import (
    json_relative_path as _json_relative_path_impl,
)
from weiss_rl.training.eval_schedule import (
    league_eval_warmup_gate_status as _league_eval_warmup_gate_status_impl,
)
from weiss_rl.training.eval_schedule import (
    periodic_dev_eval_anchor_weight_map as _periodic_dev_eval_anchor_weight_map_impl,
)
from weiss_rl.training.eval_schedule import (
    periodic_dev_eval_duplicate_policy_ids as _periodic_dev_eval_duplicate_policy_ids_impl,
)
from weiss_rl.training.eval_schedule import (
    periodic_dev_eval_schedule as _periodic_dev_eval_schedule_impl,
)
from weiss_rl.training.eval_schedule import (
    periodic_dev_eval_schedule_for_seed_items as _periodic_dev_eval_schedule_for_seed_items_impl,
)
from weiss_rl.training.eval_schedule import (
    resolve_periodic_dev_eval_seed_file as _resolve_periodic_dev_eval_seed_file_impl,
)
from weiss_rl.training.eval_schedule import (
    resolve_repo_path as _resolve_repo_path_impl,
)
from weiss_rl.training.eval_schedule import (
    resolved_periodic_dev_eval_worker_devices as _resolved_periodic_dev_eval_worker_devices_impl,
)
from weiss_rl.training.eval_schedule import (
    resolved_promotion_gate_worker_devices as _resolved_promotion_gate_worker_devices_impl,
)
from weiss_rl.training.eval_schedule import (
    shard_periodic_dev_eval_opponents as _shard_periodic_dev_eval_opponents_impl,
)
from weiss_rl.training.eval_schedule import (
    shard_periodic_dev_eval_seed_block_jobs as _shard_periodic_dev_eval_seed_block_jobs_impl,
)
from weiss_rl.training.eval_schedule import (
    shard_promotion_gate_anchor_specs as _shard_promotion_gate_anchor_specs_impl,
)
from weiss_rl.training.eval_schedule import (
    shard_promotion_gate_seed_block_jobs as _shard_promotion_gate_seed_block_jobs_impl,
)
from weiss_rl.training.eval_schedule import (
    should_defer_noleague_baseline_alias_refresh as _should_defer_noleague_baseline_alias_refresh_impl,
)
from weiss_rl.training.eval_schedule import (
    should_run_periodic_dev_eval as _should_run_periodic_dev_eval_impl,
)
from weiss_rl.training.eval_schedule import (
    split_periodic_dev_eval_seed_blocks as _split_periodic_dev_eval_seed_blocks_impl,
)
from weiss_rl.training.eval_schedule import (
    sync_runtime_league_eval_warmup_gate as _sync_runtime_league_eval_warmup_gate_impl,
)
from weiss_rl.training.eval_schedule import (
    validate_parallel_worker_device_pool as _validate_parallel_worker_device_pool_impl,
)
from weiss_rl.training.eval_schedule import (
    validate_periodic_dev_eval_contract as _validate_periodic_dev_eval_contract_impl,
)
from weiss_rl.training.eval_seeds import (
    expand_periodic_dev_eval_paired_seeds as _expand_periodic_dev_eval_paired_seeds,  # noqa: F401
)
from weiss_rl.training.eval_seeds import (
    periodic_dev_eval_bootstrap_seed as _periodic_dev_eval_bootstrap_seed,
)
from weiss_rl.training.eval_seeds import (
    periodic_dev_eval_rng_seed as _periodic_dev_eval_rng_seed,
)
from weiss_rl.training.eval_seeds import (
    promotion_gate_bootstrap_seed as _promotion_gate_bootstrap_seed,
)
from weiss_rl.training.eval_seeds import (
    promotion_gate_rng_seed as _promotion_gate_rng_seed,
)
from weiss_rl.training.guidance_schedules import (
    apply_guidance_schedule_for_next_update as _apply_guidance_schedule_for_next_update_impl,
)
from weiss_rl.training.guidance_schedules import (
    counterfactual_positive_coef_for_next_update as _counterfactual_positive_coef_for_next_update_impl,
)
from weiss_rl.training.guidance_schedules import (
    entropy_coef_for_next_update as _entropy_coef_for_next_update_impl,
)
from weiss_rl.training.guidance_schedules import (
    format_attached_reference_policy_message as _format_attached_reference_policy_message_impl,
)
from weiss_rl.training.guidance_schedules import (
    model_guidance_payload as _model_guidance_payload_impl,
)
from weiss_rl.training.guidance_schedules import (
    public_heuristic_actor_logit_bias_scale_for_next_update as _public_heuristic_actor_logit_bias_scale_for_next_update_impl,
)
from weiss_rl.training.guidance_schedules import (
    public_heuristic_logit_bias_scale_for_next_update as _public_heuristic_logit_bias_scale_for_next_update_impl,
)
from weiss_rl.training.guidance_schedules import (
    raw_b1_distill_coef_for_next_update as _raw_b1_distill_coef_for_next_update_impl,
)
from weiss_rl.training.guidance_schedules import (
    reference_policy_top_action_bc_coef_for_next_update as _reference_policy_top_action_bc_coef_for_next_update_impl,
)
from weiss_rl.training.guidance_schedules import (
    reference_policy_top_action_family_bc_coef_for_next_update as _reference_policy_top_action_family_bc_coef_for_next_update_impl,
)
from weiss_rl.training.guidance_schedules import (
    restore_model_guidance_from_payload as _restore_model_guidance_from_payload_impl,
)
from weiss_rl.training.guidance_schedules import (
    teacher_public_heuristic_coef_for_next_update as _teacher_public_heuristic_coef_for_next_update_impl,
)
from weiss_rl.training.learner_setup import (
    format_trainable_main_residual_policy_enabled_message as _format_trainable_main_residual_policy_enabled_message_impl,
)
from weiss_rl.training.learner_setup import maybe_compile_learner_model as _maybe_compile_learner_model_impl
from weiss_rl.training.manifest_payloads import evaluation_pinning as _evaluation_pinning_impl
from weiss_rl.training.manifest_payloads import hardware_summary as _hardware_summary_impl
from weiss_rl.training.manifest_payloads import manifest_actor_device_layout as _manifest_actor_device_layout_impl
from weiss_rl.training.manifest_payloads import training_controls_payload as _training_controls_payload_impl
from weiss_rl.training.paths import TrainingPaths
from weiss_rl.training.paths import build_training_paths as _build_training_paths
from weiss_rl.training.paths import run_artifacts_from_existing_run_dir as _run_artifacts_from_existing_run_dir_impl
from weiss_rl.training.profiling import build_training_profiler as _build_training_profiler_impl
from weiss_rl.training.profiling import (
    format_torch_profiler_trace_written_message as _format_torch_profiler_trace_written_message_impl,
)
from weiss_rl.training.profiling import profile_block as _profile_block_impl
from weiss_rl.training.promotion_artifacts import (
    assemble_parallel_promotion_gate_result as _assemble_parallel_promotion_gate_result_impl,
)
from weiss_rl.training.promotion_artifacts import (
    build_parallel_promotion_gate_plan as _build_parallel_promotion_gate_plan_impl,
)
from weiss_rl.training.promotion_artifacts import (
    build_promotion_gate_worker_payloads as _build_promotion_gate_worker_payloads_impl,
)
from weiss_rl.training.promotion_artifacts import (
    format_optional_heuristic_public_anchors_skipped_message as _format_optional_heuristic_public_anchors_skipped_message_impl,
)
from weiss_rl.training.promotion_artifacts import (
    format_promotion_gate_discarded_after_rollback_message as _format_promotion_gate_discarded_after_rollback_message_impl,
)
from weiss_rl.training.promotion_artifacts import (
    format_promotion_gate_missing_anchors_message as _format_promotion_gate_missing_anchors_message_impl,
)
from weiss_rl.training.promotion_artifacts import (
    format_promotion_gate_skipped_eval_warmup_gate_message as _format_promotion_gate_skipped_eval_warmup_gate_message_impl,
)
from weiss_rl.training.promotion_artifacts import (
    format_promotion_gate_skipped_league_warmup_message as _format_promotion_gate_skipped_league_warmup_message_impl,
)
from weiss_rl.training.promotion_artifacts import (
    format_scheduled_async_promotion_gate_message as _format_scheduled_async_promotion_gate_message_impl,
)
from weiss_rl.training.promotion_artifacts import (
    promotion_gate_policy_maps as _promotion_gate_policy_maps_impl,
)
from weiss_rl.training.promotion_runner import PromotionGateRunnerCore
from weiss_rl.training.provenance import git_commit as _git_commit_impl
from weiss_rl.training.provenance import git_dirty as _git_dirty_impl
from weiss_rl.training.provenance import git_output as _git_output_impl
from weiss_rl.training.provenance import load_json_object as _load_json_object_impl
from weiss_rl.training.provenance import manifest_source_path as _manifest_source_path_impl
from weiss_rl.training.provenance import start_nonce as _start_nonce_impl
from weiss_rl.training.session import (
    format_resume_config_hash_mismatch_warning as _format_resume_config_hash_mismatch_warning_impl,
)
from weiss_rl.training.session import (
    format_resumed_learner_state_message as _format_resumed_learner_state_message_impl,
)
from weiss_rl.training.session import (
    format_seeded_checkpoint_best_alias_message as _format_seeded_checkpoint_best_alias_message_impl,
)
from weiss_rl.training.session import (
    format_seeded_resume_dev_eval_summary_message as _format_seeded_resume_dev_eval_summary_message_impl,
)
from weiss_rl.training.session import (
    format_structured_profiling_enabled_message as _format_structured_profiling_enabled_message_impl,
)
from weiss_rl.training.session import format_training_completed_message as _format_training_completed_message_impl
from weiss_rl.training.session import (
    format_wall_clock_budget_reached_message as _format_wall_clock_budget_reached_message_impl,
)
from weiss_rl.training.session import wall_clock_budget_metric_updates as _wall_clock_budget_metric_updates_impl
from weiss_rl.training.session import wall_clock_budget_reached as _wall_clock_budget_reached_impl
from weiss_rl.training.session import wall_clock_budget_seconds as _wall_clock_budget_seconds_impl
from weiss_rl.training.snapshot_artifacts import (
    apply_promotion_gate_payload as _apply_promotion_gate_payload,
)
from weiss_rl.training.snapshot_artifacts import (
    apply_promotion_gate_result as _apply_promotion_gate_result_impl,
)
from weiss_rl.training.snapshot_artifacts import (
    format_promotion_gate_registry_update_message as _format_promotion_gate_registry_update_message_impl,
)
from weiss_rl.training.snapshot_artifacts import (
    persist_snapshot_registry_entry as _persist_snapshot_registry_entry_impl,
)
from weiss_rl.training.snapshot_artifacts import (
    pin_snapshot_ids as _pin_snapshot_ids,
)
from weiss_rl.training.snapshot_artifacts import sha256_file as _snapshot_sha256_file
from weiss_rl.training.snapshot_artifacts import (
    unpin_snapshot_ids as _unpin_snapshot_ids,
)
from weiss_rl.training.snapshot_artifacts import (
    write_snapshot_artifact as _write_snapshot_artifact,  # noqa: F401 - compatibility helper used by tests/ad hoc imports.
)
from weiss_rl.training.snapshot_imports import (
    assert_noleague_baseline_config as _assert_noleague_baseline_config_impl,
)
from weiss_rl.training.snapshot_imports import (
    canonical_config_sections as _canonical_config_sections_impl,
)
from weiss_rl.training.snapshot_imports import (
    config_marks_noleague_baseline as _config_marks_noleague_baseline_impl,
)
from weiss_rl.training.snapshot_imports import (
    ensure_noleague_baseline_anchor as _ensure_noleague_baseline_anchor_impl,
)
from weiss_rl.training.snapshot_imports import (
    find_noleague_baseline_snapshot as _find_noleague_baseline_snapshot_impl,
)
from weiss_rl.training.snapshot_imports import (
    import_noleague_baseline_anchor as _import_noleague_baseline_anchor_impl,
)
from weiss_rl.training.snapshot_imports import (
    import_resume_league_snapshot_pool as _import_resume_league_snapshot_pool_impl,
)
from weiss_rl.training.snapshot_imports import (
    import_seed_snapshot_pool as _import_seed_snapshot_pool_impl,
)
from weiss_rl.training.snapshot_imports import (
    infer_run_dir_from_checkpoint_path as _infer_run_dir_from_checkpoint_path_impl,
)
from weiss_rl.training.snapshot_imports import (
    legacy_noleague_baseline_mode as _legacy_noleague_baseline_mode_impl,
)
from weiss_rl.training.snapshot_imports import (
    read_optional_hash_file as _read_optional_hash_file_impl,
)
from weiss_rl.training.snapshot_imports import (
    role_from_config_canonical as _role_from_config_canonical_impl,
)
from weiss_rl.training.snapshot_imports import (
    seed_snapshot_policy_id as _seed_snapshot_policy_id_impl,
)
from weiss_rl.training.snapshot_imports import (
    source_snapshot_is_resume_league_snapshot as _source_snapshot_is_resume_league_snapshot_impl,
)
from weiss_rl.training.snapshot_imports import (
    validate_existing_resume_league_import as _validate_existing_resume_league_import_impl,
)
from weiss_rl.training.snapshot_imports import (
    validate_imported_snapshot_contract as _validate_imported_snapshot_contract_impl,
)
from weiss_rl.training.snapshot_imports import (
    validate_seed_snapshot_import_contract as _validate_seed_snapshot_import_contract_impl,
)
from weiss_rl.training.snapshot_imports import (
    validate_snapshot_tensor_contract as _validate_snapshot_tensor_contract_impl,
)

_PROMOTION_GATE_RANDOMLEGAL_NAME = "B0 RandomLegal"
_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID = "b0_randomlegal"
_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME = "B1 NoLeague baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT = "baseline_checkpoint.pt"
_FIXED_OPPONENT_EXCLUSIONS = frozenset({_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID})
_LATEST_CHECKPOINT_FILENAME = "latest.pt"
_BEST_CHECKPOINT_FILENAME = "best.pt"
_CHECKPOINT_TRACKER_FILENAME = "checkpoint_tracker.json"
_CHECKPOINT_TRACKER_FORMAT = "checkpoint_tracker_v2"
_CONFIRMATORY_DEV_EVAL_MAX_PROB_SHORTFALL = 0.1
_CONFIRMATORY_DEV_EVAL_MAX_CI_EXCESS = 0.05
_EVAL_SNAPSHOT_MODEL_CACHE_MAX_ENTRIES = 12
_EVAL_SNAPSHOT_MODEL_CACHE: OrderedDict[tuple[Any, ...], PolicyValueModel] = OrderedDict()


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    checkpoint_path: Path
    update_count: int
    policy_version: int
    total_samples_processed: int


class _PeriodicDevEvalRunner(_TrainingPeriodicDevEvalRunner):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            **kwargs,
            build_ids_eval_env_fn=_build_ids_eval_env,
            legal_ids_for_env_row_fn=_legal_ids_for_env_row,
            periodic_dev_eval_rng_seed_fn=_periodic_dev_eval_rng_seed,
        )


def _normalize_sha256(value: str) -> str:
    return _normalize_sha256_impl(value)


def _expected_sha256(value: str, *, flag_name: str) -> str:
    return _expected_sha256_impl(value, flag_name=flag_name)


def _sha256_file(path: Path) -> str:
    return _snapshot_sha256_file(path)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _persist_snapshot_registry_entry(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    checkpoint_path: Path,
    model_state_dict: dict[str, Any],
    config_hash256: str,
    device: torch.device,
    update: int,
    policy_version: int,
    model: PolicyValueModel | None = None,
) -> str:
    return _persist_snapshot_registry_entry_impl(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        model_state_dict=model_state_dict,
        config_hash256=config_hash256,
        device=device,
        update=update,
        policy_version=policy_version,
        guidance_payload=_model_guidance_payload(model),
    )


def _require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    _require_matching_hash_impl(flag_name=flag_name, expected=expected, actual=actual)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_output(args: list[str]) -> str:
    return _git_output_impl(args, repo_root=_repo_root())


def _git_commit() -> str:
    return _git_commit_impl(repo_root=_repo_root())


def _git_dirty() -> bool:
    return _git_dirty_impl(repo_root=_repo_root())


def _start_nonce() -> int:
    return _start_nonce_impl()


def _hardware_summary(
    learner_device: torch.device | str = "cpu",
    *,
    actor_device: torch.device | str = "cpu",
    actor_device_layout: Sequence[str] | None = None,
) -> dict[str, str | int]:
    return _hardware_summary_impl(
        learner_device,
        actor_device=actor_device,
        actor_device_layout=actor_device_layout,
    )


def _scaling_request_from_config(training_config: Any) -> ScalingRequest:
    scaling = getattr(training_config, "scaling", None)
    return ScalingRequest(
        learner_parallelism=str(getattr(scaling, "learner_parallelism", "auto")),
        learner_gpu_count=str(getattr(scaling, "learner_gpu_count", "auto")),
        actor_topology=str(getattr(scaling, "actor_topology", "auto")),
        target_envs_per_gpu=int(getattr(scaling, "target_envs_per_gpu", 512)),
        min_envs_per_actor=int(getattr(scaling, "min_envs_per_actor", 32)),
        max_envs_per_actor=int(getattr(scaling, "max_envs_per_actor", 64)),
        max_actor_process_count=int(getattr(scaling, "max_actor_process_count", 64)),
        reserve_cpu_cores=int(getattr(scaling, "reserve_cpu_cores", 4)),
        learner_cpu_cores_per_gpu=int(getattr(scaling, "learner_cpu_cores_per_gpu", 2)),
        queue_depth_multiplier=int(getattr(scaling, "queue_depth_multiplier", 2)),
        ram_queue_fraction=float(getattr(scaling, "ram_queue_fraction", 0.25)),
        vram_fraction=float(getattr(scaling, "vram_fraction", 0.85)),
    )


def _resolve_autoscale_topology(
    *,
    stack: StackConfig,
    hardware_profile_name: str,
    runtime_mode: QueueRuntimeMode,
) -> ResolvedTrainingTopology:
    if stack.config.system is None or stack.config.training is None:
        raise RuntimeError("autoscale requires system and training config blocks")
    hardware = hardware_profile_from_name(hardware_profile_name)
    return resolve_training_topology(
        hardware=hardware,
        request=_scaling_request_from_config(stack.config.training),
        configured_actor_count=int(stack.config.system.actor_process_count),
        configured_envs_per_actor=int(stack.config.system.envs_per_actor),
        configured_batch_unrolls_per_update=int(stack.config.training.batch_unrolls_per_update),
        configured_queue_capacity_unrolls=int(stack.config.system.actor_queue_capacity_unrolls),
        runtime_mode=str(runtime_mode),
    )


def _resolve_ddp_backend(stack: StackConfig, *, device_override: str, backend: str) -> str:
    selected = str(backend).strip().lower()
    if selected != "auto":
        return selected
    requested = str(device_override).strip()
    if not requested:
        system_config = stack.config.system
        requested = "cpu" if system_config is None else str(getattr(system_config, "learner_device", "cpu"))
    if str(requested).strip().lower() == "cpu":
        return "gloo"
    return "auto"


def _ddp_indexed_cuda_override_error(device_override: str, *, world_size: int) -> str | None:
    requested = str(device_override).strip().lower()
    if int(world_size) <= 1 or not requested:
        return None
    if re.fullmatch(r"cuda:\d+", requested):
        return (
            "DDP multi-rank launches must not use an indexed CUDA learner override. "
            f"Received --device {device_override!r}; use --device cuda, --device cuda:auto, or omit --device "
            "so each rank maps to LOCAL_RANK."
        )
    return None


def _manifest_actor_device_layout(
    *,
    stack: StackConfig,
    num_envs: int,
    unroll_length: int,
    profile: str,
    seed: int,
    pass_action_id: int,
    runtime_mode: QueueRuntimeMode,
    learner_device: torch.device,
    resolved_topology: ResolvedTrainingTopology | None = None,
    rank_local_actor_devices: bool = False,
) -> tuple[str, ...] | None:
    return _manifest_actor_device_layout_impl(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
        learner_device=learner_device,
        resolved_topology=resolved_topology,
        rank_local_actor_devices=rank_local_actor_devices,
    )


def _evaluation_pinning(stack: StackConfig) -> dict[str, str | bool]:
    return _evaluation_pinning_impl(stack)


def _manifest_source_path(path: Path, *, root: Path) -> str:
    return _manifest_source_path_impl(path, root=root)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    return _load_json_object_impl(path, label=label)


def _apply_training_flag_overrides(
    stack: StackConfig,
    *,
    enable_profile_timers: bool,
    enable_torch_profiler: bool,
) -> StackConfig:
    return _apply_training_flag_overrides_impl(
        stack,
        enable_profile_timers=enable_profile_timers,
        enable_torch_profiler=enable_torch_profiler,
    )


def _experiment_role(stack: StackConfig) -> str:
    experiment = stack.config.experiment
    return "" if experiment is None else str(experiment.role).strip()


def _is_noleague_baseline_role(role: str) -> bool:
    return _is_noleague_baseline_role_impl(role)


def _canonical_config_sections(config_canonical: Mapping[str, Any]) -> Mapping[str, Any]:
    return _canonical_config_sections_impl(config_canonical)


def _role_from_config_canonical(config_canonical: Mapping[str, Any]) -> str:
    return _role_from_config_canonical_impl(config_canonical)


def _legacy_noleague_baseline_mode(config_canonical: Mapping[str, Any]) -> str:
    return _legacy_noleague_baseline_mode_impl(config_canonical)


def _config_marks_noleague_baseline(config_canonical: Mapping[str, Any]) -> bool:
    return _config_marks_noleague_baseline_impl(config_canonical)


def _assert_noleague_baseline_config(config_canonical: Mapping[str, Any]) -> None:
    _assert_noleague_baseline_config_impl(config_canonical)


def _read_optional_hash_file(path: Path) -> str | None:
    return _read_optional_hash_file_impl(path)


def _validate_imported_snapshot_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    _validate_imported_snapshot_contract_impl(
        source_run_dir=source_run_dir,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )


def _load_snapshot_registry(path: Path) -> SnapshotRegistry:
    if not path.exists():
        raise FileNotFoundError(path)
    return SnapshotRegistry.load(path)


def _load_dev_eval_summaries(path: Path) -> dict[str, float | DevEvalPolicySummary]:
    payload = _load_json_object(path, label="dev-eval summaries")
    summaries: dict[str, float | DevEvalPolicySummary] = {}
    for policy_id, raw_summary in payload.items():
        if isinstance(raw_summary, bool):
            raise TypeError(f"dev-eval summary for {policy_id!r} cannot be a boolean")
        if isinstance(raw_summary, (int, float)):
            summaries[policy_id] = float(raw_summary)
            continue
        if not isinstance(raw_summary, dict):
            raise TypeError(
                "dev-eval summary values must be numbers or objects with aggregate_score/anchor_scores, "
                f"got {type(raw_summary).__name__} for {policy_id!r}"
            )
        aggregate_score = raw_summary.get("aggregate_score")
        if isinstance(aggregate_score, bool) or not isinstance(aggregate_score, (int, float)):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include numeric aggregate_score")
        anchor_scores = raw_summary.get("anchor_scores", {})
        if not isinstance(anchor_scores, dict) or any(not isinstance(key, str) for key in anchor_scores):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include object anchor_scores")
        summaries[policy_id] = DevEvalPolicySummary(
            policy_id=policy_id,
            aggregate_score=float(aggregate_score),
            anchor_scores=anchor_scores,
        )
    return summaries


def _selection_requires_snapshot_registry(stack: StackConfig) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    selection = evaluation.final_policy_set_selection
    return selection.include_final_champion_snapshot or bool(selection.include_spaced_snapshots_near_percent_updates)


def _selection_requires_dev_eval_summaries(stack: StackConfig) -> bool:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return False
    selection = evaluation.final_policy_set_selection
    fixed_slots = int(selection.include_random_legal_baseline_b0) + int(selection.include_no_league_baseline_b1)
    fixed_slots += int(selection.include_final_champion_snapshot)
    fixed_slots += len(selection.include_spaced_snapshots_near_percent_updates)
    if selection.include_heuristic_public_b2_if_exists:
        return True
    return evaluation.final_policy_set_size > fixed_slots


def _policy_set_selection(
    stack: StackConfig,
    *,
    snapshot_registry: SnapshotRegistry | None = None,
    dev_eval_summaries: Mapping[str, float | DevEvalPolicySummary] | None = None,
) -> list[str]:
    evaluation = stack.config.evaluation
    if evaluation is None:
        return []
    selection = evaluation.final_policy_set_selection
    if selection.version != "deterministic_v1":
        raise ValueError(f"unsupported final_policy_set_selection.version: {selection.version!r}")
    return select_final_policy_set_deterministic_v1(
        snapshot_registry=snapshot_registry or SnapshotRegistry(),
        dev_eval_summaries=dev_eval_summaries or {},
        config=selection,
        final_policy_set_size=evaluation.final_policy_set_size,
    )


def _resolve_policy_set_selection(
    stack: StackConfig,
    *,
    snapshot_registry_path: Path | None = None,
    dev_eval_summaries_path: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    evaluation = stack.config.evaluation
    source_paths = {
        "snapshot_registry_json": None
        if snapshot_registry_path is None
        else _manifest_source_path(snapshot_registry_path, root=stack.root),
        "dev_eval_summaries_json": None
        if dev_eval_summaries_path is None
        else _manifest_source_path(dev_eval_summaries_path, root=stack.root),
    }
    if evaluation is None:
        return [], {"mode": "not_configured", "status": "not_configured", "source_paths": source_paths}

    snapshot_registry = None if snapshot_registry_path is None else _load_snapshot_registry(snapshot_registry_path)
    dev_eval_summaries = None if dev_eval_summaries_path is None else _load_dev_eval_summaries(dev_eval_summaries_path)

    missing_inputs: list[str] = []
    if _selection_requires_snapshot_registry(stack) and snapshot_registry is None:
        missing_inputs.append("snapshot_registry_json")
    if _selection_requires_dev_eval_summaries(stack) and dev_eval_summaries is None:
        missing_inputs.append("dev_eval_summaries_json")

    details: dict[str, Any] = {
        "mode": evaluation.final_policy_set_selection.version,
        "status": "resolved",
        "version": evaluation.final_policy_set_selection.version,
        "final_policy_set_size": evaluation.final_policy_set_size,
        "source_paths": source_paths,
        "missing_inputs": missing_inputs,
    }
    if missing_inputs:
        details["mode"] = "unresolved"
        details["status"] = "unresolved"
        details["reason"] = "deterministic final policy set inputs were not provided"
        return [], details

    policy_ids = _policy_set_selection(
        stack,
        snapshot_registry=snapshot_registry,
        dev_eval_summaries=dev_eval_summaries,
    )
    details["selected_policy_count"] = len(policy_ids)
    return policy_ids, details


def _spec_mismatch_policy(stack: StackConfig) -> str:
    return "hard_fail"


def _resolve_run_label(parser: Any, run_label: str, run_id_alias: str) -> str:
    return _resolve_run_label_impl(parser, run_label, run_id_alias)


def _require_positive_int(name: str, value: int) -> int:
    return _require_positive_int_impl(name, value)


def _require_positive_optional_float(name: str, value: float | None) -> float | None:
    return _require_positive_optional_float_impl(name, value)


def _wall_clock_budget_seconds(max_wall_clock_minutes: float | None) -> float | None:
    return _wall_clock_budget_seconds_impl(max_wall_clock_minutes)


def _wall_clock_budget_reached(
    *,
    start_time: float,
    max_wall_clock_seconds: float | None,
    now: float | None = None,
) -> bool:
    return _wall_clock_budget_reached_impl(
        start_time=start_time,
        max_wall_clock_seconds=max_wall_clock_seconds,
        now=now,
    )


def _resolve_runtime_profile(stack: StackConfig, profile_override: str) -> str:
    return _resolve_runtime_profile_impl(stack, profile_override)


def _resolve_device(stack: StackConfig, device_override: str) -> torch.device:
    return _resolve_device_impl(stack, device_override)


def _resolve_seed(stack: StackConfig, seed_override: int | None) -> int:
    return _resolve_seed_impl(stack, seed_override)


def _manifest_scaffold_only_reason(stack: StackConfig) -> str | None:
    return _manifest_scaffold_only_reason_impl(stack)


def _runtime_training_prerequisite_failure(stack: StackConfig) -> str | None:
    return _runtime_training_prerequisite_failure_impl(stack)


def _print_manifest_only_message(reason: str) -> None:
    for line in _format_manifest_scaffold_only_message_impl(reason):
        print(line)


def _raise_runtime_prerequisite_failure(reason: str) -> None:
    raise RuntimeError(
        "Canonical simulator-backed training requires a weiss_sim runtime with stepping support. "
        f"Startup failed because {reason}."
    )


def _training_paths(run_dir: Path) -> TrainingPaths:
    return _build_training_paths(run_dir)


def _run_artifacts_from_existing_run_dir(run_dir: Path) -> RunArtifacts:
    return _run_artifacts_from_existing_run_dir_impl(run_dir)


def _configure_torch_threads(stack: StackConfig) -> None:
    system_config = stack.config.system
    if system_config is None:
        return
    torch.set_num_threads(int(system_config.learner_torch_threads))
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _torch_num_threads_scope(num_threads: int | None):
    return _torch_num_threads_scope_impl(num_threads)


def _central_runtime_actor_torch_threads(stack: StackConfig, runtime: QueueRuntime) -> int | None:
    system_config = stack.config.system
    if system_config is None:
        return None
    if str(system_config.actor_device).strip().lower() != "cpu":
        return None
    if bool(getattr(runtime, "_use_process_collectors", False)):
        return None
    if not bool(getattr(runtime, "_use_central_batched_collection", False)):
        return None
    return int(system_config.actor_torch_threads)


def _spec_dimensions(contract: SimulatorContract) -> tuple[int, int]:
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    return observation_dim, action_dim


def _env_pool_config(stack: StackConfig, *, seed: int) -> dict[str, Any]:
    return build_env_config_from_stack(stack, seed=int(seed))


def _build_env(
    stack: StackConfig,
    *,
    profile: str,
    num_envs: int,
    seed: int,
) -> DecisionBoundaryEnv:
    env_config = _env_pool_config(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(
        env_config,
        profile=profile,  # type: ignore[arg-type]
        num_envs=num_envs,
    )
    if layout_name != "mask":
        raise RuntimeError(
            "The compatibility training path expects mask legality because ImpalaLearner consumes legal_mask. "
            f"Profile {profile!r} resolved to layout {layout_name!r}."
        )
    max_no_progress_decisions = None
    curriculum = stack.config.curriculum
    if curriculum is not None:
        raw_limit = curriculum.simulator.get("max_no_progress_decisions")
        if raw_limit is not None:
            max_no_progress_decisions = int(raw_limit)
    return DecisionBoundaryEnv(
        pool,
        legality="mask",
        engine_status_policy="hard_fail",
        max_decisions=int(env_config["max_decisions"]),
        max_ticks=int(env_config["max_ticks"]),
        max_no_progress_decisions=max_no_progress_decisions,
    )


def _build_ids_eval_env(
    stack: StackConfig,
    *,
    seed: int,
    pass_action_id: int,
) -> DecisionBoundaryEnv:
    env_config = _env_pool_config(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(
        env_config,
        profile="fast",
        num_envs=1,
    )
    if layout_name != "i16_legal_ids":
        raise RuntimeError(
            "Periodic dev eval requires ids-based legality for the pinned eval protocol. "
            f"Profile 'fast' resolved to layout {layout_name!r}."
        )
    max_no_progress_decisions = None
    curriculum = stack.config.curriculum
    if curriculum is not None:
        raw_limit = curriculum.simulator.get("max_no_progress_decisions")
        if raw_limit is not None:
            max_no_progress_decisions = int(raw_limit)
    return DecisionBoundaryEnv(
        pool,
        legality="ids_offsets",
        pass_action_id=pass_action_id,
        engine_status_policy="hard_fail",
        max_decisions=int(env_config["max_decisions"]),
        max_ticks=int(env_config["max_ticks"]),
        max_no_progress_decisions=max_no_progress_decisions,
    )


def _bootstrap_values(
    model: PolicyValueModel,
    rollout: MinimalRollout,
    final_seat_hidden: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    return _bootstrap_values_impl(model, rollout, final_seat_hidden, device=device)


def _build_learner_batch(
    stack: StackConfig,
    rollout: MinimalRollout,
    bootstrap_value: np.ndarray,
    *,
    action_dim: int,
    initial_hidden_state: torch.Tensor,
    pass_action_id: int,
) -> dict[str, Any]:
    return _build_learner_batch_impl(
        stack,
        rollout,
        bootstrap_value,
        action_dim=action_dim,
        initial_hidden_state=initial_hidden_state,
        pass_action_id=pass_action_id,
    )


def _write_scalars_record(
    *,
    scalars_path: Path,
    learner: ImpalaLearner,
    metrics: dict[str, float],
    start_time: float,
) -> None:
    wall_clock_seconds = time.time() - start_time
    record = {
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_ms": int(wall_clock_seconds * 1000),
        **metrics,
    }
    with scalars_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Cannot write a checkpoint without a learner model")

    payload = {
        "format": "minimal_train_checkpoint_v1",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "device": str(device),
        "config_hash256": compute_config_hash256(stack),
        "spec_hash256": spec_hash256,
        "algorithm": algorithm,
        "recurrent_core": getattr(stack.config.model, "recurrent_core", None),
        "total_samples_processed": int(getattr(learner, "total_samples_processed", 0)),
        "model_state_dict": learner.model.state_dict(),
        **_model_guidance_payload(learner.model),
        "optimizer_state_dict": None if learner.optimizer is None else learner.optimizer.state_dict(),
        "grad_scaler_state_dict": (
            None if getattr(learner, "_grad_scaler", None) is None else learner._grad_scaler.state_dict()
        ),
    }
    torch.save(payload, checkpoint_path)
    return payload


def _relative_path_text(path: Path, *, root: Path) -> str:
    return _checkpoint_relative_path_text(path, root=root)


def _checkpoint_guard_log_path(training_paths: TrainingPaths) -> Path:
    return _checkpoint_guard_log_path_impl(training_paths)


def _b2_disagreement_audit_requests_path(training_paths: TrainingPaths) -> Path:
    return _b2_disagreement_audit_requests_path_impl(training_paths)


def _periodic_dev_eval_anchor_weight_map(stack: StackConfig) -> dict[str, float]:
    return _periodic_dev_eval_anchor_weight_map_impl(stack)


def _league_eval_warmup_gate_status(
    stack: StackConfig,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _league_eval_warmup_gate_status_impl(stack, dev_eval_summary)


def _sync_runtime_league_eval_warmup_gate(
    *,
    runtime: QueueRuntime,
    stack: StackConfig,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _sync_runtime_league_eval_warmup_gate_impl(
        runtime=runtime,
        stack=stack,
        dev_eval_summary=dev_eval_summary,
    )


def _dev_eval_batched_screen_enabled(dev_eval_summary: Mapping[str, Any] | None) -> bool:
    surface = _dev_eval_surface(dev_eval_summary)
    return str(surface.get("kind", "")).strip() == "fast_batched_screen"


def _periodic_dev_eval_fast_screens_path(training_paths: TrainingPaths) -> Path:
    return _periodic_dev_eval_fast_screens_path_impl(training_paths)


def _persist_periodic_dev_eval_fast_screen(
    *,
    training_paths: TrainingPaths,
    payload: Mapping[str, Any],
) -> None:
    _persist_periodic_dev_eval_fast_screen_impl(training_paths=training_paths, payload=payload)


def _build_periodic_dev_eval_summary_record(
    *,
    payload: Mapping[str, Any],
    prior_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    return _build_periodic_dev_eval_summary_record_impl(
        payload=payload,
        prior_summaries=prior_summaries,
        b2_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
    )


def _publish_checkpoint_aliases(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _publish_checkpoint_aliases_impl(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
        b2_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
    )


def _append_checkpoint_guard_event(training_paths: TrainingPaths, payload: Mapping[str, Any]) -> None:
    _append_checkpoint_guard_event_impl(training_paths, payload)


def _append_b2_disagreement_audit_request(training_paths: TrainingPaths, payload: Mapping[str, Any]) -> None:
    _append_b2_disagreement_audit_request_impl(training_paths, payload)


def _run_stack_config_path(artifacts: RunArtifacts) -> Path | None:
    if not artifacts.run_summary_path.is_file():
        return None
    run_summary = _load_json_object(artifacts.run_summary_path, label="run summary")
    raw_path = run_summary.get("stack_config_path")
    if not isinstance(raw_path, str) or not str(raw_path).strip():
        return None
    return Path(raw_path)


def _dev_eval_has_confidence_only_block(dev_eval_summary: Mapping[str, Any] | None, *, stack: StackConfig) -> bool:
    return _dev_eval_has_confidence_only_block_impl(dev_eval_summary, stack=stack)


def _maybe_request_b2_disagreement_audit(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    return _maybe_request_b2_disagreement_audit_impl(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        dev_eval_summary=dev_eval_summary,
        b2_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
    )


def _maybe_log_structured_mainmove_guard(
    *,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    return _maybe_log_structured_mainmove_guard_impl(
        training_paths=training_paths,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
    )


def _demote_registry_champions_newer_than(training_paths: TrainingPaths, *, update_count: int) -> list[str]:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return []
    registry = SnapshotRegistry.load(registry_path)
    removed = registry.demote_champions_newer_than(int(update_count))
    if removed:
        registry.save(registry_path)
    return removed


def _is_current_run_train_snapshot_for_rollback(training_paths: TrainingPaths, snapshot: SnapshotMeta) -> bool:
    policy_id = str(snapshot.policy_id).strip()
    if not policy_id.startswith("policy_"):
        return False
    metadata_path = training_paths.snapshots_dir / policy_id / SNAPSHOT_METADATA_FILENAME
    if metadata_path.is_file():
        try:
            metadata = _load_json_object(metadata_path, label="snapshot metadata")
        except Exception:
            metadata = {}
        if isinstance(metadata, Mapping) and (
            "imported_from_run_dir" in metadata
            or "imported_from_policy_id" in metadata
            or bool(metadata.get("seeded_from_external_registry", False))
        ):
            return False
    return True


def _reject_registry_snapshots_newer_than(training_paths: TrainingPaths, *, update_count: int) -> list[str]:
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return []
    registry = SnapshotRegistry.load(registry_path)
    update_count_i = int(update_count)
    rejected: list[str] = []
    for snapshot in registry.snapshots:
        if int(snapshot.update) <= update_count_i:
            continue
        if not _is_current_run_train_snapshot_for_rollback(training_paths, snapshot):
            continue
        registry.reject_snapshot(snapshot.policy_id)
        rejected.append(snapshot.policy_id)
    if rejected:
        registry.save(registry_path)
    return rejected


def _best_checkpoint_record(training_paths: TrainingPaths) -> Mapping[str, Any] | None:
    tracker = _load_checkpoint_tracker(training_paths)
    best_record = tracker.get("best")
    return best_record if isinstance(best_record, Mapping) else None


def _restore_checkpoint_to_latest_alias(
    *,
    checkpoint_path: Path,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
    restore_counters: bool = True,
) -> ResumeCheckpoint:
    resume_state = _restore_learner_from_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=expected_spec_hash256,
        algorithm=algorithm,
        restore_counters=restore_counters,
    )
    shutil.copy2(checkpoint_path, training_paths.latest_checkpoint_path)
    return resume_state


def _resolve_resume_checkpoint_path(
    *,
    resume_from: str,
    resume_run_dir: Path | None,
) -> Path | None:
    normalized = str(resume_from).strip()
    if not normalized:
        if resume_run_dir is None:
            return None
        normalized = "latest"
    alias_name = normalized.lower()
    if alias_name in {"latest", "best"}:
        if resume_run_dir is None:
            raise ValueError("--resume-from latest|best requires --resume-run-dir")
        filename = _LATEST_CHECKPOINT_FILENAME if alias_name == "latest" else _BEST_CHECKPOINT_FILENAME
        checkpoint_path = Path(resume_run_dir).resolve() / "training" / "checkpoints" / filename
    else:
        checkpoint_path = Path(normalized).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _infer_seed_snapshot_run_dir_from_resume_checkpoint(
    *,
    stack: StackConfig,
    resume_checkpoint_path: Path | None,
    resume_run_dir: Path | None,
) -> Path | None:
    if resume_checkpoint_path is None or resume_run_dir is not None:
        return None
    league = stack.config.league
    if league is None or not bool(league.enabled):
        return None
    checkpoint_path = Path(resume_checkpoint_path).resolve()
    checkpoint_dir = checkpoint_path.parent
    training_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints" or training_dir.name != "training":
        return None
    source_run_dir = training_dir.parent
    registry_path = source_run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    if not registry_path.is_file():
        return None
    return source_run_dir


def _infer_run_dir_from_checkpoint_path(checkpoint_path: Path | None) -> Path | None:
    return _infer_run_dir_from_checkpoint_path_impl(checkpoint_path)


def _seed_snapshot_import_max_update(
    *,
    resume_state: ResumeCheckpoint | None,
    seed_snapshot_run_dir: Path | None,
    seed_snapshot_run_dir_auto_inferred: bool,
) -> int | None:
    if resume_state is None or seed_snapshot_run_dir is None:
        return None
    if not bool(seed_snapshot_run_dir_auto_inferred):
        return None
    return int(resume_state.update_count)


def _load_resume_checkpoint_dev_eval_summary(
    *,
    stack: StackConfig,
    resume_checkpoint_path: Path,
    update_count: int,
    allow_config_hash_mismatch: bool = False,
) -> dict[str, Any] | None:
    checkpoint_path = Path(resume_checkpoint_path).resolve()
    checkpoint_dir = checkpoint_path.parent
    training_dir = checkpoint_dir.parent
    if checkpoint_dir.name != "checkpoints" or training_dir.name != "training":
        return None
    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(checkpoint_payload, Mapping):
        return None
    checkpoint_config_hash = str(checkpoint_payload.get("config_hash256", "")).strip().lower()
    if checkpoint_config_hash != compute_config_hash256(stack) and not bool(allow_config_hash_mismatch):
        return None

    source_run_dir = training_dir.parent
    for artifact_dir_name in ("dev_eval_confirmatory", "dev_eval"):
        summary_path = source_run_dir / "eval" / artifact_dir_name / f"update_{int(update_count)}" / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and _dev_eval_is_authoritative(payload):
            return payload
    return None


def _restore_learner_from_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
    restore_counters: bool = True,
    restore_optimizer_state: bool = True,
    allow_config_hash_mismatch: bool = False,
) -> ResumeCheckpoint:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict: {checkpoint_path}")
    if str(payload.get("format", "")).strip() != "minimal_train_checkpoint_v1":
        raise RuntimeError(f"unsupported checkpoint format in {checkpoint_path}")
    payload_config_hash = str(payload.get("config_hash256", "")).strip().lower()
    expected_config_hash = compute_config_hash256(stack)
    if payload_config_hash != expected_config_hash:
        if not allow_config_hash_mismatch:
            raise RuntimeError(
                f"checkpoint config hash mismatch for {checkpoint_path}: expected {expected_config_hash}, got {payload_config_hash}"
            )
        print(
            _format_resume_config_hash_mismatch_warning_impl(
                checkpoint_config_hash=payload_config_hash,
                current_config_hash=expected_config_hash,
            )
        )
    payload_spec_hash = payload.get("spec_hash256")
    if payload_spec_hash is not None and str(payload_spec_hash).strip().lower() != expected_spec_hash256:
        raise RuntimeError(
            f"checkpoint spec hash mismatch for {checkpoint_path}: expected {expected_spec_hash256}, got {payload_spec_hash}"
        )
    payload_algorithm = payload.get("algorithm")
    if payload_algorithm is not None and str(payload_algorithm).strip() and str(payload_algorithm).strip() != algorithm:
        raise RuntimeError(
            f"checkpoint algorithm mismatch for {checkpoint_path}: expected {algorithm}, got {payload_algorithm}"
        )
    model_state_dict = payload.get("model_state_dict")
    if learner.model is None or not isinstance(model_state_dict, dict):
        raise RuntimeError(f"checkpoint is missing a model_state_dict: {checkpoint_path}")
    learner.model.load_state_dict(model_state_dict)
    _restore_model_guidance_from_payload(learner.model, payload)
    optimizer_state_dict = payload.get("optimizer_state_dict")
    if restore_optimizer_state and optimizer_state_dict is not None:
        optimizer = learner._optimizer_for_step()
        optimizer.load_state_dict(optimizer_state_dict)
        for group in optimizer.param_groups:
            group["lr"] = float(learner.learning_rate)
    grad_scaler_state_dict = payload.get("grad_scaler_state_dict")
    if (
        restore_optimizer_state
        and grad_scaler_state_dict is not None
        and getattr(learner, "_grad_scaler", None) is not None
    ):
        learner._grad_scaler.load_state_dict(grad_scaler_state_dict)
    if restore_counters:
        learner.update_count = int(payload.get("update_count", 0))
        learner.policy_version = int(payload.get("policy_version", 0))
        learner.total_samples_processed = int(payload.get("total_samples_processed", 0))
        learner.start_time = time.time()
    return ResumeCheckpoint(
        checkpoint_path=checkpoint_path.resolve(),
        update_count=learner.update_count,
        policy_version=learner.policy_version,
        total_samples_processed=learner.total_samples_processed,
    )


def _build_training_learner(
    *,
    algorithm: str,
    model: PolicyValueModel,
    compiled_model: nn.Module | None,
    training_config: Any,
    training_paths: TrainingPaths,
    pass_action_id: int,
    checkpoint_interval_updates: int,
    gradient_sync: Any | None = None,
    artifact_writes_enabled: bool = True,
) -> ImpalaLearner | PpoLiteLearner:
    common_kwargs = {
        "model": model,
        "compiled_model": compiled_model,
        "learning_rate": training_config.learning_rate,
        "policy_loss_coef": float(getattr(training_config, "policy_loss_coef", 1.0)),
        "value_loss_coef": training_config.value_loss_coef,
        "entropy_coef": training_config.entropy_coef,
        "grad_norm_clip": training_config.grad_norm_clip,
        "optimizer_backend": str(getattr(training_config, "optimizer_backend", "auto")),
        "mixed_precision": bool(training_config.mixed_precision),
        "checkpoint_dir": training_paths.checkpoints_dir if artifact_writes_enabled else None,
        "checkpoint_interval_updates": int(checkpoint_interval_updates),
        "logs_dir": training_paths.logs_dir if artifact_writes_enabled else None,
        "logging_interval_updates": 1,
        "pass_action_id": pass_action_id,
        "teacher_family_coef": training_config.teacher_family_coef,
        "teacher_slot_coef": training_config.teacher_slot_coef,
        "teacher_move_source_coef": training_config.teacher_move_source_coef,
        "teacher_attack_type_coef": training_config.teacher_attack_type_coef,
        "teacher_action_coef": training_config.teacher_action_coef,
        "teacher_same_family_action_coef": training_config.teacher_same_family_action_coef,
        "teacher_public_heuristic_coef": training_config.teacher_public_heuristic_coef,
        "teacher_public_main_move_coef": training_config.teacher_public_main_move_coef,
        "teacher_development_pass_suppression_coef": training_config.teacher_development_pass_suppression_coef,
        "teacher_public_heuristic_temperature": training_config.teacher_public_heuristic_temperature,
        "teacher_public_heuristic_families": training_config.teacher_public_heuristic_families,
        "teacher_public_heuristic_profiles": training_config.teacher_public_heuristic_profiles,
        "teacher_public_heuristic_profile_mode": training_config.teacher_public_heuristic_profile_mode,
        "teacher_public_heuristic_profiles_end_updates": training_config.teacher_public_heuristic_profiles_end_updates,
        "behavior_action_bc_coef": float(getattr(training_config, "behavior_action_bc_coef", 0.0)),
        "b1_opponent_anchor_only": bool(getattr(training_config, "b1_opponent_anchor_only", False)),
        "reference_policy_top_action_bc_coef": float(
            getattr(training_config, "reference_policy_top_action_bc_coef", 0.0)
        ),
        "b1_opponent_reference_policy_top_action_bc_coef": float(
            getattr(training_config, "b1_opponent_reference_policy_top_action_bc_coef", 0.0)
        ),
        "b1_second_seat_positive_advantage_policy_coef": float(
            getattr(training_config, "b1_second_seat_positive_advantage_policy_coef", 0.0)
        ),
        "b1_second_seat_reference_top_action_avoidance_coef": float(
            getattr(training_config, "b1_second_seat_reference_top_action_avoidance_coef", 0.0)
        ),
        "reference_policy_top_action_family_bc_coef": float(
            getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0)
        ),
        "raw_b1_distill_coef": float(getattr(getattr(training_config, "raw_b1_distill", None), "coef", 0.0)),
        "raw_b1_distill_teacher_bias_scale": float(
            getattr(
                getattr(training_config, "raw_b1_distill", None),
                "teacher_public_heuristic_bias_scale",
                0.0,
            )
        ),
        "raw_b1_distill_student_bias_scale": float(
            getattr(
                getattr(training_config, "raw_b1_distill", None),
                "student_public_heuristic_bias_scale",
                0.0,
            )
        ),
        "raw_b1_distill_top_k": int(getattr(getattr(training_config, "raw_b1_distill", None), "top_k", 16)),
        "raw_b1_distill_temperature": float(
            getattr(getattr(training_config, "raw_b1_distill", None), "temperature", 1.5)
        ),
        "raw_b1_distill_top_action_ce_coef": float(
            getattr(getattr(training_config, "raw_b1_distill", None), "top_action_ce_coef", 0.0)
        ),
        "counterfactual_positive_label_dirs": tuple(
            getattr(getattr(training_config, "counterfactual_positive", None), "label_dirs", ()) or ()
        ),
        "counterfactual_positive_coef": float(
            getattr(getattr(training_config, "counterfactual_positive", None), "coef", 0.0)
        ),
        "counterfactual_positive_margin_coef": float(
            getattr(getattr(training_config, "counterfactual_positive", None), "margin_coef", 0.0)
        ),
        "counterfactual_positive_margin": float(
            getattr(getattr(training_config, "counterfactual_positive", None), "margin", 1.0)
        ),
        "counterfactual_positive_max_labels": int(
            getattr(getattr(training_config, "counterfactual_positive", None), "max_labels", 0)
        ),
        "profile_timers": bool(getattr(training_config, "profile_timers", False)),
        "structured_metrics_mode": str(getattr(training_config, "structured_metrics_mode", "full")),
        "teacher_aux_mode": str(getattr(training_config, "teacher_aux_mode", "always")),
        "gradient_sync": gradient_sync,
    }
    if algorithm in _IMPALA_ALGORITHMS:
        return ImpalaLearner(
            **common_kwargs,
            vtrace_rho_bar=training_config.vtrace_rho_bar,
            vtrace_c_bar=training_config.vtrace_c_bar,
        )
    if algorithm in _PPO_ALGORITHMS:
        return PpoLiteLearner(
            **common_kwargs,
            ppo_clip_epsilon=training_config.ppo_clip_epsilon,
            value_clip_epsilon=training_config.ppo_value_clip_epsilon,
            ppo_epochs=int(training_config.ppo_epochs),
            target_kl=training_config.ppo_target_kl,
            normalize_advantages=bool(training_config.ppo_normalize_advantages),
        )
    raise RuntimeError(f"Unsupported training.algorithm: {algorithm}")


def _entropy_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return _entropy_coef_for_next_update_impl(training_config, update_count=update_count)


def _teacher_public_heuristic_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return _teacher_public_heuristic_coef_for_next_update_impl(training_config, update_count=update_count)


def _reference_policy_top_action_bc_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return _reference_policy_top_action_bc_coef_for_next_update_impl(training_config, update_count=update_count)


def _reference_policy_top_action_family_bc_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return _reference_policy_top_action_family_bc_coef_for_next_update_impl(
        training_config,
        update_count=update_count,
    )


def _raw_b1_distill_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return _raw_b1_distill_coef_for_next_update_impl(training_config, update_count=update_count)


def _counterfactual_positive_coef_for_next_update(training_config: Any, *, update_count: int) -> float:
    return _counterfactual_positive_coef_for_next_update_impl(training_config, update_count=update_count)


def _public_heuristic_logit_bias_scale_for_next_update(model_config: Any, *, update_count: int) -> float:
    return _public_heuristic_logit_bias_scale_for_next_update_impl(model_config, update_count=update_count)


def _public_heuristic_actor_logit_bias_scale_for_next_update(
    model_config: Any,
    *,
    learner_bias_scale: float,
) -> float:
    return _public_heuristic_actor_logit_bias_scale_for_next_update_impl(
        model_config,
        learner_bias_scale=learner_bias_scale,
    )


def _apply_guidance_schedule_for_next_update(
    *,
    learner: ImpalaLearner,
    model: PolicyValueModel | None,
    stack: StackConfig,
    update_count: int,
) -> dict[str, float]:
    return _apply_guidance_schedule_for_next_update_impl(
        learner=learner,
        model=model,
        stack=stack,
        update_count=update_count,
    )


def _model_guidance_payload(model: PolicyValueModel | None) -> dict[str, float]:
    return _model_guidance_payload_impl(model)


def _restore_model_guidance_from_payload(
    model: PolicyValueModel | None,
    payload: Mapping[str, Any],
) -> None:
    _restore_model_guidance_from_payload_impl(model, payload)


def _maybe_compile_learner_model(
    *,
    model: PolicyValueModel,
    training_config: Any,
    device: torch.device,
) -> nn.Module | None:
    return _maybe_compile_learner_model_impl(model=model, training_config=training_config, device=device)


def _profile_block(enabled: bool, name: str):
    return _profile_block_impl(enabled, name)


def _build_training_profiler(
    *,
    enabled: bool,
    run_dir: Path,
    device: torch.device,
) -> tuple[torch.profiler.profile | None, Any, Path | None]:
    return _build_training_profiler_impl(enabled=enabled, run_dir=run_dir, device=device)


def _collect_training_batch(
    *,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
) -> Any:
    return _collect_training_batch_impl(
        runtime=runtime,
        algorithm=algorithm,
        training_config=training_config,
        rewards_config=rewards_config,
    )


def _collect_training_batch_prefetch(
    *,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
    actor_torch_threads: int | None,
) -> Any:
    return _collect_training_batch_prefetch_impl(
        runtime=runtime,
        algorithm=algorithm,
        training_config=training_config,
        rewards_config=rewards_config,
        actor_torch_threads=actor_torch_threads,
    )


def _run_structured_warmstart(
    *,
    learner: ImpalaLearner,
    runtime: QueueRuntime,
    algorithm: str,
    training_config: Any,
    rewards_config: Any,
    training_paths: TrainingPaths,
    tensorboard_logger: TensorBoardLogger | None,
    start_time: float,
    profile_timers: bool = False,
    actor_torch_threads: int | None = None,
    learner_torch_threads: int | None = None,
) -> dict[str, float]:
    if not bool(getattr(training_config, "structured_warmstart_enabled", False)):
        return {}
    if algorithm not in _IMPALA_ALGORITHMS:
        raise RuntimeError("structured warmstart currently supports only IMPALA learners")
    warmstart_cfg = training_config.structured_warmstart
    updates = int(warmstart_cfg.updates)
    if updates <= 0:
        return {}

    previous_family = float(training_config.teacher_family_coef)
    previous_slot = float(training_config.teacher_slot_coef)
    previous_move_source = float(training_config.teacher_move_source_coef)
    previous_attack_type = float(training_config.teacher_attack_type_coef)
    previous_action = float(training_config.teacher_action_coef)
    previous_same_family_action = float(training_config.teacher_same_family_action_coef)
    previous_public_heuristic = float(training_config.teacher_public_heuristic_coef)
    previous_public_heuristic_temperature = float(training_config.teacher_public_heuristic_temperature)
    previous_public_heuristic_families = tuple(training_config.teacher_public_heuristic_families)
    previous_public_heuristic_profiles = tuple(training_config.teacher_public_heuristic_profiles)
    previous_public_heuristic_profile_mode = str(training_config.teacher_public_heuristic_profile_mode)
    previous_public_heuristic_profiles_end_updates = int(training_config.teacher_public_heuristic_profiles_end_updates)
    learner.set_teacher_aux_coefs(
        family=float(warmstart_cfg.teacher_family_coef),
        slot=float(warmstart_cfg.teacher_slot_coef),
        move_source=float(warmstart_cfg.teacher_move_source_coef),
        attack_type=float(warmstart_cfg.teacher_attack_type_coef),
        action=float(warmstart_cfg.teacher_action_coef),
        same_family_action=float(warmstart_cfg.teacher_same_family_action_coef),
        public_heuristic=float(warmstart_cfg.teacher_public_heuristic_coef),
        public_heuristic_temperature=float(warmstart_cfg.teacher_public_heuristic_temperature),
        public_heuristic_families=tuple(warmstart_cfg.teacher_public_heuristic_families),
        public_heuristic_profiles=tuple(warmstart_cfg.teacher_public_heuristic_profiles),
        public_heuristic_profile_mode=str(warmstart_cfg.teacher_public_heuristic_profile_mode),
        public_heuristic_profiles_end_updates=int(warmstart_cfg.teacher_public_heuristic_profiles_end_updates),
    )
    latest_metrics: dict[str, float] = {}
    try:
        with (
            runtime.structured_warmstart_source_mix() as warmstart_source_metrics,
            runtime.disable_mirror_policy_fusion(),
        ):
            for warmstart_step in range(updates):
                with (
                    _profile_block(profile_timers, "collect_training_batch"),
                    _torch_num_threads_scope(actor_torch_threads),
                ):
                    runtime_batch = _collect_training_batch(
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                    )
                with (
                    _profile_block(profile_timers, "learner_auxiliary_update"),
                    _torch_num_threads_scope(learner_torch_threads),
                ):
                    latest_metrics = learner.auxiliary_update(runtime_batch.learner_batch)
                latest_metrics.update(runtime_batch.runtime_metrics)
                latest_metrics.update(warmstart_source_metrics)
                latest_metrics["warmstart_phase"] = 1.0
                latest_metrics["warmstart_step"] = float(warmstart_step + 1)
                _write_scalars_record(
                    scalars_path=training_paths.scalars_path,
                    learner=learner,
                    metrics=latest_metrics,
                    start_time=start_time,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time.time() - start_time,
                        metrics=latest_metrics,
                    )
    finally:
        learner.set_teacher_aux_coefs(
            family=previous_family,
            slot=previous_slot,
            move_source=previous_move_source,
            attack_type=previous_attack_type,
            action=previous_action,
            same_family_action=previous_same_family_action,
            public_heuristic=previous_public_heuristic,
            public_heuristic_temperature=previous_public_heuristic_temperature,
            public_heuristic_families=previous_public_heuristic_families,
            public_heuristic_profiles=previous_public_heuristic_profiles,
            public_heuristic_profile_mode=previous_public_heuristic_profile_mode,
            public_heuristic_profiles_end_updates=previous_public_heuristic_profiles_end_updates,
        )
    return latest_metrics


def _validate_algorithm_model_contract(*, algorithm: str, recurrent_core: str, encoder_kind: str) -> None:
    normalized_core = str(recurrent_core).strip().lower()
    normalized_encoder = str(encoder_kind).strip().lower()
    if algorithm == "impala_vtrace_gru" and normalized_core != "gru":
        raise RuntimeError("impala_vtrace_gru requires model.recurrent_core=gru")
    if algorithm == "impala_vtrace_ff" and normalized_core != "none":
        raise RuntimeError("impala_vtrace_ff requires model.recurrent_core=none")
    if algorithm in {"structured_v2", "impala_vtrace_structured_v1"} and normalized_core not in {"gru", "none"}:
        raise RuntimeError(f"{algorithm} requires a supported model.recurrent_core value")
    if algorithm in {"structured_v2", "impala_vtrace_structured_v1"} and normalized_encoder != "structured_v2":
        raise RuntimeError(f"{algorithm} requires model.encoder_kind=structured_v2")


def _json_relative_path(path: Path, *, root: Path) -> str:
    return _json_relative_path_impl(path, root=root)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug_policy_id(value: str) -> str:
    return _slug_policy_id_impl(value)


def _promotion_anchor_policy_id_candidates(anchor_name: str) -> tuple[str, ...]:
    return _promotion_anchor_policy_id_candidates_impl(anchor_name)


def _resolve_symbolic_promotion_anchor_policy_id(
    anchor_name: str,
    *,
    registry: SnapshotRegistry,
    promotion_gate_enabled: bool = False,
) -> str | None:
    return _resolve_symbolic_promotion_anchor_policy_id_impl(
        anchor_name,
        registry=registry,
        promotion_gate_enabled=promotion_gate_enabled,
    )


def _true_local_recent_snapshot_ids(
    registry: SnapshotRegistry,
    *,
    promotion_gate_enabled: bool = False,
) -> tuple[str, ...]:
    return _true_local_recent_snapshot_ids_impl(registry, promotion_gate_enabled=promotion_gate_enabled)


def _build_heuristic_public_policy(
    spec_bundle: Mapping[str, object],
    *,
    scoring_profile: str,
) -> HeuristicPublicPolicy:
    factory = HeuristicPublicPolicy.from_spec_bundle
    supports_scoring_profile = False
    try:
        supports_scoring_profile = "scoring_profile" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        supports_scoring_profile = False
    if supports_scoring_profile:
        return factory(spec_bundle, scoring_profile=scoring_profile)
    return factory(spec_bundle)


def _find_noleague_baseline_snapshot(run_dir: Path) -> SnapshotMeta | None:
    return _find_noleague_baseline_snapshot_impl(run_dir)


def _import_noleague_baseline_anchor(
    *,
    training_paths: TrainingPaths,
    run_dir: Path,
    baseline_run_dir: Path,
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> tuple[Path, str, int]:
    return _import_noleague_baseline_anchor_impl(
        training_paths=training_paths,
        run_dir=run_dir,
        baseline_run_dir=baseline_run_dir,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )


def _attach_reference_policy_model_if_configured(
    *,
    learner: ImpalaLearner | PpoLiteLearner,
    training_config: Any,
    training_paths: TrainingPaths,
    model_config: Any,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, Any] | None,
    spec_bundle: Mapping[str, Any],
    device: torch.device,
) -> None:
    if not isinstance(learner, ImpalaLearner):
        return
    coef = float(getattr(training_config, "reference_policy_top_action_bc_coef", 0.0))
    family_coef = float(getattr(training_config, "reference_policy_top_action_family_bc_coef", 0.0))
    raw_b1_distill = getattr(training_config, "raw_b1_distill", None)
    raw_b1_distill_enabled = bool(getattr(raw_b1_distill, "enabled", False)) and (
        float(getattr(raw_b1_distill, "coef", 0.0)) != 0.0 or float(getattr(raw_b1_distill, "final_coef", 0.0)) != 0.0
    )
    if coef == 0.0 and family_coef == 0.0 and not raw_b1_distill_enabled:
        return
    policy_id = str(getattr(training_config, "reference_policy_id", "") or "").strip()
    if raw_b1_distill_enabled:
        raw_policy_id = str(getattr(raw_b1_distill, "teacher_policy_id", "") or "").strip()
        if raw_policy_id:
            if policy_id and raw_policy_id != policy_id and (coef != 0.0 or family_coef != 0.0):
                raise ValueError(
                    "training.raw_b1_distill.teacher_policy_id must match training.reference_policy_id "
                    "when both reference BC and raw B1 distill are enabled"
                )
            policy_id = raw_policy_id
    if not policy_id:
        policy_id = _PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID
    weights_path = training_paths.snapshots_dir / policy_id / SNAPSHOT_WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"reference policy weights not found for policy_id={policy_id!r}: {weights_path}")
    payload = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"reference policy weights payload is missing model_state_dict: {weights_path}")
    reference_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(device)
    reference_model.load_state_dict(payload["model_state_dict"])
    _restore_model_guidance_from_payload(reference_model, payload)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    learner.reference_policy_model = reference_model
    print(
        _format_attached_reference_policy_message_impl(
            policy_id=policy_id,
            coef=coef,
            family_coef=family_coef,
            raw_b1_distill_enabled=raw_b1_distill_enabled,
            weights_path=weights_path,
        )
    )


def _validate_seed_snapshot_import_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    _validate_seed_snapshot_import_contract_impl(
        source_run_dir=source_run_dir,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )


def _validate_snapshot_tensor_contract(
    *,
    label: str,
    source_path: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
) -> None:
    _validate_snapshot_tensor_contract_impl(
        label=label,
        source_path=source_path,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
    )


def _seed_snapshot_policy_id(*, source_run_dir: Path, source_policy_id: str) -> str:
    return _seed_snapshot_policy_id_impl(source_run_dir=source_run_dir, source_policy_id=source_policy_id)


def _import_seed_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    seed_snapshot_run_dir: Path,
    max_update: int | None = None,
    exclude_source_policy_ids: Sequence[str] = (),
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> list[str]:
    return _import_seed_snapshot_pool_impl(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        max_update=max_update,
        exclude_source_policy_ids=exclude_source_policy_ids,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )


def _source_snapshot_is_resume_league_snapshot(snapshot: SnapshotMeta, *, rejected_policy_ids: set[str]) -> bool:
    return _source_snapshot_is_resume_league_snapshot_impl(snapshot, rejected_policy_ids=rejected_policy_ids)


def _validate_existing_resume_league_import(
    *,
    training_paths: TrainingPaths,
    source_run_dir: Path,
    source_snapshot: SnapshotMeta,
) -> None:
    _validate_existing_resume_league_import_impl(
        training_paths=training_paths,
        source_run_dir=source_run_dir,
        source_snapshot=source_snapshot,
    )


def _import_resume_league_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    resume_checkpoint_path: Path,
    max_update: int,
    expected_model_state_dict: dict[str, Any],
) -> list[str]:
    return _import_resume_league_snapshot_pool_impl(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        resume_checkpoint_path=resume_checkpoint_path,
        max_update=max_update,
        expected_model_state_dict=expected_model_state_dict,
    )


def _ensure_noleague_baseline_anchor(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    learner: ImpalaLearner,
    device: torch.device,
    config_hash256: str,
    spec_hash256: str | None = None,
    baseline_run_dir: Path | None = None,
    permit_current_run_alias: bool = False,
    source_checkpoint_path: Path | None = None,
    update: int | None = None,
) -> str | None:
    training_config = stack.config.training
    if learner.model is None:
        raise RuntimeError("Cannot ensure the NoLeague baseline anchor without a learner model")

    def write_baseline_checkpoint(checkpoint_path: Path) -> None:
        _write_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            algorithm=str(training_config.algorithm).strip() if training_config is not None else None,
            spec_hash256=spec_hash256,
        )

    result = _ensure_noleague_baseline_anchor_impl(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        model_state_dict=learner.model.state_dict(),
        learner_update_count=int(learner.update_count),
        device=device,
        config_hash256=config_hash256,
        expected_config_canonical=canonical_config_dict(stack),
        spec_hash256=spec_hash256,
        baseline_run_dir=baseline_run_dir,
        permit_current_run_alias=permit_current_run_alias,
        source_checkpoint_path=source_checkpoint_path,
        update=update,
        write_checkpoint=write_baseline_checkpoint,
        guidance_payload=_model_guidance_payload(learner.model),
        experiment_role=_experiment_role(stack),
    )
    if result.message is not None:
        print(result.message)
    return result.policy_id


def _resolve_promotion_anchor_policy_ids(
    *,
    stack: StackConfig,
    registry: SnapshotRegistry,
) -> tuple[dict[str, str], tuple[str, ...]]:
    return _resolve_promotion_anchor_policy_ids_impl(stack=stack, registry=registry)


def _snapshot_meta_by_policy_id(registry: SnapshotRegistry) -> dict[str, Any]:
    return _snapshot_meta_by_policy_id_impl(registry)


def _get_cached_eval_snapshot_model(cache_key: tuple[Any, ...]) -> PolicyValueModel | None:
    return _get_cached_eval_model_impl(_EVAL_SNAPSHOT_MODEL_CACHE, cache_key)


def _remember_eval_snapshot_model(cache_key: tuple[Any, ...], eval_model: PolicyValueModel) -> None:
    _remember_eval_model_impl(
        _EVAL_SNAPSHOT_MODEL_CACHE,
        cache_key,
        eval_model,
        max_entries=_EVAL_SNAPSHOT_MODEL_CACHE_MAX_ENTRIES,
    )


def _load_snapshot_eval_model(
    *,
    run_dir: Path,
    snapshot_path: str,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    eval_device: torch.device | str | None = None,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    resolved_snapshot_path = (run_dir / snapshot_path).resolve()
    stat = resolved_snapshot_path.stat()
    resolved_eval_device = _resolve_eval_device(stack, eval_device=eval_device)
    cache_key = (
        str(resolved_snapshot_path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(resolved_eval_device),
        int(observation_dim),
        int(action_dim),
    )
    cached_model = _get_cached_eval_snapshot_model(cache_key)
    if cached_model is not None:
        return cached_model

    payload = torch.load(resolved_snapshot_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Snapshot weights payload missing model_state_dict: {snapshot_path}")

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")

    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(resolved_eval_device)
    eval_model.load_state_dict(
        {name: value.detach().to(device=resolved_eval_device).clone() for name, value in model_state_dict.items()}
    )
    _restore_model_guidance_from_payload(eval_model, payload)
    eval_model.eval()
    _remember_eval_snapshot_model(cache_key, eval_model)
    return eval_model


def _load_checkpoint_eval_model(
    *,
    checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    eval_device: torch.device | str | None = None,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise RuntimeError(f"Checkpoint payload missing model_state_dict: {checkpoint_path}")

    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")

    resolved_eval_device = _resolve_eval_device(stack, eval_device=eval_device)
    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(resolved_eval_device)
    eval_model.load_state_dict(
        {name: value.detach().to(device=resolved_eval_device).clone() for name, value in model_state_dict.items()}
    )
    _restore_model_guidance_from_payload(eval_model, payload)
    eval_model.eval()
    return eval_model


class _PromotionGateRunner(PromotionGateRunnerCore):
    def __init__(
        self,
        *,
        stack: StackConfig,
        focal_policy_id: str,
        focal_model: PolicyValueModel,
        anchor_models: dict[str, PolicyValueModel],
        heuristic_policies: dict[str, HeuristicPublicPolicy],
        observation_dim: int,
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        require_sorted_legal_ids: bool,
        eval_device: torch.device | str | None = None,
    ) -> None:
        self.stack = stack
        self.observation_dim = observation_dim
        super().__init__(
            focal_policy_id=focal_policy_id,
            focal_model=focal_model,
            anchor_models=anchor_models,
            heuristic_policies=heuristic_policies,
            action_dim=action_dim,
            pass_action_id=pass_action_id,
            artifact_dir=artifact_dir,
            require_sorted_legal_ids=require_sorted_legal_ids,
            eval_device=_resolve_eval_device(stack, eval_device=eval_device),
            randomlegal_policy_id=_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID,
            build_ids_eval_env=lambda seed: _build_ids_eval_env(
                stack,
                seed=seed,
                pass_action_id=pass_action_id,
            ),
            legal_ids_for_env_row=lambda batch, env_index, require_sorted: _legal_ids_for_env_row(
                batch=batch,
                env_index=env_index,
                require_sorted=require_sorted,
            ),
            promotion_gate_rng_seed=_promotion_gate_rng_seed,
        )


def _evaluation_config_or_raise(stack: StackConfig):
    return _evaluation_config_or_raise_impl(stack)


def _validate_periodic_dev_eval_contract(stack: StackConfig) -> Any:
    return _validate_periodic_dev_eval_contract_impl(stack)


def _resolve_repo_path(root: Path, path_text: str) -> Path:
    return _resolve_repo_path_impl(root, path_text)


def _action_family_ids_from_spec(*, action_dim: int, spec_bundle: Mapping[str, Any]) -> tuple[torch.Tensor | None, int]:
    try:
        catalog = ActionCatalog.from_spec_bundle(spec_bundle)
    except Exception:
        return None, 0
    family_names = tuple(family.name for family in catalog.families)
    family_index = {name: index for index, name in enumerate(family_names)}
    ids = torch.full((int(action_dim),), -1, dtype=torch.long)
    for action_id in range(int(action_dim)):
        try:
            decoded = catalog.decode(action_id)
        except Exception:
            continue
        ids[action_id] = int(family_index.get(decoded.family, -1))
    return ids, len(family_names)


def _load_main_residual_base_model(
    *,
    stack: StackConfig,
    checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, Any] | None,
    spec_bundle: Mapping[str, Any],
    device: torch.device,
) -> PolicyValueModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"main residual base checkpoint is missing model_state_dict: {checkpoint_path}")
    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("main residual base model requires stack.config.model")
    base_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(device)
    base_model.load_state_dict(payload["model_state_dict"])
    _restore_model_guidance_from_payload(base_model, payload)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    return base_model


def _maybe_build_main_residual_model(
    *,
    stack: StackConfig,
    observation_dim: int,
    action_dim: int,
    observation_spec: Mapping[str, Any] | None,
    spec_bundle: Mapping[str, Any],
    device: torch.device,
) -> nn.Module | None:
    training_config = stack.config.training
    if training_config is None:
        return None
    residual_config = getattr(training_config, "main_residual_policy", None)
    if residual_config is None or not bool(getattr(residual_config, "enabled", False)):
        return None
    checkpoint_path = _resolve_repo_path(stack.root, str(residual_config.base_snapshot_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"training.main_residual_policy.base_snapshot_path not found: {checkpoint_path}")
    base_model = _load_main_residual_base_model(
        stack=stack,
        checkpoint_path=checkpoint_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
        device=device,
    )
    set_bias = getattr(base_model, "set_public_heuristic_logit_bias_scale", None)
    if callable(set_bias):
        scale = float(getattr(residual_config, "public_heuristic_bias_scale", 1.0))
        set_bias(scale, actor_value=scale)
    initial_state_path_text = str(getattr(residual_config, "initial_residual_state_path", "") or "").strip()
    if initial_state_path_text:
        initial_state_path = _resolve_repo_path(stack.root, initial_state_path_text)
        if not initial_state_path.is_file():
            raise FileNotFoundError(
                f"training.main_residual_policy.initial_residual_state_path not found: {initial_state_path}"
            )
        residual = load_frozen_stored_logit_residual(initial_state_path, device=device)
        if int(residual.action_dim) != int(action_dim):
            raise RuntimeError(
                "training.main_residual_policy.initial_residual_state_path action_dim mismatch: "
                f"expected {action_dim}, got {residual.action_dim}"
            )
        residual.alpha = float(getattr(residual_config, "alpha", residual.alpha))
        residual.train()
    else:
        action_family_ids: torch.Tensor | None = None
        family_count = 0
        if str(getattr(residual_config, "residual_mode", "plain")) == "family_gated":
            action_family_ids, family_count = _action_family_ids_from_spec(
                action_dim=action_dim,
                spec_bundle=spec_bundle,
            )
        residual = FrozenStoredLogitResidual(
            obs_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=int(getattr(residual_config, "hidden_dim", 256)),
            alpha=float(getattr(residual_config, "alpha", 0.1)),
            residual_mode=str(getattr(residual_config, "residual_mode", "plain")),
            action_family_ids=action_family_ids,
            family_count=family_count,
            gate_bias=float(getattr(residual_config, "gate_bias", 0.0)),
        ).to(device)
    wrapper = TrainableLiveFrozenB1Residual(base_model=base_model, residual_probe=residual).to(device)
    print(
        _format_trainable_main_residual_policy_enabled_message_impl(
            checkpoint_path=checkpoint_path,
            alpha=float(residual.alpha),
            hidden_dim=int(getattr(residual_config, "hidden_dim", 256)),
            residual_mode=str(residual.residual_mode),
            initial_state_path_text=initial_state_path_text,
        )
    )
    return wrapper


def _resolve_periodic_dev_eval_seed_file(stack: StackConfig) -> tuple[Path, dict[str, str]]:
    return _resolve_periodic_dev_eval_seed_file_impl(stack)


def _periodic_dev_eval_schedule(stack: StackConfig) -> tuple[Path, dict[str, str], list[int], str]:
    return _periodic_dev_eval_schedule_impl(stack)


def _legal_ids_for_env_row(
    *,
    batch: DecisionBoundaryBatch,
    env_index: int,
    require_sorted: bool,
) -> np.ndarray:
    if batch.ids_offsets is None:
        raise RuntimeError("Expected ids_offsets legality during periodic dev eval")
    legal_ids, legal_offsets = batch.ids_offsets
    start = int(legal_offsets[env_index])
    end = int(legal_offsets[env_index + 1])
    row = np.asarray(legal_ids[start:end], dtype=np.uint32)
    if require_sorted:
        assert_strictly_increasing_legal_ids(row)
    return row


def _clone_eval_model(
    *,
    learner_model: PolicyValueModel,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    eval_device: torch.device | str | None = None,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    model_config = stack.config.model
    if model_config is None:
        raise RuntimeError("The locked stack is missing the model config block")
    resolved_eval_device = _resolve_eval_device(stack, eval_device=eval_device)
    eval_model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(resolved_eval_device)
    eval_state_dict = {
        name: value.detach().to(device=resolved_eval_device).clone()
        for name, value in learner_model.state_dict().items()
    }
    eval_model.load_state_dict(eval_state_dict)
    _restore_model_guidance_from_payload(eval_model, _model_guidance_payload(learner_model))
    eval_model.eval()
    return eval_model


def _current_focal_policy_id(*, learner: ImpalaLearner) -> str:
    return f"train_u{int(learner.update_count)}_p{int(learner.get_policy_version())}"


def _checkpoint_path_for_update(checkpoints_dir: Path, *, update_count: int) -> Path:
    return checkpoints_dir / f"checkpoint_{update_count}.pt"


def _ensure_current_checkpoint(
    *,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
) -> Path:
    checkpoint_path = _checkpoint_path_for_update(
        training_paths.checkpoints_dir,
        update_count=int(learner.update_count),
    )
    if checkpoint_path.is_file():
        return checkpoint_path

    _write_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    return checkpoint_path


def _should_run_periodic_dev_eval(stack: StackConfig, *, update_count: int) -> bool:
    return _should_run_periodic_dev_eval_impl(stack, update_count=update_count)


def _should_defer_noleague_baseline_alias_refresh(
    *,
    stack: StackConfig,
    experiment_role: str,
    update_count: int,
) -> bool:
    return _should_defer_noleague_baseline_alias_refresh_impl(
        stack=stack,
        experiment_role=experiment_role,
        update_count=update_count,
    )


def _periodic_dev_eval_summaries_path(training_paths: TrainingPaths) -> Path:
    return _periodic_dev_eval_summaries_path_impl(training_paths)


def _periodic_dev_eval_opponents(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
) -> list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]]:
    evaluation = _evaluation_config_or_raise(stack)
    registry_path = ArtifactLayout.from_run_dir(run_dir).training_snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path) if registry_path.is_file() else SnapshotRegistry()
    anchor_policy_ids, missing_required = _resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        missing_text = ",".join(missing_required)
        raise RuntimeError(f"Periodic dev eval is missing required anchors: {missing_text}")

    league = stack.config.league
    anchor_names: list[str]
    if league is None:
        anchor_names = [_PROMOTION_GATE_RANDOMLEGAL_NAME, _PROMOTION_GATE_NOLEAGUE_BASELINE_NAME]
    else:
        anchor_names = [
            *league.promotion_anchor_set_v1.required,
            *league.promotion_anchor_set_v1.optional_if_available,
        ]

    snapshot_index = _snapshot_meta_by_policy_id(registry)
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    opponents: list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]] = []
    for anchor_name in anchor_names:
        policy_id = anchor_policy_ids.get(anchor_name)
        if policy_id is None:
            continue
        if policy_id == _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID:
            opponents.append((policy_id, anchor_name, None, None))
            continue
        heuristic_profile = heuristic_public_profile_name_for_policy_id(policy_id)
        if heuristic_profile is not None:
            try:
                heuristic_policy = _build_heuristic_public_policy(
                    contract.spec_bundle,
                    scoring_profile=heuristic_profile,
                )
            except Exception as exc:
                if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                    raise RuntimeError(
                        f"Periodic dev eval requires a heuristic-compatible simulator contract for {policy_id}"
                    ) from exc
                continue
            opponents.append((policy_id, anchor_name, None, heuristic_policy))
            continue
        snapshot = snapshot_index.get(policy_id)
        if snapshot is None:
            if league is not None and anchor_name in league.promotion_anchor_set_v1.required:
                raise RuntimeError(f"Periodic dev eval could not resolve required snapshot anchor {anchor_name!r}")
            continue
        opponents.append(
            (
                policy_id,
                anchor_name,
                _load_snapshot_eval_model(
                    run_dir=run_dir,
                    snapshot_path=snapshot.path,
                    stack=stack,
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    eval_device=evaluation.eval_device,
                    observation_spec=observation_spec,
                    spec_bundle=spec_bundle,
                ),
                None,
            )
        )
    return opponents


def _resolve_periodic_dev_eval_opponent_specs(
    *,
    stack: StackConfig,
    run_dir: Path,
) -> tuple[tuple[PeriodicDevEvalOpponentSpec, ...], tuple[str, ...]]:
    return _resolve_periodic_dev_eval_opponent_specs_impl(stack=stack, run_dir=run_dir)


def _materialize_periodic_dev_eval_opponents(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device_override: torch.device | str | None = None,
) -> list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]]:
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    evaluation = _evaluation_config_or_raise(stack)
    opponents: list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]] = []
    for spec in opponent_specs:
        if spec.kind == "random_legal":
            opponents.append((spec.policy_id, spec.display_name, None, None))
            continue
        if spec.kind == "heuristic_public":
            heuristic_profile = str(spec.heuristic_profile or "").strip()
            if not heuristic_profile:
                raise RuntimeError(f"Periodic dev eval heuristic opponent is missing a profile: {spec.policy_id}")
            heuristic_policy = _build_heuristic_public_policy(
                contract.spec_bundle,
                scoring_profile=heuristic_profile,
            )
            opponents.append((spec.policy_id, spec.display_name, None, heuristic_policy))
            continue
        if spec.kind != "snapshot" or spec.snapshot_path is None:
            raise RuntimeError(f"Unsupported periodic dev eval opponent kind: {spec.kind!r}")
        opponents.append(
            (
                spec.policy_id,
                spec.display_name,
                _load_snapshot_eval_model(
                    run_dir=run_dir,
                    snapshot_path=spec.snapshot_path,
                    stack=stack,
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
                    observation_spec=observation_spec,
                    spec_bundle=spec_bundle,
                ),
                None,
            )
        )
    return opponents


def _persist_periodic_dev_eval_summary(
    *,
    training_paths: TrainingPaths,
    payload: Mapping[str, Any],
) -> None:
    _persist_periodic_dev_eval_summary_impl(
        training_paths=training_paths,
        payload=payload,
        b2_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
    )


def _persist_periodic_dev_eval_result(
    *,
    training_paths: TrainingPaths,
    payload: Mapping[str, Any],
    force_summary: bool = False,
) -> str:
    return _persist_periodic_dev_eval_result_impl(
        training_paths=training_paths,
        payload=payload,
        b2_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
        force_summary=force_summary,
    )


def _publish_best_checkpoint_from_dev_eval(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    checkpoint_path: Path,
    update_count: int,
    policy_version: int,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _publish_best_checkpoint_from_dev_eval_impl(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        update_count=update_count,
        policy_version=policy_version,
        dev_eval_summary=dev_eval_summary,
        b2_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
    )


def _update_stall_monitor(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    return _update_stall_monitor_impl(
        stack=stack,
        training_paths=training_paths,
        update_count=update_count,
        summary_payload=summary_payload,
    )


def _update_early_cutoff(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    return _update_early_cutoff_impl(
        stack=stack,
        training_paths=training_paths,
        update_count=update_count,
        summary_payload=summary_payload,
    )


def _maybe_rollback_to_best_checkpoint(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    runtime: QueueRuntime,
    learner: ImpalaLearner,
    model: PolicyValueModel,
    device: torch.device,
    spec_hash256: str,
    algorithm: str,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
    last_rollback_update: int | None,
) -> dict[str, Any] | None:
    tracker = _load_checkpoint_tracker(training_paths)
    best_record = tracker.get("best")
    if not isinstance(best_record, Mapping):
        best_record = None
    rollback_plan = _checkpoint_guard_rollback_plan(
        stack=stack,
        learner_update_count=int(learner.update_count),
        last_rollback_update=last_rollback_update,
        best_record=cast(Mapping[str, Any] | None, best_record),
        dev_eval_summary=dev_eval_summary,
    )
    if rollback_plan is None:
        return None

    pre_rollback_update_count = int(learner.update_count)
    pre_rollback_policy_version = int(learner.get_policy_version())
    best_checkpoint_path = training_paths.best_checkpoint_path
    _restore_checkpoint_to_latest_alias(
        checkpoint_path=best_checkpoint_path,
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=spec_hash256,
        algorithm=algorithm,
        restore_counters=False,
    )
    learner.update_count = pre_rollback_update_count
    learner.policy_version = max(int(learner.get_policy_version()), pre_rollback_policy_version)
    demoted_champions = _demote_registry_champions_newer_than(
        training_paths,
        update_count=int(rollback_plan.best_update_count),
    )
    rejected_snapshots = _reject_registry_snapshots_newer_than(
        training_paths,
        update_count=int(rollback_plan.best_update_count),
    )
    publish_metrics = runtime.maybe_publish_snapshot(
        learner_model=model,
        learner_update_count=int(learner.update_count),
        force=True,
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()
    tracker["latest"] = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=best_checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind="dev_eval_mean",
        metric_value=rollback_plan.best_score,
    )
    _write_checkpoint_tracker(training_paths, tracker)

    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "rollback_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "restored_weight_update_count": int(rollback_plan.best_update_count),
        "current_score": rollback_plan.current_score,
        "best_score": rollback_plan.best_score,
        "best_update_count": int(rollback_plan.best_update_count),
        "worst_stall_rate": rollback_plan.worst_stall_rate,
        "worst_truncation_rate": rollback_plan.worst_truncation_rate,
        "worst_no_progress_timeout_rate": rollback_plan.worst_no_progress_timeout_rate,
        "worst_natural_timeout_rate": rollback_plan.worst_natural_timeout_rate,
        "min_prob_gt_half": rollback_plan.min_prob_gt_half,
        "max_prob_lt_half": rollback_plan.max_prob_lt_half,
        "max_ci_half_width": rollback_plan.max_ci_half_width,
        "reasons": list(rollback_plan.reasons),
        "best_checkpoint_path": _relative_path_text(best_checkpoint_path, root=artifacts.run_dir),
        "latest_checkpoint_path": _relative_path_text(training_paths.latest_checkpoint_path, root=artifacts.run_dir),
        "snapshot_publish_latency_ms": publish_metrics.get("snapshot_publish_latency_ms", 0.0),
        "snapshot_apply_latency_ms": publish_metrics.get("snapshot_apply_latency_ms", 0.0),
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "demoted_champions": demoted_champions,
        "rejected_snapshots": rejected_snapshots,
    }
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _maybe_finalize_from_best_checkpoint(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    artifacts: RunArtifacts,
    runtime: QueueRuntime,
    learner: ImpalaLearner,
    device: torch.device,
    spec_hash256: str,
    algorithm: str,
    latest_metrics: Mapping[str, float] | None,
    dev_eval_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    curriculum = stack.config.curriculum
    if curriculum is None or not curriculum.checkpoint_guard.enabled:
        return None
    if dev_eval_summary is not None and not _dev_eval_is_authoritative(dev_eval_summary):
        return None
    best_record = _best_checkpoint_record(training_paths)
    if best_record is None:
        return None
    best_metric_kind = str(best_record.get("metric_kind", "")).strip()
    best_metric_value = best_record.get("metric_value")
    best_update_count = best_record.get("update_count")
    if best_metric_kind != "dev_eval_mean":
        return None
    if not isinstance(best_metric_value, (int, float)) or not np.isfinite(float(best_metric_value)):
        return None
    if not isinstance(best_update_count, int):
        return None
    current_score = _dev_eval_aggregate_score(dev_eval_summary)
    best_score = float(best_metric_value)
    if current_score is None or current_score >= best_score:
        return None
    confidence = _dev_eval_confidence_stats(dev_eval_summary)
    best_checkpoint_path = training_paths.best_checkpoint_path
    _restore_checkpoint_to_latest_alias(
        checkpoint_path=best_checkpoint_path,
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        expected_spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    demoted_champions = _demote_registry_champions_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    rejected_snapshots = _reject_registry_snapshots_newer_than(
        training_paths,
        update_count=int(best_update_count),
    )
    runtime.reset_outcome_tracker()
    runtime.refresh_opponent_pool()
    tracker = _load_checkpoint_tracker(training_paths)
    tracker["latest"] = _build_checkpoint_record(
        alias_name="latest",
        alias_path=training_paths.latest_checkpoint_path,
        source_checkpoint_path=best_checkpoint_path,
        artifacts=artifacts,
        learner=learner,
        metric_kind="dev_eval_mean",
        metric_value=best_score,
    )
    _write_checkpoint_tracker(training_paths, tracker)
    payload = {
        "format": "checkpoint_guard_event_v1",
        "action": "finalize_to_best",
        "update_count": int(learner.update_count),
        "policy_version": int(learner.get_policy_version()),
        "current_score": current_score,
        "best_score": best_score,
        "best_update_count": int(best_update_count),
        "min_prob_gt_half": confidence["min_prob_gt_half"],
        "max_prob_lt_half": confidence["max_prob_lt_half"],
        "max_ci_half_width": confidence["max_ci_half_width"],
        "latest_loss": None if latest_metrics is None else latest_metrics.get("loss"),
        "best_checkpoint_path": _relative_path_text(best_checkpoint_path, root=artifacts.run_dir),
        "latest_checkpoint_path": _relative_path_text(training_paths.latest_checkpoint_path, root=artifacts.run_dir),
        "demoted_champions": demoted_champions,
        "rejected_snapshots": rejected_snapshots,
    }
    _append_checkpoint_guard_event(training_paths, payload)
    return payload


def _resolved_periodic_dev_eval_worker_devices(
    *,
    stack: StackConfig,
    parallel_workers: int,
    explicit_worker_devices: Sequence[str],
    eval_device: str,
    learner_device: torch.device | None = None,
) -> tuple[str, ...]:
    return _resolved_periodic_dev_eval_worker_devices_impl(
        stack=stack,
        parallel_workers=parallel_workers,
        explicit_worker_devices=explicit_worker_devices,
        eval_device=eval_device,
        learner_device=learner_device,
        actor_device_layout_resolver=resolve_actor_device_layout,
    )


def _validate_parallel_worker_device_pool(device_pool: Sequence[str], *, source: str) -> None:
    _validate_parallel_worker_device_pool_impl(device_pool, source=source)


def _shard_periodic_dev_eval_opponents(
    *,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    shard_count: int,
) -> list[list[PeriodicDevEvalOpponentSpec]]:
    return _shard_periodic_dev_eval_opponents_impl(opponent_specs=opponent_specs, shard_count=shard_count)


def _periodic_dev_eval_duplicate_policy_ids(
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
) -> set[str]:
    return _periodic_dev_eval_duplicate_policy_ids_impl(opponent_specs)


def _periodic_dev_eval_matchup_dir(
    *,
    update_dir: Path,
    opponent_spec: PeriodicDevEvalOpponentSpec,
    duplicate_policy_ids: set[str],
) -> Path:
    return _periodic_dev_eval_matchup_dir_impl(
        update_dir=update_dir,
        opponent_spec=opponent_spec,
        duplicate_policy_ids=duplicate_policy_ids,
    )


def _split_periodic_dev_eval_seed_blocks(
    paired_seeds: Sequence[int],
    *,
    block_count: int,
) -> list[tuple[tuple[int, int], ...]]:
    return _split_periodic_dev_eval_seed_blocks_impl(paired_seeds, block_count=block_count)


def _build_periodic_dev_eval_seed_block_jobs(
    *,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    paired_seeds: Sequence[int],
    configured_parallel_workers: int,
) -> list[PeriodicDevEvalSeedBlockJob]:
    return _build_periodic_dev_eval_seed_block_jobs_impl(
        opponent_specs=opponent_specs,
        paired_seeds=paired_seeds,
        configured_parallel_workers=configured_parallel_workers,
    )


def _shard_periodic_dev_eval_seed_block_jobs(
    *,
    jobs: Sequence[PeriodicDevEvalSeedBlockJob],
    shard_count: int,
) -> list[list[PeriodicDevEvalSeedBlockJob]]:
    return _shard_periodic_dev_eval_seed_block_jobs_impl(jobs=jobs, shard_count=shard_count)


def _periodic_dev_eval_schedule_for_seed_items(
    *,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seed_items: Sequence[tuple[int, int]],
) -> list[ScheduledGame]:
    return _periodic_dev_eval_schedule_for_seed_items_impl(
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        paired_seed_items=paired_seed_items,
    )


def _sum_periodic_dev_eval_counter_payloads(counter_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _sum_periodic_dev_eval_counter_payloads_impl(counter_payloads)


def _resolve_async_periodic_dev_eval_device(
    *,
    stack: StackConfig,
    learner_device: torch.device,
    distributed_context: DistributedContext | None = None,
) -> str | None:
    evaluation = _evaluation_config_or_raise(stack)
    requested = str(evaluation.eval_device).strip()
    if not requested:
        return None
    normalized = requested.lower()
    if normalized not in {"auto", "cuda:auto"}:
        return requested
    if distributed_context is not None and distributed_context.enabled and learner_device.type == "cuda":
        return str(learner_device)
    actor_count = 1 if stack.config.system is None else int(stack.config.system.actor_process_count)
    actor_layout = resolve_actor_device_layout(
        stack,
        actor_count=actor_count,
        learner_device=learner_device,
        prefer_process_collectors=True,
    )
    unique_cuda_devices = [
        device_name
        for device_name in dict.fromkeys(actor_layout)
        if torch.device(device_name).type == "cuda" and str(device_name) != str(learner_device)
    ]
    if unique_cuda_devices:
        return str(unique_cuda_devices[-1])
    return requested


def _resolve_async_promotion_gate_device(
    *,
    stack: StackConfig,
    learner_device: torch.device,
    distributed_context: DistributedContext | None = None,
) -> str | None:
    evaluation = _evaluation_config_or_raise(stack)
    requested = str(evaluation.eval_device).strip()
    if not requested:
        return None
    normalized = requested.lower()
    if normalized not in {"auto", "cuda:auto"}:
        return requested
    if distributed_context is not None and distributed_context.enabled and learner_device.type == "cuda":
        return str(learner_device)
    actor_count = 1 if stack.config.system is None else int(stack.config.system.actor_process_count)
    actor_layout = resolve_actor_device_layout(
        stack,
        actor_count=actor_count,
        learner_device=learner_device,
        prefer_process_collectors=True,
    )
    unique_cuda_devices = [
        device_name
        for device_name in dict.fromkeys(actor_layout)
        if torch.device(device_name).type == "cuda" and str(device_name) != str(learner_device)
    ]
    if unique_cuda_devices:
        return str(unique_cuda_devices[0])
    return requested


def _resolve_promotion_gate_anchor_specs(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
) -> tuple[dict[str, str], tuple[PeriodicDevEvalOpponentSpec, ...], tuple[str, ...]]:
    return _resolve_promotion_gate_anchor_specs_impl(
        stack=stack,
        snapshots_dir=training_paths.snapshots_dir,
    )


def _resolved_promotion_gate_worker_devices(
    *,
    stack: StackConfig,
    parallel_workers: int,
    explicit_worker_devices: Sequence[str],
    eval_device: str,
) -> tuple[str, ...]:
    return _resolved_promotion_gate_worker_devices_impl(
        stack=stack,
        parallel_workers=parallel_workers,
        explicit_worker_devices=explicit_worker_devices,
        eval_device=eval_device,
    )


def _shard_promotion_gate_anchor_specs(
    *,
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    shard_count: int,
) -> list[list[tuple[int, PeriodicDevEvalOpponentSpec]]]:
    return _shard_promotion_gate_anchor_specs_impl(anchor_specs=anchor_specs, shard_count=shard_count)


def _build_promotion_gate_seed_block_jobs(
    *,
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    paired_seeds: Sequence[int],
    configured_parallel_workers: int,
) -> list[PromotionGateSeedBlockJob]:
    return _build_promotion_gate_seed_block_jobs_impl(
        anchor_specs=anchor_specs,
        paired_seeds=paired_seeds,
        configured_parallel_workers=configured_parallel_workers,
    )


def _shard_promotion_gate_seed_block_jobs(
    *,
    jobs: Sequence[PromotionGateSeedBlockJob],
    shard_count: int,
) -> list[list[PromotionGateSeedBlockJob]]:
    return _shard_promotion_gate_seed_block_jobs_impl(jobs=jobs, shard_count=shard_count)


def _run_periodic_dev_eval_matchups_for_opponents(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str,
    artifact_scope: str,
    paired_seeds: Sequence[int],
    scheduled_paired_seed_count: int,
    validated_sources: Mapping[str, str],
    seed_file: Path,
    seed_file_sha256: str,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device_override: torch.device | str | None,
    batched_inference_override: bool | None = None,
) -> list[dict[str, Any]]:
    evaluation = _validate_periodic_dev_eval_contract(stack)
    observation_dim, action_dim = _spec_dimensions(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    update_dir = run_dir / "eval" / artifact_dir_name / f"update_{update_count}"
    eval_model = _load_checkpoint_eval_model(
        checkpoint_path=checkpoint_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
    )
    opponents = _materialize_periodic_dev_eval_opponents(
        stack=stack,
        contract=contract,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        opponent_specs=opponent_specs,
        eval_device_override=eval_device_override,
    )
    matchup_results: list[dict[str, Any]] = []
    duplicate_policy_ids = _periodic_dev_eval_duplicate_policy_ids(opponent_specs)
    for opponent_spec, (opponent_policy_id, display_name, opponent_model, heuristic_policy) in zip(
        opponent_specs,
        opponents,
        strict=True,
    ):
        matchup_dir = _periodic_dev_eval_matchup_dir(
            update_dir=update_dir,
            opponent_spec=opponent_spec,
            duplicate_policy_ids=duplicate_policy_ids,
        )
        runner = _PeriodicDevEvalRunner(
            stack=stack,
            model=eval_model,
            opponent_policy_id=opponent_policy_id,
            opponent_model=opponent_model,
            heuristic_policy=heuristic_policy,
            observation_dim=observation_dim,
            action_dim=action_dim,
            pass_action_id=pass_action_id,
            artifact_dir=matchup_dir,
            focal_policy_id=focal_policy_id,
            require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
            eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
        )
        matchup_started = time.perf_counter()

        seed_usage_payload = _build_periodic_dev_eval_seed_usage_payload_impl(
            seed_file=seed_file,
            seed_root=stack.root,
            seed_file_sha256=seed_file_sha256,
            validated_sources=validated_sources,
            artifact_scope=artifact_scope,
            scheduled_paired_seed_count=scheduled_paired_seed_count,
            paired_seeds=paired_seeds,
            seat_swap=bool(evaluation.seat_swap),
            eval_device=str(evaluation.eval_device if eval_device_override is None else eval_device_override),
            eval_inference_mode=bool(evaluation.eval_inference_mode),
            eval_sampling_algorithm=evaluation.eval_sampling_algorithm,
            eval_assert_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
            focal_policy_id=focal_policy_id,
            update_count=update_count,
            policy_version=policy_version,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            opponent_policy_id=opponent_policy_id,
            opponent_display_name=display_name,
        )
        _write_json(matchup_dir / "seed_usage.json", seed_usage_payload)

        batched_inference_enabled = (
            bool(getattr(evaluation, "periodic_dev_eval_batched_inference_enabled", False))
            if batched_inference_override is None
            else bool(batched_inference_override)
        )
        try:
            if batched_inference_enabled:
                scheduled_games = build_seat_swapped_schedule(
                    focal_policy_id=focal_policy_id,
                    opponent_policy_id=opponent_policy_id,
                    paired_seeds=paired_seeds,
                )
                completed_games = runner.run_scheduled_games_batched(scheduled_games)
                records = tuple(
                    record_completed_game(
                        scheduled_game=scheduled_game,
                        result=result,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    for scheduled_game, result in completed_games
                )
                write_episodes_jsonl(matchup_dir / "episodes.jsonl", records)
            else:
                matchup = run_seat_swapped_matchup(
                    focal_policy_id=focal_policy_id,
                    opponent_policy_id=opponent_policy_id,
                    paired_seeds=paired_seeds,
                    runner=runner,
                    episodes_path=matchup_dir / "episodes.jsonl",
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                )
                records = matchup.records
            runner_counters = runner.drain_runtime_counters()
        finally:
            close_runner = getattr(runner, "close", None)
            if callable(close_runner):
                close_runner()
        matchup_wall_clock_seconds = max(0.0, time.perf_counter() - matchup_started)

        matchup_payload = build_matchup_export(
            records,
            stop_rules=evaluation.stop_rules,
            max_paired_seeds=len(paired_seeds),
            scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
            sample_count=1000,
            seed=_periodic_dev_eval_bootstrap_seed(update_count=update_count, policy_version=policy_version),
        )
        seat_diagnostics = build_seat_advantage_diagnostics(records)
        matchup_payload["seat_diagnostics"] = seat_diagnostics
        matchup_payload["evaluation_context"] = _build_periodic_dev_eval_matchup_context_payload_impl(
            artifact_scope=artifact_scope,
            update_count=update_count,
            policy_version=policy_version,
            checkpoint_path=checkpoint_path,
            matchup_dir=matchup_dir,
            run_dir=run_dir,
            anchor_display_name=display_name,
        )
        matchup_payload["evaluation_runtime"] = _build_periodic_dev_eval_matchup_runtime_payload_impl(
            wall_clock_seconds=matchup_wall_clock_seconds,
            game_count=len(records),
            runner_counters=runner_counters,
            batched_model_inference=batched_inference_enabled,
        )
        write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
        write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", matchup_payload)
        write_matchup_diagnostics_json(
            matchup_dir / "diagnostics.json",
            seat_diagnostics,
        )
        matchup_results.append(
            {
                "policy_id": opponent_policy_id,
                "display_name": display_name,
                "matchup_dir": matchup_dir,
                "episodes_path": matchup_dir / "episodes.jsonl",
                "matchup_payload": matchup_payload,
            }
        )
    return matchup_results


def _run_periodic_dev_eval_matchup_worker(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str,
    artifact_scope: str,
    paired_seeds: Sequence[int],
    scheduled_paired_seed_count: int,
    validated_sources: Mapping[str, str],
    seed_file: Path,
    seed_file_sha256: str,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec],
    eval_device_override: str,
    batched_inference_override: bool | None = None,
) -> list[dict[str, Any]]:
    return _run_periodic_dev_eval_matchups_for_opponents(
        stack=stack,
        contract=contract,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        focal_policy_id=focal_policy_id,
        update_count=update_count,
        policy_version=policy_version,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        artifact_dir_name=artifact_dir_name,
        artifact_scope=artifact_scope,
        paired_seeds=paired_seeds,
        scheduled_paired_seed_count=scheduled_paired_seed_count,
        validated_sources=validated_sources,
        seed_file=seed_file,
        seed_file_sha256=seed_file_sha256,
        opponent_specs=opponent_specs,
        eval_device_override=eval_device_override,
        batched_inference_override=batched_inference_override,
    )


def _run_periodic_dev_eval_seed_block_worker(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str,
    jobs: Sequence[PeriodicDevEvalSeedBlockJob],
    eval_device_override: str,
    worker_index: int,
    batched_inference_override: bool | None = None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    evaluation = _validate_periodic_dev_eval_contract(stack)
    batched_inference_enabled = (
        bool(getattr(evaluation, "periodic_dev_eval_batched_inference_enabled", False))
        if batched_inference_override is None
        else bool(batched_inference_override)
    )
    observation_dim, action_dim = _spec_dimensions(contract)
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    update_dir = run_dir / "eval" / artifact_dir_name / f"update_{update_count}"
    worker_artifact_dir = update_dir / "_seed_block_workers" / f"worker_{worker_index}"
    eval_model = _load_checkpoint_eval_model(
        checkpoint_path=checkpoint_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
    )

    unique_specs: list[PeriodicDevEvalOpponentSpec] = []
    unique_spec_keys: set[tuple[str, str, str, str | None, str | None]] = set()
    for job in jobs:
        key = (
            job.opponent_spec.policy_id,
            job.opponent_spec.display_name,
            job.opponent_spec.kind,
            job.opponent_spec.snapshot_path,
            job.opponent_spec.heuristic_profile,
        )
        if key in unique_spec_keys:
            continue
        unique_spec_keys.add(key)
        unique_specs.append(job.opponent_spec)

    materialized = _materialize_periodic_dev_eval_opponents(
        stack=stack,
        contract=contract,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        opponent_specs=tuple(unique_specs),
        eval_device_override=eval_device_override,
    )
    opponent_by_key = {
        (
            spec.policy_id,
            spec.display_name,
            spec.kind,
            spec.snapshot_path,
            spec.heuristic_profile,
        ): opponent
        for spec, opponent in zip(unique_specs, materialized, strict=True)
    }
    runners: dict[tuple[str, str, str, str | None, str | None], _PeriodicDevEvalRunner] = {}
    results: list[dict[str, Any]] = []
    try:
        for job in jobs:
            key = (
                job.opponent_spec.policy_id,
                job.opponent_spec.display_name,
                job.opponent_spec.kind,
                job.opponent_spec.snapshot_path,
                job.opponent_spec.heuristic_profile,
            )
            opponent_policy_id, display_name, opponent_model, heuristic_policy = opponent_by_key[key]
            runner = runners.get(key)
            if runner is None:
                runner = _PeriodicDevEvalRunner(
                    stack=stack,
                    model=eval_model,
                    opponent_policy_id=opponent_policy_id,
                    opponent_model=opponent_model,
                    heuristic_policy=heuristic_policy,
                    observation_dim=observation_dim,
                    action_dim=action_dim,
                    pass_action_id=pass_action_id,
                    artifact_dir=worker_artifact_dir / f"opponent_{job.opponent_index}",
                    focal_policy_id=focal_policy_id,
                    require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
                    eval_device=(evaluation.eval_device if eval_device_override is None else eval_device_override),
                )
                runners[key] = runner
            block_started = time.perf_counter()
            scheduled_games = _periodic_dev_eval_schedule_for_seed_items(
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
                paired_seed_items=job.paired_seed_items,
            )
            if batched_inference_enabled:
                completed_games = runner.run_scheduled_games_batched(scheduled_games)
                records = tuple(
                    record_completed_game(
                        scheduled_game=scheduled_game,
                        result=result,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    for scheduled_game, result in completed_games
                )
            else:
                records = tuple(
                    record_completed_game(
                        scheduled_game=scheduled_game,
                        result=runner.run_game(scheduled_game),
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                    )
                    for scheduled_game in scheduled_games
                )
            results.append(
                {
                    "opponent_index": int(job.opponent_index),
                    "block_index": int(job.block_index),
                    "policy_id": opponent_policy_id,
                    "display_name": display_name,
                    "paired_seed_items": tuple(job.paired_seed_items),
                    "records": records,
                    "wall_clock_seconds": max(0.0, time.perf_counter() - block_started),
                    "runner_counters": runner.drain_runtime_counters(),
                    "worker_index": int(worker_index),
                    "worker_device": str(eval_device_override),
                }
            )
    finally:
        for runner in runners.values():
            runner.close()
    return results


def _run_periodic_dev_eval_for_checkpoint(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    checkpoint_path: Path,
    focal_policy_id: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    opponent_specs: Sequence[PeriodicDevEvalOpponentSpec] | None = None,
    eval_device_override: torch.device | str | None = None,
    parallel_workers_override: int | None = None,
    parallel_worker_devices_override: Sequence[str] | None = None,
    artifact_dir_name: str = "dev_eval",
    artifact_scope: str = "periodic_dev_eval",
    paired_seeds_override: Sequence[int] | None = None,
    batched_inference_override: bool | None = None,
    process_pool_executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    evaluation = _validate_periodic_dev_eval_contract(stack)
    seed_file, validated_sources, scheduled_paired_seeds, seed_file_sha256 = _periodic_dev_eval_schedule(stack)
    eval_started = time.perf_counter()
    paired_seeds = (
        [int(seed) for seed in paired_seeds_override]
        if paired_seeds_override is not None
        else [int(seed) for seed in scheduled_paired_seeds]
    )
    if not paired_seeds:
        raise RuntimeError("Periodic dev eval requires at least one paired seed")
    batched_inference_enabled = (
        bool(getattr(evaluation, "periodic_dev_eval_batched_inference_enabled", False))
        if batched_inference_override is None
        else bool(batched_inference_override)
    )
    observation_dim, action_dim = _spec_dimensions(contract)
    update_dir = run_dir / "eval" / artifact_dir_name / f"update_{update_count}"
    resolved_opponent_specs = opponent_specs
    if resolved_opponent_specs is None:
        resolved_opponent_specs, _ignored_pinned_ids = _resolve_periodic_dev_eval_opponent_specs(
            stack=stack,
            run_dir=run_dir,
        )
    requested_eval_device = evaluation.eval_device if eval_device_override is None else str(eval_device_override)
    configured_parallel_workers = max(
        1,
        int(
            getattr(evaluation, "periodic_dev_eval_parallel_workers", 1)
            if parallel_workers_override is None
            else parallel_workers_override
        ),
    )
    explicit_parallel_devices = tuple(
        getattr(evaluation, "periodic_dev_eval_parallel_worker_devices", ())
        if parallel_worker_devices_override is None
        else parallel_worker_devices_override
    )
    seed_block_jobs = _build_periodic_dev_eval_seed_block_jobs(
        opponent_specs=resolved_opponent_specs,
        paired_seeds=tuple(paired_seeds),
        configured_parallel_workers=configured_parallel_workers,
    )
    effective_parallel_workers = min(configured_parallel_workers, max(1, len(seed_block_jobs)))
    matchup_results: list[dict[str, Any]]
    worker_devices: tuple[str, ...]
    worker_devices = _resolved_periodic_dev_eval_worker_devices(
        stack=stack,
        parallel_workers=max(1, effective_parallel_workers),
        explicit_worker_devices=explicit_parallel_devices,
        eval_device=requested_eval_device,
        learner_device=None,
    )
    if effective_parallel_workers > 1:
        job_shards = _shard_periodic_dev_eval_seed_block_jobs(
            jobs=seed_block_jobs,
            shard_count=effective_parallel_workers,
        )
        executor_context = (
            nullcontext(process_pool_executor)
            if process_pool_executor is not None
            else ProcessPoolExecutor(max_workers=len(job_shards), mp_context=mp.get_context("spawn"))
        )
        with executor_context as executor:
            assert executor is not None
            futures = [
                executor.submit(
                    _run_periodic_dev_eval_seed_block_worker,
                    stack=stack,
                    contract=contract,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    focal_policy_id=focal_policy_id,
                    update_count=update_count,
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                    artifact_dir_name=artifact_dir_name,
                    jobs=tuple(shard),
                    eval_device_override=worker_devices[shard_index],
                    worker_index=shard_index,
                    batched_inference_override=batched_inference_enabled,
                )
                for shard_index, shard in enumerate(job_shards)
            ]
            block_results: list[dict[str, Any]] = []
            for future in futures:
                block_results.extend(future.result())

        duplicate_policy_ids = _periodic_dev_eval_duplicate_policy_ids(resolved_opponent_specs)
        block_results_by_opponent = _group_periodic_dev_eval_seed_block_results_impl(block_results)
        matchup_results = []
        for opponent_index, opponent_spec in enumerate(resolved_opponent_specs):
            collated_matchup = _collate_periodic_dev_eval_seed_block_matchup_impl(
                block_results_by_opponent=block_results_by_opponent,
                opponent_index=opponent_index,
                opponent_display_name=opponent_spec.display_name,
            )
            matchup_dir = _periodic_dev_eval_matchup_dir(
                update_dir=update_dir,
                opponent_spec=opponent_spec,
                duplicate_policy_ids=duplicate_policy_ids,
            )
            records = cast(tuple[EvalGameRecord, ...], collated_matchup["records"])
            seed_usage_payload = _build_periodic_dev_eval_seed_usage_payload_impl(
                seed_file=seed_file,
                seed_root=stack.root,
                seed_file_sha256=seed_file_sha256,
                validated_sources=validated_sources,
                artifact_scope=artifact_scope,
                scheduled_paired_seed_count=len(scheduled_paired_seeds),
                paired_seeds=paired_seeds,
                seat_swap=bool(evaluation.seat_swap),
                eval_device=str(requested_eval_device),
                eval_inference_mode=bool(evaluation.eval_inference_mode),
                eval_sampling_algorithm=evaluation.eval_sampling_algorithm,
                eval_assert_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
                focal_policy_id=focal_policy_id,
                update_count=update_count,
                policy_version=policy_version,
                checkpoint_path=checkpoint_path,
                run_dir=run_dir,
                opponent_policy_id=opponent_spec.policy_id,
                opponent_display_name=opponent_spec.display_name,
                parallel_seed_blocks=cast(Sequence[Mapping[str, Any]], collated_matchup["parallel_seed_blocks"]),
            )
            _write_json(matchup_dir / "seed_usage.json", seed_usage_payload)
            write_episodes_jsonl(matchup_dir / "episodes.jsonl", records)
            matchup_payload = build_matchup_export(
                records,
                stop_rules=evaluation.stop_rules,
                max_paired_seeds=len(paired_seeds),
                scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
                sample_count=1000,
                seed=_periodic_dev_eval_bootstrap_seed(update_count=update_count, policy_version=policy_version),
            )
            seat_diagnostics = build_seat_advantage_diagnostics(records)
            matchup_payload["seat_diagnostics"] = seat_diagnostics
            matchup_payload["evaluation_context"] = _build_periodic_dev_eval_matchup_context_payload_impl(
                artifact_scope=artifact_scope,
                update_count=update_count,
                policy_version=policy_version,
                checkpoint_path=checkpoint_path,
                matchup_dir=matchup_dir,
                run_dir=run_dir,
                anchor_display_name=opponent_spec.display_name,
            )
            matchup_payload["evaluation_runtime"] = _build_periodic_dev_eval_matchup_runtime_payload_impl(
                wall_clock_seconds=float(collated_matchup["wall_clock_seconds"]),
                game_count=len(records),
                runner_counters=cast(Mapping[str, Any], collated_matchup["runner_counters"]),
                batched_model_inference=batched_inference_enabled,
                seed_block_count=int(collated_matchup["seed_block_count"]),
                serial_worker_wall_clock_seconds_sum=float(collated_matchup["serial_worker_wall_clock_seconds_sum"]),
            )
            write_matchup_summary_json(matchup_dir / "matchup_summary.json", matchup_payload)
            write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", matchup_payload)
            write_matchup_diagnostics_json(
                matchup_dir / "diagnostics.json",
                seat_diagnostics,
            )
            matchup_results.append(
                {
                    "policy_id": opponent_spec.policy_id,
                    "display_name": opponent_spec.display_name,
                    "matchup_dir": matchup_dir,
                    "episodes_path": matchup_dir / "episodes.jsonl",
                    "matchup_payload": matchup_payload,
                }
            )
    else:
        matchup_results = _run_periodic_dev_eval_matchups_for_opponents(
            stack=stack,
            contract=contract,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            focal_policy_id=focal_policy_id,
            update_count=update_count,
            policy_version=policy_version,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            artifact_dir_name=artifact_dir_name,
            artifact_scope=artifact_scope,
            paired_seeds=tuple(paired_seeds),
            scheduled_paired_seed_count=len(scheduled_paired_seeds),
            validated_sources=validated_sources,
            seed_file=seed_file,
            seed_file_sha256=seed_file_sha256,
            opponent_specs=resolved_opponent_specs,
            eval_device_override=worker_devices[0],
            batched_inference_override=batched_inference_enabled,
        )

    spec_order = {spec.display_name: index for index, spec in enumerate(resolved_opponent_specs)}
    matchup_results.sort(key=lambda item: spec_order.get(str(item["display_name"]), 10**9))
    anchor_weight_config = _periodic_dev_eval_anchor_weight_map(stack)
    total_eval_wall_clock_seconds = max(0.0, time.perf_counter() - eval_started)
    summary_payload = _build_periodic_dev_eval_checkpoint_summary_impl(
        focal_policy_id=focal_policy_id,
        update_count=update_count,
        policy_version=policy_version,
        matchup_results=matchup_results,
        anchor_weight_config=anchor_weight_config,
        effective_parallel_workers=effective_parallel_workers,
        worker_devices=worker_devices,
        seed_block_job_count=len(seed_block_jobs),
        batched_inference_enabled=batched_inference_enabled,
        total_eval_wall_clock_seconds=total_eval_wall_clock_seconds,
    )
    _write_json(update_dir / "summary.json", summary_payload)
    return summary_payload


def _run_async_periodic_dev_eval_worker(
    request: AsyncPeriodicDevEvalRequest,
    process_pool_executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    contract = load_verified_simulator_contract(request.stack.root, expected_spec_hash=request.spec_hash256)
    effective_eval_device = request.eval_device_override
    effective_worker_devices = (
        request.parallel_worker_devices
        if request.parallel_worker_devices
        else _resolved_periodic_dev_eval_worker_devices(
            stack=request.stack,
            parallel_workers=max(1, int(request.parallel_workers)),
            explicit_worker_devices=(),
            eval_device=str(effective_eval_device or _evaluation_config_or_raise(request.stack).eval_device),
            learner_device=None,
        )
    )
    return _run_periodic_dev_eval_for_checkpoint(
        stack=request.stack,
        contract=contract,
        run_dir=request.run_dir,
        checkpoint_path=request.checkpoint_path,
        focal_policy_id=request.focal_policy_id,
        update_count=request.update_count,
        policy_version=request.policy_version,
        run_id256=request.run_id256,
        config_hash256=request.config_hash256,
        spec_hash256=request.spec_hash256,
        opponent_specs=request.opponents,
        eval_device_override=effective_eval_device,
        parallel_workers_override=int(request.parallel_workers),
        parallel_worker_devices_override=tuple(effective_worker_devices),
        artifact_dir_name=request.artifact_dir_name,
        artifact_scope=request.artifact_scope,
        paired_seeds_override=request.paired_seeds,
        process_pool_executor=process_pool_executor,
    )


def _run_periodic_dev_eval(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    device: torch.device,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    artifact_dir_name: str = "dev_eval",
    artifact_scope: str = "periodic_dev_eval",
    paired_seeds_override: Sequence[int] | None = None,
    persist_summary: bool = True,
    update_stall_monitor: bool = True,
    batched_inference_override: bool | None = None,
    process_pool_executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    if learner.model is None:
        raise RuntimeError("Periodic dev eval requires an attached learner model")

    checkpoint_path = _ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=str(stack.config.training.algorithm).strip() if stack.config.training is not None else None,
    )
    summary_payload = _run_periodic_dev_eval_for_checkpoint(
        stack=stack,
        contract=contract,
        run_dir=artifacts.run_dir,
        checkpoint_path=checkpoint_path,
        focal_policy_id=_current_focal_policy_id(learner=learner),
        update_count=int(learner.update_count),
        policy_version=int(learner.get_policy_version()),
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        artifact_dir_name=artifact_dir_name,
        artifact_scope=artifact_scope,
        paired_seeds_override=paired_seeds_override,
        batched_inference_override=batched_inference_override,
        process_pool_executor=process_pool_executor,
    )
    if update_stall_monitor:
        stall_monitor = _apply_stall_monitor_to_dev_eval_summary_impl(
            stack=stack,
            training_paths=training_paths,
            summary_payload=summary_payload,
            summary_path=(
                artifacts.run_dir
                / "eval"
                / artifact_dir_name
                / f"update_{int(summary_payload['update_count'])}"
                / "summary.json"
            ),
        )
        if stall_monitor is not None:
            if bool(stall_monitor.get("stall_risk", False)):
                print(
                    _format_stall_monitor_warning_impl(stall_monitor, update_count=int(summary_payload["update_count"]))
                )
    if persist_summary:
        _persist_periodic_dev_eval_result(training_paths=training_paths, payload=summary_payload)
    return summary_payload


def _process_completed_periodic_dev_eval(
    *,
    pending_eval: PendingPeriodicDevEval,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: RunArtifacts,
    training_paths: TrainingPaths,
    runtime: QueueRuntime,
    learner: ImpalaLearner,
    device: torch.device,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    last_rollback_update: int | None,
    tensorboard_logger: TensorBoardLogger | None,
    process_pool_executor: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        summary_payload = pending_eval.future.result()
        stall_monitor = _apply_stall_monitor_to_dev_eval_summary_impl(
            stack=stack,
            training_paths=training_paths,
            summary_payload=summary_payload,
            summary_path=(
                artifacts.run_dir
                / "eval"
                / "dev_eval"
                / f"update_{int(summary_payload['update_count'])}"
                / "summary.json"
            ),
        )
        if stall_monitor is not None:
            if bool(stall_monitor.get("stall_risk", False)):
                print(
                    _format_stall_monitor_warning_impl(stall_monitor, update_count=int(summary_payload["update_count"]))
                )
        effective_summary = summary_payload
        tracker_before_dev_eval = _load_checkpoint_tracker(training_paths)
        existing_best_record = tracker_before_dev_eval.get("best")
        if not isinstance(existing_best_record, Mapping):
            existing_best_record = None
        confirmatory_plan = _build_confirmatory_dev_eval_plan_impl(
            stack=stack,
            existing_best_record=cast(Mapping[str, Any] | None, existing_best_record),
            dev_eval_summary=effective_summary,
        )
        if confirmatory_plan is not None:
            effective_summary = _run_periodic_dev_eval_for_checkpoint(
                stack=stack,
                contract=contract,
                run_dir=artifacts.run_dir,
                checkpoint_path=pending_eval.request.checkpoint_path,
                focal_policy_id=str(summary_payload["policy_id"]),
                update_count=int(summary_payload["update_count"]),
                policy_version=int(summary_payload["policy_version"]),
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                opponent_specs=pending_eval.request.opponents,
                eval_device_override=pending_eval.request.eval_device_override,
                artifact_dir_name="dev_eval_confirmatory",
                artifact_scope="periodic_dev_eval_confirmatory",
                paired_seeds_override=confirmatory_plan.paired_seeds,
                batched_inference_override=False,
                process_pool_executor=process_pool_executor,
            )
            print(
                _format_confirmatory_dev_eval_message_impl(
                    update_count=int(summary_payload["update_count"]),
                    paired_seed_count=len(confirmatory_plan.paired_seeds),
                    aggregate_score=float(effective_summary["aggregate_score"]),
                    reasons=confirmatory_plan.reasons,
                    seed_file=confirmatory_plan.seed_file,
                )
            )
            _persist_periodic_dev_eval_result(
                training_paths=training_paths,
                payload=effective_summary,
                force_summary=True,
            )
        else:
            _persist_periodic_dev_eval_result(training_paths=training_paths, payload=effective_summary)

        tracker_payload = _publish_best_checkpoint_from_dev_eval(
            stack=stack,
            training_paths=training_paths,
            artifacts=artifacts,
            checkpoint_path=pending_eval.request.checkpoint_path,
            update_count=int(effective_summary["update_count"]),
            policy_version=int(effective_summary["policy_version"]),
            dev_eval_summary=effective_summary,
        )
        _maybe_log_structured_mainmove_guard(
            training_paths=training_paths,
            learner=learner,
            latest_metrics=pending_eval.latest_metrics,
            dev_eval_summary=effective_summary,
        )
        guard_event = None
        if int(effective_summary["update_count"]) == int(learner.update_count):
            guard_event = _maybe_rollback_to_best_checkpoint(
                stack=stack,
                training_paths=training_paths,
                artifacts=artifacts,
                runtime=runtime,
                learner=learner,
                model=cast(PolicyValueModel, learner.model),
                device=device,
                spec_hash256=spec_hash256,
                algorithm=str(stack.config.training.algorithm).strip() if stack.config.training is not None else None,
                latest_metrics=pending_eval.latest_metrics,
                dev_eval_summary=effective_summary,
                last_rollback_update=last_rollback_update,
            )
        if tensorboard_logger is not None:
            tensorboard_logger.log_periodic_dev_eval(effective_summary, step=int(effective_summary["update_count"]))
            tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(effective_summary["update_count"]))
        audit_request = _maybe_request_b2_disagreement_audit(
            stack=stack,
            training_paths=training_paths,
            artifacts=artifacts,
            dev_eval_summary=effective_summary,
        )
        if audit_request is not None:
            print(_format_b2_disagreement_audit_request_message_impl(audit_request))
        return effective_summary, guard_event
    finally:
        _unpin_snapshot_ids(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            snapshot_ids=pending_eval.pinned_snapshot_ids,
        )


def _run_async_promotion_gate_worker(request: AsyncPromotionGateRequest) -> dict[str, Any]:
    contract = load_verified_simulator_contract(request.stack.root, expected_spec_hash=request.spec_hash256)
    evaluation = _validate_periodic_dev_eval_contract(request.stack)
    observation_dim, action_dim = _spec_dimensions(contract)
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    focal_model = _load_snapshot_eval_model(
        run_dir=request.run_dir,
        snapshot_path=request.candidate_snapshot_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=request.stack,
        eval_device=(evaluation.eval_device if request.eval_device_override is None else request.eval_device_override),
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    )
    opponents = _materialize_periodic_dev_eval_opponents(
        stack=request.stack,
        contract=contract,
        run_dir=request.run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        opponent_specs=request.anchor_specs,
        eval_device_override=request.eval_device_override,
    )
    anchor_models, heuristic_policies = _promotion_gate_policy_maps_impl(opponents)
    runner = _PromotionGateRunner(
        stack=request.stack,
        focal_policy_id=request.candidate_policy_id,
        focal_model=focal_model,
        anchor_models=anchor_models,
        heuristic_policies=heuristic_policies,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        artifact_dir=request.run_dir / "eval" / "promotion_gate" / f"update_{request.update_count}",
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        eval_device=(evaluation.eval_device if request.eval_device_override is None else request.eval_device_override),
    )
    try:
        result = run_promotion_gate(
            stack=request.stack,
            run_dir=request.run_dir / "eval" / "promotion_gate" / f"update_{request.update_count}",
            focal_policy_id=request.candidate_policy_id,
            anchor_policy_ids=request.anchor_policy_ids,
            runner=runner,
            run_id256=request.run_id256,
            config_hash256=request.config_hash256,
            spec_hash256=request.spec_hash256,
            bootstrap_seed=_promotion_gate_bootstrap_seed(
                update_count=request.update_count,
                policy_version=request.policy_version,
            ),
        )
    finally:
        close_runner = getattr(runner, "close", None)
        if callable(close_runner):
            close_runner()
    return {
        "candidate_policy_id": request.candidate_policy_id,
        "update_count": int(request.update_count),
        "policy_version": int(request.policy_version),
        "passed": bool(result.passed),
        "ordered_opponents": list(result.ordered_opponents),
        "reasons": [dict(reason) for reason in result.reasons],
        "result": result.to_dict(),
    }


def _run_parallel_promotion_gate_anchor_worker(
    *,
    stack: StackConfig,
    run_dir: Path,
    candidate_policy_id: str,
    candidate_snapshot_path: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    seed_block_jobs: Sequence[PromotionGateSeedBlockJob],
    eval_device_override: str,
) -> list[dict[str, Any]]:
    if not seed_block_jobs:
        return []
    contract = load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash256)
    evaluation = _validate_periodic_dev_eval_contract(stack)
    observation_dim, action_dim = _spec_dimensions(contract)
    observation_spec = cast(dict[str, Any] | None, contract.spec_bundle.get("observation"))
    spec_bundle = cast(dict[str, Any] | None, contract.spec_bundle)
    focal_model = _load_snapshot_eval_model(
        run_dir=run_dir,
        snapshot_path=candidate_snapshot_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        eval_device=eval_device_override,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    )
    ordered_specs = list({(job.anchor_index, job.anchor_spec): job.anchor_spec for job in seed_block_jobs}.values())
    opponents = _materialize_periodic_dev_eval_opponents(
        stack=stack,
        contract=contract,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        opponent_specs=ordered_specs,
        eval_device_override=eval_device_override,
    )
    anchor_models, heuristic_policies = _promotion_gate_policy_maps_impl(opponents)
    seed_file = resolve_promotion_gate_seed_file(stack)
    paired_seeds = parse_seed_file(seed_file)
    league = stack.config.league
    if league is None:
        raise RuntimeError("Parallel promotion gate requires stack.config.league")
    if len(paired_seeds) != int(league.promotion_gate_paired_seeds):
        raise RuntimeError(
            f"Promotion gate expected {int(league.promotion_gate_paired_seeds)} paired seeds in {seed_file}, "
            f"found {len(paired_seeds)}"
        )
    runner = _PromotionGateRunner(
        stack=stack,
        focal_policy_id=candidate_policy_id,
        focal_model=focal_model,
        anchor_models=anchor_models,
        heuristic_policies=heuristic_policies,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        artifact_dir=run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        eval_device=eval_device_override,
    )
    try:
        return _build_promotion_gate_worker_payloads_impl(
            seed_block_jobs=seed_block_jobs,
            runner=runner,
            candidate_policy_id=candidate_policy_id,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
        )
    finally:
        close_runner = getattr(runner, "close", None)
        if callable(close_runner):
            close_runner()


def _run_parallel_snapshot_promotion_gate(
    *,
    stack: StackConfig,
    artifacts: RunArtifacts,
    training_paths: TrainingPaths,
    candidate_policy_id: str,
    update_count: int,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    anchor_policy_ids: Mapping[str, str],
    anchor_specs: Sequence[PeriodicDevEvalOpponentSpec],
    candidate_snapshot_path: str,
) -> PromotionGateResult:
    evaluation = _validate_periodic_dev_eval_contract(stack)
    gate_plan = _build_parallel_promotion_gate_plan_impl(
        stack=stack,
        anchor_policy_ids=anchor_policy_ids,
        anchor_specs=anchor_specs,
        eval_device=str(evaluation.eval_device),
    )
    worker_payloads: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gate_plan.job_shards), mp_context=ctx) as executor:
        futures = [
            executor.submit(
                _run_parallel_promotion_gate_anchor_worker,
                stack=stack,
                run_dir=artifacts.run_dir,
                candidate_policy_id=candidate_policy_id,
                candidate_snapshot_path=candidate_snapshot_path,
                update_count=update_count,
                policy_version=policy_version,
                run_id256=run_id256,
                config_hash256=config_hash256,
                spec_hash256=spec_hash256,
                seed_block_jobs=tuple(shard),
                eval_device_override=gate_plan.worker_devices[shard_index],
            )
            for shard_index, shard in enumerate(gate_plan.job_shards)
        ]
        for future in futures:
            worker_payloads.extend(future.result())
    records_by_anchor_index = _promotion_gate_records_by_anchor_index_impl(
        worker_payloads=worker_payloads,
        anchor_count=len(anchor_specs),
    )

    return _assemble_parallel_promotion_gate_result_impl(
        stack=stack,
        artifacts=artifacts,
        focal_policy_id=candidate_policy_id,
        update_count=update_count,
        policy_version=policy_version,
        anchors=gate_plan.ordered_anchors,
        records_by_anchor_index=records_by_anchor_index,
        paired_seeds=gate_plan.paired_seeds,
    )


def _process_completed_promotion_gate(
    *,
    pending_gate: PendingPromotionGate,
    stack: StackConfig,
    artifacts: RunArtifacts,
    training_paths: TrainingPaths,
) -> bool:
    try:
        payload = pending_gate.future.result()
        registry_update = _apply_promotion_gate_payload(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            payload=payload,
        )
        if registry_update.passed:
            print(_format_promotion_gate_registry_update_message_impl(registry_update))
            return True

        print(_format_promotion_gate_registry_update_message_impl(registry_update))
        return False
    finally:
        _unpin_snapshot_ids(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            snapshot_ids=pending_gate.pinned_snapshot_ids,
        )


def _drop_stale_pending_promotion_gate(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    pending_gate: PendingPromotionGate | None,
    rollback_best_update_count: int,
) -> PendingPromotionGate | None:
    if pending_gate is None:
        return None
    if int(pending_gate.request.update_count) <= int(rollback_best_update_count):
        return pending_gate
    print(
        _format_promotion_gate_discarded_after_rollback_message_impl(
            candidate_policy_id=pending_gate.request.candidate_policy_id,
            candidate_update=int(pending_gate.request.update_count),
            rollback_best_update=int(rollback_best_update_count),
        )
    )
    _unpin_snapshot_ids(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        snapshot_ids=pending_gate.pinned_snapshot_ids,
    )
    return None


def _run_snapshot_promotion_gate(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    candidate_policy_id: str,
    update_count: int,
    league_reference_update: int | None,
    league_eval_warmup_gate_open: bool,
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
) -> bool | None:
    league = stack.config.league
    if league is None or not league.enabled or not league.promotion_gate_enabled:
        return None
    reference_update = int(update_count if league_reference_update is None else league_reference_update)
    if reference_update < int(league.warmup.first_updates):
        print(
            _format_promotion_gate_skipped_league_warmup_message_impl(
                update_count=update_count,
                effective_update=reference_update,
                threshold=int(league.warmup.first_updates),
                candidate_policy_id=candidate_policy_id,
            )
        )
        return None
    if bool(getattr(league.warmup, "eval_gate_enabled", False)) and not bool(league_eval_warmup_gate_open):
        print(
            _format_promotion_gate_skipped_eval_warmup_gate_message_impl(
                update_count=update_count,
                effective_update=reference_update,
                candidate_policy_id=candidate_policy_id,
            )
        )
        return None
    if learner.model is None:
        raise RuntimeError("Promotion gate requires an attached learner model")

    evaluation = _validate_periodic_dev_eval_contract(stack)
    registry_path = training_paths.snapshots_dir / REGISTRY_FILENAME
    registry = SnapshotRegistry.load(registry_path)
    anchor_policy_ids, missing_required = _resolve_promotion_anchor_policy_ids(
        stack=stack,
        registry=registry,
    )
    if missing_required:
        print(
            _format_promotion_gate_missing_anchors_message_impl(
                update_count=update_count,
                candidate_policy_id=candidate_policy_id,
                missing_anchors=missing_required,
            )
        )
        return None

    observation_dim, action_dim = _spec_dimensions(contract)
    snapshot_index = _snapshot_meta_by_policy_id(registry)
    configured_parallel_workers = max(1, int(getattr(league.promotion_gate, "parallel_workers", 1)))
    if configured_parallel_workers > 1:
        candidate_snapshot = snapshot_index.get(candidate_policy_id)
        if candidate_snapshot is None:
            raise RuntimeError(f"Promotion gate could not resolve candidate snapshot {candidate_policy_id!r}")
        anchor_policy_ids_for_parallel, anchor_specs, _pinned_anchor_snapshot_ids = (
            _resolve_promotion_gate_anchor_specs(
                stack=stack,
                training_paths=training_paths,
            )
        )
        result = _run_parallel_snapshot_promotion_gate(
            stack=stack,
            artifacts=artifacts,
            training_paths=training_paths,
            candidate_policy_id=candidate_policy_id,
            update_count=update_count,
            policy_version=policy_version,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            anchor_policy_ids=anchor_policy_ids_for_parallel,
            anchor_specs=anchor_specs,
            candidate_snapshot_path=candidate_snapshot.path,
        )
        registry_update = _apply_promotion_gate_result_impl(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            registry=registry,
            candidate_policy_id=candidate_policy_id,
            update_count=update_count,
            result=result,
        )
        if registry_update.passed:
            print(_format_promotion_gate_registry_update_message_impl(registry_update))
            return True

        print(_format_promotion_gate_registry_update_message_impl(registry_update))
        return False
    anchor_models = {
        policy_id: _load_snapshot_eval_model(
            run_dir=artifacts.run_dir,
            snapshot_path=snapshot_index[policy_id].path,
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            eval_device=evaluation.eval_device,
            observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
            spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
        )
        for policy_id in set(anchor_policy_ids.values())
        if policy_id != _PROMOTION_GATE_RANDOMLEGAL_POLICY_ID
        and heuristic_public_profile_name_for_policy_id(policy_id) is None
    }
    heuristic_policies: dict[str, HeuristicPublicPolicy] = {}
    heuristic_policy_ids = {
        policy_id
        for policy_id in set(anchor_policy_ids.values())
        if heuristic_public_profile_name_for_policy_id(policy_id) is not None
    }
    if heuristic_policy_ids:
        try:
            heuristic_policies = {
                policy_id: _build_heuristic_public_policy(
                    contract.spec_bundle,
                    scoring_profile=cast(str, heuristic_public_profile_name_for_policy_id(policy_id)),
                )
                for policy_id in heuristic_policy_ids
            }
        except Exception as exc:
            assert league is not None
            missing_required = [
                policy_id for policy_id in heuristic_policy_ids if policy_id in league.promotion_anchor_set_v1.required
            ]
            if missing_required:
                missing_text = ", ".join(missing_required)
                raise RuntimeError(
                    f"Promotion gate requires a heuristic-compatible simulator contract for {missing_text}"
                ) from exc
            anchor_policy_ids = {
                anchor_name: policy_id
                for anchor_name, policy_id in anchor_policy_ids.items()
                if heuristic_public_profile_name_for_policy_id(policy_id) is None
            }
            print(_format_optional_heuristic_public_anchors_skipped_message_impl(exc))
    runner = _PromotionGateRunner(
        stack=stack,
        focal_policy_id=candidate_policy_id,
        focal_model=_clone_eval_model(
            learner_model=cast(PolicyValueModel, learner.model),
            observation_dim=observation_dim,
            action_dim=action_dim,
            stack=stack,
            eval_device=evaluation.eval_device,
            observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
            spec_bundle=cast(dict[str, Any] | None, contract.spec_bundle),
        ),
        anchor_models=anchor_models,
        heuristic_policies=heuristic_policies,
        observation_dim=observation_dim,
        action_dim=action_dim,
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        artifact_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        eval_device=evaluation.eval_device,
    )
    try:
        result = run_promotion_gate(
            stack=stack,
            run_dir=artifacts.run_dir / "eval" / "promotion_gate" / f"update_{update_count}",
            focal_policy_id=candidate_policy_id,
            anchor_policy_ids=anchor_policy_ids,
            runner=runner,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            bootstrap_seed=_promotion_gate_bootstrap_seed(
                update_count=update_count,
                policy_version=policy_version,
            ),
        )
    finally:
        close_runner = getattr(runner, "close", None)
        if callable(close_runner):
            close_runner()
    registry_update = _apply_promotion_gate_result_impl(
        stack=stack,
        training_paths=training_paths,
        run_dir=artifacts.run_dir,
        registry=registry,
        candidate_policy_id=candidate_policy_id,
        update_count=update_count,
        result=result,
    )
    if registry_update.passed:
        print(_format_promotion_gate_registry_update_message_impl(registry_update))
        return True

    print(_format_promotion_gate_registry_update_message_impl(registry_update))
    return False


def _apply_freeze_parameter_prefixes(model: nn.Module, prefixes: Sequence[str]) -> dict[str, int]:
    normalized = tuple(str(prefix).strip() for prefix in prefixes if str(prefix).strip())
    if not normalized:
        return {"frozen": 0, "trainable": sum(1 for _ in model.parameters())}

    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        should_freeze = any(name == prefix or name.startswith(prefix + ".") for prefix in normalized)
        param.requires_grad_(not should_freeze)
        if should_freeze:
            frozen += 1
        else:
            trainable += 1

    if trainable <= 0:
        raise RuntimeError(
            "freeze_parameter_prefixes froze every parameter; at least one trainable parameter is required"
        )

    print(
        "Applied parameter freeze: "
        f"prefixes={list(normalized)} frozen_tensors={frozen} trainable_tensors={trainable}"
    )
    return {"frozen": frozen, "trainable": trainable}


def _run_minimal_training(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    num_envs: int,
    unroll_length: int,
    max_updates: int,
    max_wall_clock_minutes: float | None,
    profile: str,
    device: torch.device,
    seed: int,
    checkpoint_interval_updates: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    runtime_mode: QueueRuntimeMode,
    b1_baseline_run_dir: Path | None,
    seed_snapshot_run_dir: Path | None = None,
    seed_snapshot_run_dir_auto_inferred: bool = False,
    profile_timers: bool = False,
    torch_profiler: bool = False,
    resume_checkpoint_path: Path | None = None,
    resume_allow_config_mismatch: bool = False,
    resume_reset_optimizer: bool = False,
    tensorboard_logger: TensorBoardLogger | None = None,
    resolved_topology: ResolvedTrainingTopology | None = None,
    distributed_context: DistributedContext | None = None,
) -> dict[str, float]:
    _configure_torch_threads(stack)
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)

    observation_dim, action_dim = _spec_dimensions(contract)
    training_config = stack.config.training
    model_config = stack.config.model
    environment_config = stack.config.environment
    rewards_config = stack.config.rewards
    experiment_role = _experiment_role(stack)
    if training_config is None or model_config is None or environment_config is None or rewards_config is None:
        raise RuntimeError("The locked stack is missing training, model, environment, or rewards config")
    main_residual_policy_enabled = bool(
        getattr(getattr(training_config, "main_residual_policy", None), "enabled", False)
    )

    training_paths = _training_paths(artifacts.run_dir)
    ddp_context = DistributedContext(enabled=False) if distributed_context is None else distributed_context
    rank0 = (not ddp_context.enabled) or ddp_context.is_rank0
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    algorithm = str(training_config.algorithm).strip()
    _validate_algorithm_model_contract(
        algorithm=algorithm,
        recurrent_core=model_config.recurrent_core,
        encoder_kind=model_config.encoder_kind,
    )
    model = _maybe_build_main_residual_model(
        stack=stack,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
        device=device,
    )
    if model is None:
        model = build_policy_value_model(
            observation_dim=observation_dim,
            config=model_config,
            action_dim=action_dim,
            observation_spec=contract.spec_bundle.get("observation"),
            spec_bundle=contract.spec_bundle,
        ).to(device)
    compiled_model = _maybe_compile_learner_model(
        model=model,
        training_config=training_config,
        device=device,
    )
    learner = _build_training_learner(
        algorithm=algorithm,
        model=model,
        compiled_model=compiled_model,
        training_config=training_config,
        training_paths=training_paths,
        pass_action_id=pass_action_id,
        checkpoint_interval_updates=checkpoint_interval_updates,
        gradient_sync=(None if not ddp_context.enabled else lambda: average_gradients(model, context=ddp_context)),
        artifact_writes_enabled=rank0,
    )
    resume_state = None
    resume_dev_eval_summary: dict[str, Any] | None = None
    if resume_checkpoint_path is not None:
        resume_state = _restore_learner_from_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
            restore_optimizer_state=not bool(resume_reset_optimizer),
            allow_config_hash_mismatch=resume_allow_config_mismatch,
        )
        print(
            _format_resumed_learner_state_message_impl(
                checkpoint_path=resume_state.checkpoint_path,
                update_count=int(resume_state.update_count),
                policy_version=int(resume_state.policy_version),
            )
        )
        if rank0:
            seeded_best_record = _seed_checkpoint_tracker_from_resume_best(
                stack=stack,
                training_paths=training_paths,
                artifacts=artifacts,
                resume_checkpoint_path=resume_state.checkpoint_path,
            )
            if seeded_best_record is not None:
                print(_format_seeded_checkpoint_best_alias_message_impl(seeded_best_record))
        resume_dev_eval_summary = _load_resume_checkpoint_dev_eval_summary(
            stack=stack,
            resume_checkpoint_path=resume_state.checkpoint_path,
            update_count=int(resume_state.update_count),
            allow_config_hash_mismatch=resume_allow_config_mismatch,
        )
        if rank0 and resume_dev_eval_summary is not None:
            print(
                _format_seeded_resume_dev_eval_summary_message_impl(
                    update_count=int(resume_state.update_count),
                    aggregate_score=float(_dev_eval_aggregate_score(resume_dev_eval_summary) or 0.0),
                )
            )

    freeze_prefixes = tuple(getattr(training_config, "freeze_parameter_prefixes", ()) or ())
    _apply_freeze_parameter_prefixes(model, freeze_prefixes)

    config_hash256 = compute_config_hash256(stack)
    if rank0:
        _ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            learner=learner,
            device=device,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            baseline_run_dir=b1_baseline_run_dir,
            permit_current_run_alias=_is_noleague_baseline_role(experiment_role),
            update=int(learner.update_count),
        )
    distributed_barrier(ddp_context)
    _attach_reference_policy_model_if_configured(
        learner=learner,
        training_config=training_config,
        training_paths=training_paths,
        model_config=model_config,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any], contract.spec_bundle),
        device=device,
    )
    imported_resume_league_policy_ids: tuple[str, ...] = ()
    if rank0 and resume_state is not None:
        imported_resume_league_policy_ids = tuple(
            _import_resume_league_snapshot_pool(
                stack=stack,
                training_paths=training_paths,
                run_dir=artifacts.run_dir,
                resume_checkpoint_path=resume_state.checkpoint_path,
                max_update=int(resume_state.update_count),
                expected_model_state_dict=learner.model.state_dict(),
            )
        )
    seed_snapshot_max_update = _seed_snapshot_import_max_update(
        resume_state=resume_state,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        seed_snapshot_run_dir_auto_inferred=seed_snapshot_run_dir_auto_inferred,
    )
    if rank0 and seed_snapshot_run_dir is not None:
        _import_seed_snapshot_pool(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            seed_snapshot_run_dir=seed_snapshot_run_dir,
            max_update=seed_snapshot_max_update,
            exclude_source_policy_ids=imported_resume_league_policy_ids,
            expected_model_state_dict=learner.model.state_dict(),
            expected_config_canonical=canonical_config_dict(stack),
            expected_spec_hash256=spec_hash256,
        )
    if seed_snapshot_run_dir is not None or resume_state is not None:
        distributed_barrier(ddp_context)
    runtime_config = build_runtime_config(
        stack=stack,
        num_envs=shard_env_count(global_num_envs=num_envs, world_size=ddp_context.world_size, rank=ddp_context.rank)
        if ddp_context.enabled
        else num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=rank_seed(seed, rank=ddp_context.rank) if ddp_context.enabled else seed,
        pass_action_id=pass_action_id,
        runtime_mode=runtime_mode,
        resolved_actor_count=None
        if resolved_topology is None
        else max(1, int(resolved_topology.actor_count) // max(1, int(ddp_context.world_size))),
        resolved_envs_per_actor=None if resolved_topology is None else int(resolved_topology.envs_per_actor),
        resolved_batch_unrolls_per_update=(
            None
            if resolved_topology is None
            else max(1, int(resolved_topology.batch_unrolls_per_update) // max(1, int(ddp_context.world_size)))
        ),
        resolved_queue_capacity_unrolls=(
            None
            if resolved_topology is None
            else max(1, int(resolved_topology.queue_capacity_unrolls) // max(1, int(ddp_context.world_size)))
        ),
    )
    runtime = QueueRuntime(
        stack=stack,
        config=runtime_config,
        model=model,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=cast(dict[str, Any] | None, contract.spec_bundle.get("observation")),
        spec_bundle=cast(dict[str, Any], contract.spec_bundle),
        run_dir=artifacts.run_dir,
        performance_log_path=training_paths.performance_log_path,
        learner_device=device,
        rank_local_actor_devices=bool(ddp_context.enabled),
        initial_learner_update=int(learner.update_count),
    )
    if int(learner.update_count) > 0:
        runtime.maybe_publish_snapshot(
            learner_model=model,
            learner_update_count=int(learner.update_count),
            force=True,
        )
    actor_torch_threads = _central_runtime_actor_torch_threads(stack, runtime)
    learner_torch_threads = None if stack.config.system is None else int(stack.config.system.learner_torch_threads)
    latest_metrics: dict[str, float] = {}
    last_checkpoint_guard_rollback_update: int | None = None
    last_dev_eval_summary: Mapping[str, Any] | None = resume_dev_eval_summary
    last_dev_eval_update_count: int | None = (
        int(resume_state.update_count) if resume_state is not None and resume_dev_eval_summary is not None else None
    )
    league_eval_warmup_gate_status = _sync_runtime_league_eval_warmup_gate(
        runtime=runtime,
        stack=stack,
        dev_eval_summary=last_dev_eval_summary,
    )
    league_eval_warmup_gate_open = bool(league_eval_warmup_gate_status["open"])
    collect_batch_prefetch_enabled = bool(getattr(training_config, "collect_batch_prefetch_enabled", False))
    start_time = time.time()
    max_wall_clock_seconds = _wall_clock_budget_seconds(max_wall_clock_minutes)
    profiler, profiler_context, profiler_trace_dir = _build_training_profiler(
        enabled=bool(torch_profiler),
        run_dir=artifacts.run_dir,
        device=device,
    )
    prefetch_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="collect-batch-prefetch")
        if collect_batch_prefetch_enabled
        else None
    )
    async_periodic_dev_eval_enabled = bool(
        rank0
        and stack.config.evaluation is not None
        and getattr(stack.config.evaluation, "async_periodic_dev_eval_enabled", False)
    )
    async_periodic_dev_eval_executor: ThreadPoolExecutor | None = None
    if async_periodic_dev_eval_enabled:
        async_periodic_dev_eval_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="async-periodic-dev-eval",
        )
    async_promotion_gate_enabled = bool(
        rank0
        and stack.config.league is not None
        and stack.config.league.promotion_gate_enabled
        and bool(getattr(stack.config.league.promotion_gate, "async_enabled", False))
    )
    async_promotion_gate_executor: ProcessPoolExecutor | None = None
    if async_promotion_gate_enabled:
        async_promotion_gate_executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    pending_promotion_gate: PendingPromotionGate | None = None
    pending_periodic_dev_eval: PendingPeriodicDevEval | None = None
    periodic_dev_eval_process_pool: ProcessPoolExecutor | None = None
    prefetched_runtime_batch: Any | None = None
    early_cutoff_payload: Mapping[str, Any] | None = None
    with profiler_context:
        if int(learner.update_count) == 0:
            latest_metrics = _run_structured_warmstart(
                learner=learner,
                runtime=runtime,
                algorithm=algorithm,
                training_config=training_config,
                rewards_config=rewards_config,
                training_paths=training_paths,
                tensorboard_logger=tensorboard_logger,
                start_time=start_time,
                profile_timers=bool(profile_timers),
                actor_torch_threads=actor_torch_threads,
                learner_torch_threads=learner_torch_threads,
            )
        if int(learner.update_count) >= max_updates:
            raise RuntimeError(
                f"Resume checkpoint is already at update {learner.update_count}, which is >= --max-updates {max_updates}"
            )
        try:
            for _update_index in range(int(learner.update_count), max_updates):
                stop_requested = False
                refresh_opponent_pool_requested = False
                guard_event_for_distributed_sync: Mapping[str, Any] | None = None
                if pending_promotion_gate is not None and pending_promotion_gate.future.done():
                    promotion_passed = _process_completed_promotion_gate(
                        pending_gate=pending_promotion_gate,
                        stack=stack,
                        artifacts=artifacts,
                        training_paths=training_paths,
                    )
                    pending_promotion_gate = None
                    if promotion_passed:
                        refresh_opponent_pool_requested = True
                if pending_periodic_dev_eval is not None and pending_periodic_dev_eval.future.done():
                    completed_summary, guard_event = _process_completed_periodic_dev_eval(
                        pending_eval=pending_periodic_dev_eval,
                        stack=stack,
                        contract=contract,
                        artifacts=artifacts,
                        training_paths=training_paths,
                        runtime=runtime,
                        learner=learner,
                        device=device,
                        run_id256=run_id256,
                        config_hash256=config_hash256,
                        spec_hash256=spec_hash256,
                        last_rollback_update=last_checkpoint_guard_rollback_update,
                        tensorboard_logger=tensorboard_logger,
                        process_pool_executor=periodic_dev_eval_process_pool,
                    )
                    pending_periodic_dev_eval = None
                    last_dev_eval_summary = completed_summary
                    last_dev_eval_update_count = int(completed_summary["update_count"])
                    league_eval_warmup_gate_status = _sync_runtime_league_eval_warmup_gate(
                        runtime=runtime,
                        stack=stack,
                        dev_eval_summary=completed_summary,
                    )
                    league_eval_warmup_gate_open = bool(league_eval_warmup_gate_status["open"])
                    anchor_keys = sorted(cast(dict[str, Any], completed_summary["anchor_scores"]).keys())
                    print(
                        _format_periodic_dev_eval_console_message_impl(
                            label="Periodic dev eval complete",
                            update_count=int(completed_summary["update_count"]),
                            aggregate_score=float(completed_summary["aggregate_score"]),
                            anchor_names=anchor_keys,
                            opponent_slug=_slug_policy_id(anchor_keys[0]) if anchor_keys else None,
                        )
                    )
                    if bool(league_eval_warmup_gate_status.get("enabled", False)):
                        print(_format_league_eval_warmup_gate_message_impl(league_eval_warmup_gate_status))
                    if guard_event is not None:
                        guard_event_for_distributed_sync = guard_event
                        refresh_opponent_pool_requested = True
                        last_checkpoint_guard_rollback_update = int(learner.update_count)
                        pending_promotion_gate = _drop_stale_pending_promotion_gate(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            pending_gate=pending_promotion_gate,
                            rollback_best_update_count=int(guard_event["best_update_count"]),
                        )
                        prefetched_runtime_batch = None
                        print(_format_checkpoint_guard_rollback_message_impl(guard_event))
                    early_cutoff_payload = _update_early_cutoff(
                        stack=stack,
                        training_paths=training_paths,
                        update_count=int(completed_summary["update_count"]),
                        summary_payload=completed_summary,
                    )
                    if early_cutoff_payload is not None and bool(early_cutoff_payload.get("should_stop", False)):
                        latest_metrics.update(_early_cutoff_metric_updates_impl(early_cutoff_payload))
                        print(
                            _format_early_cutoff_triggered_message_impl(
                                early_cutoff_payload,
                                update_count=int(completed_summary["update_count"]),
                            )
                        )
                        stop_requested = True
                if _wall_clock_budget_reached(start_time=start_time, max_wall_clock_seconds=max_wall_clock_seconds):
                    elapsed_seconds = time.time() - start_time
                    latest_metrics.update(
                        _wall_clock_budget_metric_updates_impl(
                            max_wall_clock_seconds=float(max_wall_clock_seconds),
                            elapsed_seconds=elapsed_seconds,
                        )
                    )
                    print(
                        _format_wall_clock_budget_reached_message_impl(
                            elapsed_seconds=elapsed_seconds,
                            max_wall_clock_seconds=float(max_wall_clock_seconds),
                        )
                    )
                    stop_requested = True
                if ddp_context.enabled:
                    distributed_guard_event = broadcast_object(
                        guard_event_for_distributed_sync if rank0 else None,
                        context=ddp_context,
                    )
                    if distributed_guard_event is not None:
                        distributed_barrier(ddp_context)
                        if not rank0:
                            _restore_checkpoint_to_latest_alias(
                                checkpoint_path=training_paths.latest_checkpoint_path,
                                training_paths=training_paths,
                                learner=learner,
                                stack=stack,
                                device=device,
                                expected_spec_hash256=spec_hash256,
                                algorithm=algorithm,
                                restore_counters=False,
                            )
                            prefetched_runtime_batch = None
                        runtime.maybe_publish_snapshot(
                            learner_model=model,
                            learner_update_count=int(learner.update_count),
                            force=True,
                        )
                        refresh_opponent_pool_requested = True
                        distributed_barrier(ddp_context)
                    refresh_requested = all_reduce_float(
                        1.0 if refresh_opponent_pool_requested else 0.0,
                        context=ddp_context,
                        op="sum",
                    )
                    if refresh_requested > 0.0:
                        distributed_barrier(ddp_context)
                        runtime.refresh_opponent_pool(allow_registry_write=rank0)
                        distributed_barrier(ddp_context)
                    stop_requested = (
                        all_reduce_float(1.0 if stop_requested else 0.0, context=ddp_context, op="sum") > 0.0
                    )
                elif refresh_opponent_pool_requested:
                    runtime.refresh_opponent_pool()
                if stop_requested:
                    break
                guidance_schedule_metrics = _apply_guidance_schedule_for_next_update(
                    learner=learner,
                    model=model,
                    stack=stack,
                    update_count=int(learner.update_count) + 1,
                )
                learner.set_entropy_coef(
                    _entropy_coef_for_next_update(training_config, update_count=int(learner.update_count) + 1)
                )
                upcoming_update_count = int(learner.update_count) + 1
                batch_wait_started = time.perf_counter()
                if prefetched_runtime_batch is not None:
                    runtime_batch = prefetched_runtime_batch
                    prefetched_runtime_batch = None
                else:
                    with (
                        _profile_block(profile_timers, "collect_update_batch"),
                        _torch_num_threads_scope(actor_torch_threads),
                    ):
                        runtime_batch = _collect_training_batch(
                            runtime=runtime,
                            algorithm=algorithm,
                            training_config=training_config,
                            rewards_config=rewards_config,
                        )
                learner_idle_wait_for_batch_ms = (time.perf_counter() - batch_wait_started) * 1000.0
                prefetch_future: Future[Any] | None = None
                should_prefetch_next_batch = (
                    prefetch_executor is not None
                    and upcoming_update_count < max_updates
                    and upcoming_update_count % checkpoint_interval_updates != 0
                    and not _should_run_periodic_dev_eval(stack, update_count=upcoming_update_count)
                )
                if should_prefetch_next_batch:
                    prefetch_future = prefetch_executor.submit(
                        _collect_training_batch_prefetch,
                        runtime=runtime,
                        algorithm=algorithm,
                        training_config=training_config,
                        rewards_config=rewards_config,
                        actor_torch_threads=actor_torch_threads,
                    )
                learner_update_started = time.perf_counter()
                with _profile_block(profile_timers, "learner_update"), _torch_num_threads_scope(learner_torch_threads):
                    latest_metrics = learner.update(runtime_batch.learner_batch)
                learner_update_ms = (time.perf_counter() - learner_update_started) * 1000.0
                learner_idle_wait_for_prefetch_ms = 0.0
                if prefetch_future is not None:
                    prefetch_wait_started = time.perf_counter()
                    with _profile_block(profile_timers, "collect_update_batch_prefetch_join"):
                        prefetched_runtime_batch = prefetch_future.result()
                    learner_idle_wait_for_prefetch_ms = (time.perf_counter() - prefetch_wait_started) * 1000.0
                latest_metrics.update(runtime_batch.runtime_metrics)
                latest_metrics.update(guidance_schedule_metrics)
                latest_metrics.update(
                    {
                        "learner_idle_wait_for_batch_ms": learner_idle_wait_for_batch_ms,
                        "learner_idle_wait_for_prefetch_ms": learner_idle_wait_for_prefetch_ms,
                        "learner_update_ms": learner_update_ms,
                        "distributed_rank": float(ddp_context.rank),
                        "distributed_world_size": float(ddp_context.world_size),
                        "async_periodic_dev_eval_overlap_active": 1.0 if pending_periodic_dev_eval is not None else 0.0,
                        "async_promotion_gate_overlap_active": 1.0 if pending_promotion_gate is not None else 0.0,
                    }
                )
                if ddp_context.enabled:
                    local_batch_env_steps = float(latest_metrics.get("batch_env_steps", 0.0))
                    global_batch_env_steps = all_reduce_float(local_batch_env_steps, context=ddp_context, op="sum")
                    global_total_samples = all_reduce_float(
                        float(getattr(learner, "total_samples_processed", 0)),
                        context=ddp_context,
                        op="sum",
                    )
                    elapsed_for_global = max(time.time() - start_time, 1e-6)
                    latest_metrics.update(
                        {
                            "distributed_global_batch_env_steps": global_batch_env_steps,
                            "distributed_global_total_samples_processed": global_total_samples,
                            "distributed_global_samples_per_sec": global_total_samples / elapsed_for_global,
                            "distributed_local_batch_env_steps": local_batch_env_steps,
                        }
                    )
                with _profile_block(profile_timers, "runtime_snapshot_publish"):
                    latest_metrics.update(
                        runtime.maybe_publish_snapshot(
                            learner_model=model,
                            learner_update_count=int(learner.update_count),
                        )
                    )
                latest_metrics["snapshot_publish_reload_ms"] = float(
                    latest_metrics.get("snapshot_publish_latency_ms", 0.0)
                    + latest_metrics.get("snapshot_apply_latency_ms", 0.0)
                )
                if rank0:
                    _write_scalars_record(
                        scalars_path=training_paths.scalars_path,
                        learner=learner,
                        metrics=latest_metrics,
                        start_time=start_time,
                    )
                if rank0 and tensorboard_logger is not None:
                    tensorboard_logger.log_training_step(
                        update_count=int(learner.update_count),
                        policy_version=int(learner.get_policy_version()),
                        wall_clock_seconds=time.time() - start_time,
                        metrics=latest_metrics,
                    )
                if rank0 and learner.update_count % checkpoint_interval_updates == 0:
                    ckpt_path = training_paths.checkpoints_dir / f"checkpoint_{learner.update_count}.pt"
                    _write_checkpoint(
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        stack=stack,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                    )
                    tracker_payload = _publish_checkpoint_aliases(
                        stack=stack,
                        training_paths=training_paths,
                        artifacts=artifacts,
                        checkpoint_path=ckpt_path,
                        learner=learner,
                        latest_metrics=latest_metrics,
                    )
                    _maybe_log_structured_mainmove_guard(
                        training_paths=training_paths,
                        learner=learner,
                        latest_metrics=latest_metrics,
                        dev_eval_summary=last_dev_eval_summary,
                    )
                    if tensorboard_logger is not None:
                        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))

                    if main_residual_policy_enabled:
                        latest_metrics["main_residual_snapshot_registry_skipped"] = 1.0
                    else:
                        if learner.model is None:
                            raise RuntimeError("Cannot persist a snapshot registry entry without a learner model")
                        candidate_policy_id = _persist_snapshot_registry_entry(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            checkpoint_path=ckpt_path,
                            model_state_dict=learner.model.state_dict(),
                            config_hash256=config_hash256,
                            device=device,
                            update=int(learner.update_count),
                            policy_version=int(learner.get_policy_version()),
                            model=learner.model,
                        )
                        defer_noleague_baseline_alias_refresh = _should_defer_noleague_baseline_alias_refresh(
                            stack=stack,
                            experiment_role=experiment_role,
                            update_count=int(learner.update_count),
                        )
                        if _is_noleague_baseline_role(experiment_role) and not defer_noleague_baseline_alias_refresh:
                            _ensure_noleague_baseline_anchor(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                learner=learner,
                                device=device,
                                config_hash256=config_hash256,
                                permit_current_run_alias=True,
                                source_checkpoint_path=ckpt_path,
                                update=int(learner.update_count),
                            )
                        refresh_opponent_pool_requested = True
                    if (not main_residual_policy_enabled) and async_promotion_gate_enabled:
                        if async_promotion_gate_executor is None:
                            raise RuntimeError("async promotion gate is enabled but the worker pool was not created")
                        if pending_promotion_gate is not None:
                            promotion_passed = _process_completed_promotion_gate(
                                pending_gate=pending_promotion_gate,
                                stack=stack,
                                artifacts=artifacts,
                                training_paths=training_paths,
                            )
                            pending_promotion_gate = None
                            if promotion_passed:
                                refresh_opponent_pool_requested = True
                        league_reference_update = (
                            None
                            if "league_effective_update" not in latest_metrics
                            else int(latest_metrics["league_effective_update"])
                        )
                        league = stack.config.league
                        reference_update = int(
                            int(learner.update_count) if league_reference_update is None else league_reference_update
                        )
                        if (
                            league is not None
                            and league.enabled
                            and league.promotion_gate_enabled
                            and reference_update >= int(league.warmup.first_updates)
                            and (
                                not bool(getattr(league.warmup, "eval_gate_enabled", False))
                                or bool(league_eval_warmup_gate_open)
                            )
                        ):
                            registry = SnapshotRegistry.load(training_paths.snapshots_dir / REGISTRY_FILENAME)
                            anchor_policy_ids, anchor_specs, pinned_anchor_snapshot_ids = (
                                _resolve_promotion_gate_anchor_specs(
                                    stack=stack,
                                    training_paths=training_paths,
                                )
                            )
                            snapshot_index = _snapshot_meta_by_policy_id(registry)
                            candidate_snapshot = snapshot_index.get(candidate_policy_id)
                            if candidate_snapshot is None:
                                raise RuntimeError(
                                    f"Could not resolve persisted candidate snapshot for promotion gate: {candidate_policy_id}"
                                )
                            newly_pinned_snapshot_ids = _pin_snapshot_ids(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                snapshot_ids=(candidate_policy_id, *pinned_anchor_snapshot_ids),
                            )
                            request = _build_async_promotion_gate_request(
                                stack=stack,
                                run_dir=artifacts.run_dir,
                                candidate_policy_id=candidate_policy_id,
                                candidate_snapshot_path=candidate_snapshot.path,
                                update_count=int(learner.update_count),
                                policy_version=int(learner.get_policy_version()),
                                run_id256=run_id256,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                anchor_policy_ids=anchor_policy_ids,
                                anchor_specs=anchor_specs,
                                eval_device_override=_resolve_async_promotion_gate_device(
                                    stack=stack,
                                    learner_device=device,
                                    distributed_context=ddp_context,
                                ),
                            )
                            pending_promotion_gate = PendingPromotionGate(
                                future=async_promotion_gate_executor.submit(_run_async_promotion_gate_worker, request),
                                request=request,
                                pinned_snapshot_ids=newly_pinned_snapshot_ids,
                            )
                            print(
                                _format_scheduled_async_promotion_gate_message_impl(
                                    update_count=int(learner.update_count),
                                    candidate_policy_id=candidate_policy_id,
                                    anchor_names=tuple(anchor_policy_ids.keys()),
                                )
                            )
                        elif league is not None and league.enabled and league.promotion_gate_enabled:
                            if reference_update < int(league.warmup.first_updates):
                                print(
                                    _format_promotion_gate_skipped_league_warmup_message_impl(
                                        update_count=int(learner.update_count),
                                        effective_update=reference_update,
                                        threshold=int(league.warmup.first_updates),
                                        candidate_policy_id=candidate_policy_id,
                                    )
                                )
                            else:
                                print(
                                    _format_promotion_gate_skipped_eval_warmup_gate_message_impl(
                                        update_count=int(learner.update_count),
                                        effective_update=reference_update,
                                        candidate_policy_id=candidate_policy_id,
                                    )
                                )
                    else:
                        promotion_passed = _run_snapshot_promotion_gate(
                            stack=stack,
                            contract=contract,
                            artifacts=artifacts,
                            training_paths=training_paths,
                            learner=learner,
                            candidate_policy_id=candidate_policy_id,
                            update_count=int(learner.update_count),
                            league_reference_update=(
                                None
                                if "league_effective_update" not in latest_metrics
                                else int(latest_metrics["league_effective_update"])
                            ),
                            league_eval_warmup_gate_open=league_eval_warmup_gate_open,
                            policy_version=int(learner.get_policy_version()),
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                        )
                        if promotion_passed:
                            refresh_opponent_pool_requested = True

                if rank0 and _should_run_periodic_dev_eval(stack, update_count=int(learner.update_count)):
                    defer_noleague_baseline_alias_refresh = _should_defer_noleague_baseline_alias_refresh(
                        stack=stack,
                        experiment_role=experiment_role,
                        update_count=int(learner.update_count),
                    )
                    checkpoint_path = _ensure_current_checkpoint(
                        training_paths=training_paths,
                        learner=learner,
                        stack=stack,
                        device=device,
                        spec_hash256=spec_hash256,
                        algorithm=algorithm,
                    )
                    if async_periodic_dev_eval_enabled:
                        if async_periodic_dev_eval_executor is None:
                            raise RuntimeError("async periodic dev eval is enabled but the worker pool was not created")
                        if pending_periodic_dev_eval is not None:
                            completed_summary, guard_event = _process_completed_periodic_dev_eval(
                                pending_eval=pending_periodic_dev_eval,
                                stack=stack,
                                contract=contract,
                                artifacts=artifacts,
                                training_paths=training_paths,
                                runtime=runtime,
                                learner=learner,
                                device=device,
                                run_id256=run_id256,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                last_rollback_update=last_checkpoint_guard_rollback_update,
                                tensorboard_logger=tensorboard_logger,
                                process_pool_executor=periodic_dev_eval_process_pool,
                            )
                            pending_periodic_dev_eval = None
                            last_dev_eval_summary = completed_summary
                            last_dev_eval_update_count = int(completed_summary["update_count"])
                            if guard_event is not None:
                                guard_event_for_distributed_sync = guard_event
                                refresh_opponent_pool_requested = True
                                last_checkpoint_guard_rollback_update = int(learner.update_count)
                                pending_promotion_gate = _drop_stale_pending_promotion_gate(
                                    stack=stack,
                                    training_paths=training_paths,
                                    run_dir=artifacts.run_dir,
                                    pending_gate=pending_promotion_gate,
                                    rollback_best_update_count=int(guard_event["best_update_count"]),
                                )
                                prefetched_runtime_batch = None
                                print(_format_checkpoint_guard_rollback_message_impl(guard_event))
                        opponent_specs, pinned_snapshot_ids = _resolve_periodic_dev_eval_opponent_specs(
                            stack=stack,
                            run_dir=artifacts.run_dir,
                        )
                        newly_pinned_snapshot_ids = _pin_snapshot_ids(
                            stack=stack,
                            training_paths=training_paths,
                            run_dir=artifacts.run_dir,
                            snapshot_ids=pinned_snapshot_ids,
                        )
                        async_eval_device = _resolve_async_periodic_dev_eval_device(
                            stack=stack,
                            learner_device=device,
                            distributed_context=ddp_context,
                        )
                        periodic_parallel_workers = max(
                            1,
                            int(
                                getattr(
                                    stack.config.evaluation,
                                    "periodic_dev_eval_parallel_workers",
                                    1,
                                )
                            ),
                        )
                        request = _build_async_periodic_dev_eval_request(
                            stack=stack,
                            checkpoint_path=checkpoint_path,
                            focal_policy_id=_current_focal_policy_id(learner=learner),
                            update_count=int(learner.update_count),
                            policy_version=int(learner.get_policy_version()),
                            run_dir=artifacts.run_dir,
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                            artifact_dir_name="dev_eval",
                            artifact_scope="periodic_dev_eval",
                            paired_seeds=tuple(_periodic_dev_eval_schedule(stack)[2]),
                            opponents=tuple(opponent_specs),
                            eval_device_override=async_eval_device,
                            parallel_workers=periodic_parallel_workers,
                            parallel_worker_devices=_resolved_periodic_dev_eval_worker_devices(
                                stack=stack,
                                parallel_workers=periodic_parallel_workers,
                                explicit_worker_devices=tuple(
                                    getattr(
                                        stack.config.evaluation,
                                        "periodic_dev_eval_parallel_worker_devices",
                                        (),
                                    )
                                ),
                                eval_device=str(
                                    getattr(
                                        stack.config.evaluation,
                                        "eval_device",
                                        "cpu",
                                    )
                                ),
                                learner_device=device,
                            ),
                        )
                        if periodic_dev_eval_process_pool is None and int(request.parallel_workers) > 1:
                            periodic_dev_eval_process_pool = ProcessPoolExecutor(
                                max_workers=int(request.parallel_workers),
                                mp_context=mp.get_context("spawn"),
                            )
                        pending_periodic_dev_eval = PendingPeriodicDevEval(
                            future=async_periodic_dev_eval_executor.submit(
                                _run_async_periodic_dev_eval_worker,
                                request,
                                periodic_dev_eval_process_pool,
                            ),
                            request=request,
                            pinned_snapshot_ids=tuple(newly_pinned_snapshot_ids),
                            latest_metrics=dict(latest_metrics),
                        )
                        print(
                            _format_periodic_dev_eval_scheduled_message_impl(
                                update_count=int(learner.update_count),
                                worker_devices=request.parallel_worker_devices,
                                fallback_eval_device=str(stack.config.evaluation.eval_device),
                                anchor_names=tuple(spec.display_name for spec in request.opponents),
                            )
                        )
                        if defer_noleague_baseline_alias_refresh:
                            _ensure_noleague_baseline_anchor(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                learner=learner,
                                device=device,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                permit_current_run_alias=True,
                                source_checkpoint_path=checkpoint_path,
                                update=int(learner.update_count),
                            )
                    else:
                        if (
                            periodic_dev_eval_process_pool is None
                            and stack.config.evaluation is not None
                            and int(getattr(stack.config.evaluation, "periodic_dev_eval_parallel_workers", 1)) > 1
                        ):
                            periodic_dev_eval_process_pool = ProcessPoolExecutor(
                                max_workers=int(stack.config.evaluation.periodic_dev_eval_parallel_workers),
                                mp_context=mp.get_context("spawn"),
                            )
                        summary_payload = _run_periodic_dev_eval(
                            stack=stack,
                            contract=contract,
                            artifacts=artifacts,
                            training_paths=training_paths,
                            learner=learner,
                            device=device,
                            run_id256=run_id256,
                            config_hash256=config_hash256,
                            spec_hash256=spec_hash256,
                            process_pool_executor=periodic_dev_eval_process_pool,
                        )
                        anchor_keys = sorted(cast(dict[str, Any], summary_payload["anchor_scores"]).keys())
                        print(
                            _format_periodic_dev_eval_console_message_impl(
                                label="Periodic dev eval",
                                update_count=int(learner.update_count),
                                aggregate_score=float(summary_payload["aggregate_score"]),
                                anchor_names=anchor_keys,
                                opponent_slug=_slug_policy_id(anchor_keys[0]) if anchor_keys else None,
                            )
                        )
                        effective_summary = summary_payload
                        tracker_before_dev_eval = _load_checkpoint_tracker(training_paths)
                        existing_best_record = tracker_before_dev_eval.get("best")
                        if not isinstance(existing_best_record, Mapping):
                            existing_best_record = None
                        confirmatory_plan = _build_confirmatory_dev_eval_plan_impl(
                            stack=stack,
                            existing_best_record=cast(Mapping[str, Any] | None, existing_best_record),
                            dev_eval_summary=effective_summary,
                        )
                        if confirmatory_plan is not None:
                            effective_summary = _run_periodic_dev_eval(
                                stack=stack,
                                contract=contract,
                                artifacts=artifacts,
                                training_paths=training_paths,
                                learner=learner,
                                device=device,
                                run_id256=run_id256,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                artifact_dir_name="dev_eval_confirmatory",
                                artifact_scope="periodic_dev_eval_confirmatory",
                                paired_seeds_override=confirmatory_plan.paired_seeds,
                                persist_summary=False,
                                update_stall_monitor=False,
                                batched_inference_override=False,
                                process_pool_executor=periodic_dev_eval_process_pool,
                            )
                            print(
                                _format_confirmatory_dev_eval_message_impl(
                                    update_count=int(learner.update_count),
                                    paired_seed_count=len(confirmatory_plan.paired_seeds),
                                    aggregate_score=float(effective_summary["aggregate_score"]),
                                    reasons=confirmatory_plan.reasons,
                                    seed_file=confirmatory_plan.seed_file,
                                )
                            )
                        _persist_periodic_dev_eval_result(training_paths=training_paths, payload=effective_summary)
                        last_dev_eval_summary = effective_summary
                        last_dev_eval_update_count = int(learner.update_count)
                        league_eval_warmup_gate_status = _sync_runtime_league_eval_warmup_gate(
                            runtime=runtime,
                            stack=stack,
                            dev_eval_summary=effective_summary,
                        )
                        league_eval_warmup_gate_open = bool(league_eval_warmup_gate_status["open"])
                        if bool(league_eval_warmup_gate_status.get("enabled", False)):
                            print(_format_league_eval_warmup_gate_message_impl(league_eval_warmup_gate_status))
                        ckpt_path = _ensure_current_checkpoint(
                            training_paths=training_paths,
                            learner=learner,
                            stack=stack,
                            device=device,
                            spec_hash256=spec_hash256,
                            algorithm=algorithm,
                        )
                        tracker_payload = _publish_checkpoint_aliases(
                            stack=stack,
                            training_paths=training_paths,
                            artifacts=artifacts,
                            checkpoint_path=ckpt_path,
                            learner=learner,
                            latest_metrics=latest_metrics,
                            dev_eval_summary=effective_summary,
                        )
                        _maybe_log_structured_mainmove_guard(
                            training_paths=training_paths,
                            learner=learner,
                            latest_metrics=latest_metrics,
                            dev_eval_summary=effective_summary,
                        )
                        guard_event = _maybe_rollback_to_best_checkpoint(
                            stack=stack,
                            training_paths=training_paths,
                            artifacts=artifacts,
                            runtime=runtime,
                            learner=learner,
                            model=model,
                            device=device,
                            spec_hash256=spec_hash256,
                            algorithm=algorithm,
                            latest_metrics=latest_metrics,
                            dev_eval_summary=effective_summary,
                            last_rollback_update=last_checkpoint_guard_rollback_update,
                        )
                        if guard_event is not None:
                            guard_event_for_distributed_sync = guard_event
                            refresh_opponent_pool_requested = True
                            last_checkpoint_guard_rollback_update = int(learner.update_count)
                            pending_promotion_gate = _drop_stale_pending_promotion_gate(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                pending_gate=pending_promotion_gate,
                                rollback_best_update_count=int(guard_event["best_update_count"]),
                            )
                            prefetched_runtime_batch = None
                            print(_format_checkpoint_guard_rollback_message_impl(guard_event))
                        if defer_noleague_baseline_alias_refresh:
                            _ensure_noleague_baseline_anchor(
                                stack=stack,
                                training_paths=training_paths,
                                run_dir=artifacts.run_dir,
                                learner=learner,
                                device=device,
                                config_hash256=config_hash256,
                                spec_hash256=spec_hash256,
                                permit_current_run_alias=True,
                                update=int(learner.update_count),
                            )
                        if tensorboard_logger is not None:
                            tensorboard_logger.log_periodic_dev_eval(effective_summary, step=int(learner.update_count))
                            tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))
                        audit_request = _maybe_request_b2_disagreement_audit(
                            stack=stack,
                            training_paths=training_paths,
                            artifacts=artifacts,
                            dev_eval_summary=effective_summary,
                        )
                        if audit_request is not None:
                            latest_metrics["b2_disagreement_audit_requested"] = 1.0
                            print(_format_b2_disagreement_audit_request_message_impl(audit_request))
                        early_cutoff_payload = _update_early_cutoff(
                            stack=stack,
                            training_paths=training_paths,
                            update_count=int(learner.update_count),
                            summary_payload=effective_summary,
                        )
                        if early_cutoff_payload is not None and bool(early_cutoff_payload.get("should_stop", False)):
                            latest_metrics.update(_early_cutoff_metric_updates_impl(early_cutoff_payload))
                            print(
                                _format_early_cutoff_triggered_message_impl(
                                    early_cutoff_payload,
                                    update_count=int(learner.update_count),
                                )
                            )
                            stop_requested = True
                if ddp_context.enabled:
                    distributed_guard_event = broadcast_object(
                        guard_event_for_distributed_sync if rank0 else None,
                        context=ddp_context,
                    )
                    if distributed_guard_event is not None:
                        distributed_barrier(ddp_context)
                        if not rank0:
                            _restore_checkpoint_to_latest_alias(
                                checkpoint_path=training_paths.latest_checkpoint_path,
                                training_paths=training_paths,
                                learner=learner,
                                stack=stack,
                                device=device,
                                expected_spec_hash256=spec_hash256,
                                algorithm=algorithm,
                                restore_counters=False,
                            )
                            prefetched_runtime_batch = None
                        runtime.maybe_publish_snapshot(
                            learner_model=model,
                            learner_update_count=int(learner.update_count),
                            force=True,
                        )
                        refresh_opponent_pool_requested = True
                        distributed_barrier(ddp_context)
                    refresh_requested = all_reduce_float(
                        1.0 if refresh_opponent_pool_requested else 0.0,
                        context=ddp_context,
                        op="sum",
                    )
                    if refresh_requested > 0.0:
                        distributed_barrier(ddp_context)
                        runtime.refresh_opponent_pool(allow_registry_write=rank0)
                        distributed_barrier(ddp_context)
                    stop_requested = (
                        all_reduce_float(1.0 if stop_requested else 0.0, context=ddp_context, op="sum") > 0.0
                    )
                elif refresh_opponent_pool_requested:
                    runtime.refresh_opponent_pool()
                if stop_requested:
                    break
        finally:
            if pending_promotion_gate is not None:
                promotion_passed = _process_completed_promotion_gate(
                    pending_gate=pending_promotion_gate,
                    stack=stack,
                    artifacts=artifacts,
                    training_paths=training_paths,
                )
                pending_promotion_gate = None
                if promotion_passed:
                    runtime.refresh_opponent_pool()
            if pending_periodic_dev_eval is not None:
                completed_summary, guard_event = _process_completed_periodic_dev_eval(
                    pending_eval=pending_periodic_dev_eval,
                    stack=stack,
                    contract=contract,
                    artifacts=artifacts,
                    training_paths=training_paths,
                    runtime=runtime,
                    learner=learner,
                    device=device,
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                    last_rollback_update=last_checkpoint_guard_rollback_update,
                    tensorboard_logger=tensorboard_logger,
                    process_pool_executor=periodic_dev_eval_process_pool,
                )
                pending_periodic_dev_eval = None
                last_dev_eval_summary = completed_summary
                last_dev_eval_update_count = int(completed_summary["update_count"])
                if guard_event is not None:
                    last_checkpoint_guard_rollback_update = int(learner.update_count)
                    pending_promotion_gate = _drop_stale_pending_promotion_gate(
                        stack=stack,
                        training_paths=training_paths,
                        run_dir=artifacts.run_dir,
                        pending_gate=pending_promotion_gate,
                        rollback_best_update_count=int(guard_event["best_update_count"]),
                    )
                    prefetched_runtime_batch = None
            if prefetch_executor is not None:
                prefetch_executor.shutdown(wait=False, cancel_futures=True)
            if async_promotion_gate_executor is not None:
                async_promotion_gate_executor.shutdown(wait=True, cancel_futures=False)
            if async_periodic_dev_eval_executor is not None:
                async_periodic_dev_eval_executor.shutdown(wait=True, cancel_futures=False)
            if periodic_dev_eval_process_pool is not None:
                periodic_dev_eval_process_pool.shutdown(wait=True, cancel_futures=False)
            runtime.close()

    if profiler is not None and profiler_trace_dir is not None:
        trace_path = profiler_trace_dir / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        print(_format_torch_profiler_trace_written_message_impl(trace_path))

    if rank0 and _is_noleague_baseline_role(experiment_role):
        _ensure_noleague_baseline_anchor(
            stack=stack,
            training_paths=training_paths,
            run_dir=artifacts.run_dir,
            learner=learner,
            device=device,
            config_hash256=config_hash256,
            permit_current_run_alias=True,
            update=int(learner.update_count),
        )

    if not latest_metrics:
        raise RuntimeError("The canonical single-node run finished without producing learner metrics")
    if not rank0:
        return latest_metrics
    final_checkpoint_path = _ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        stack=stack,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
    )
    final_dev_eval_summary = last_dev_eval_summary if last_dev_eval_update_count == int(learner.update_count) else None
    tracker_payload = _publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=final_checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    finalize_guard_event = _maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        runtime=runtime,
        learner=learner,
        device=device,
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        latest_metrics=latest_metrics,
        dev_eval_summary=final_dev_eval_summary,
    )
    if finalize_guard_event is not None:
        print(_format_checkpoint_guard_final_selection_message_impl(finalize_guard_event))
        tracker_payload = _load_checkpoint_tracker(training_paths)
    if tensorboard_logger is not None:
        tensorboard_logger.log_checkpoint_tracker(tracker_payload, step=int(learner.update_count))
    if early_cutoff_payload is not None and bool(early_cutoff_payload.get("should_stop", False)):
        latest_metrics.setdefault("early_cutoff_triggered", 1.0)
    return latest_metrics


def main() -> None:
    parser = _build_train_arg_parser()
    args = parser.parse_args()
    run_label = _resolve_run_label(parser, args.run_label, args.run_id_alias)

    num_envs = _require_positive_int("--num-envs", args.num_envs)
    unroll_length = _require_positive_int("--unroll-length", args.unroll_length)
    max_updates = _require_positive_int("--max-updates", args.max_updates)
    max_wall_clock_minutes = _require_positive_optional_float(
        "--max-wall-clock-minutes",
        args.max_wall_clock_minutes,
    )
    ddp_timeout_seconds = _require_positive_int("--ddp-timeout-seconds", args.ddp_timeout_seconds)
    stack = load_stack_config(args.stack_config)
    stack = apply_stack_overrides(stack, parse_override_tokens(args.config_override))
    stack = _apply_training_flag_overrides(
        stack,
        enable_profile_timers=bool(args.profile_timers),
        enable_torch_profiler=bool(args.torch_profiler),
    )
    training_config = stack.config.training
    manifest_only_reason = _manifest_scaffold_only_reason(stack)
    if training_config is None and manifest_only_reason is None:
        parser.error("stack config is missing training")

    public_demo_enabled = bool(args.public_demo)
    resume_run_dir = None if args.resume_run_dir is None else args.resume_run_dir.resolve()
    resume_checkpoint_path = _resolve_resume_checkpoint_path(
        resume_from=str(args.resume_from),
        resume_run_dir=resume_run_dir,
    )
    seed_snapshot_run_dir = (
        args.seed_snapshot_run_dir.resolve()
        if args.seed_snapshot_run_dir is not None
        else _infer_seed_snapshot_run_dir_from_resume_checkpoint(
            stack=stack,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_run_dir=resume_run_dir,
        )
    )
    seed_snapshot_run_dir_auto_inferred = args.seed_snapshot_run_dir is None and seed_snapshot_run_dir is not None
    if public_demo_enabled and (resume_run_dir is not None or resume_checkpoint_path is not None):
        parser.error("Public demo mode does not support checkpoint resume")
    if public_demo_enabled:
        public_demo_bundle = public_demo_spec_bundle()
        assert_spec_bundle_contract(args.spec_hash, public_demo_bundle)
        spec_bundle = public_demo_bundle
        spec_hash256 = public_demo_spec_hash256()
        simulator_info = public_demo_simulator_info()
    else:
        simulator_contract = load_verified_simulator_contract(stack.root, expected_spec_hash=args.spec_hash)
        spec_bundle = simulator_contract.spec_bundle
        spec_hash256 = simulator_contract.spec_hash256
        simulator_info = simulator_contract.simulator
    config_hash256 = compute_config_hash256(stack)
    _require_matching_hash(
        flag_name="--config-hash",
        expected=_expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    ddp_context = distributed_context_from_env(
        force=bool(args.ddp),
        backend=_resolve_ddp_backend(stack, device_override=str(args.device), backend=str(args.ddp_backend)),
    )
    ddp_device_override_error = _ddp_indexed_cuda_override_error(str(args.device), world_size=int(ddp_context.world_size))
    if ddp_device_override_error is not None:
        parser.error(ddp_device_override_error)
    if ddp_context.enabled and torch.cuda.is_available() and int(torch.cuda.device_count()) > 0:
        torch.cuda.set_device(int(ddp_context.local_rank) % int(torch.cuda.device_count()))
    ddp_context = init_process_group_if_needed(ddp_context, timeout_seconds=ddp_timeout_seconds)
    rank0 = (not ddp_context.enabled) or ddp_context.is_rank0
    resolved_topology: ResolvedTrainingTopology | None = None
    if bool(args.autoscale or args.autoscale_dry_run):
        resolved_topology = _resolve_autoscale_topology(
            stack=stack,
            hardware_profile_name=str(args.hardware_profile),
            runtime_mode=cast(QueueRuntimeMode, args.runtime_mode),
        )
        if not ddp_context.enabled and str(resolved_topology.resolved_learner_parallelism) == "ddp":
            if args.autoscale_dry_run:
                pass
            else:
                parser.error(
                    "autoscale resolved a multi-GPU DDP topology; launch with torchrun/--ddp or use --autoscale-dry-run"
                )
        if ddp_context.enabled:
            try:
                validate_ddp_world_size(resolved_topology, world_size=int(ddp_context.world_size))
            except ValueError as exc:
                parser.error(str(exc))
        if args.autoscale_dry_run:
            if rank0:
                print(
                    json.dumps(
                        {
                            "format": "autoscale_training_topology_v1",
                            "hardware_profile": str(args.hardware_profile),
                            "runtime_mode": str(args.runtime_mode),
                            "scaling_request": _scaling_request_from_config(training_config).to_dict()
                            if hasattr(_scaling_request_from_config(training_config), "to_dict")
                            else _scaling_request_from_config(training_config).__dict__,
                            "resolved_topology": resolved_topology.to_dict(),
                            "distributed": ddp_context.to_dict(),
                        },
                        sort_keys=True,
                        indent=2,
                    )
                )
            destroy_process_group_if_initialized()
            return
        num_envs = int(resolved_topology.total_envs)

    git_commit = _git_commit()
    start_nonce = int(broadcast_object(_start_nonce() if rank0 else None, context=ddp_context))
    manifest_dict: dict[str, Any] | None = None
    if resume_run_dir is None:
        run_id256 = compute_run_id256(spec_hash256, config_hash256, git_commit or None, start_nonce)
        run_id64 = f"{compute_run_id64(spec_hash256, config_hash256, git_commit or None, start_nonce):016x}"
        run_dir_name = run_label or default_run_dir_name(run_id64)
    else:
        artifacts = _run_artifacts_from_existing_run_dir(resume_run_dir)
        manifest_dict = _load_json_object(artifacts.manifest_path, label="resume manifest")
        run_id256 = str(manifest_dict.get("run_id256", "")).strip().lower()
        run_id64 = str(manifest_dict.get("run_id64", "")).strip().lower()
        run_dir_name = artifacts.run_dir_name
        existing_spec_hash = str(manifest_dict.get("spec_hash256", "")).strip().lower()
        existing_config_hash = str(manifest_dict.get("config_hash256", "")).strip().lower()
        if existing_spec_hash != spec_hash256:
            raise RuntimeError(
                f"resume run spec hash mismatch: expected {spec_hash256}, found {existing_spec_hash} in {artifacts.manifest_path}"
            )
        if existing_config_hash != config_hash256:
            raise RuntimeError(
                f"resume run config hash mismatch: expected {config_hash256}, found {existing_config_hash} in {artifacts.manifest_path}"
            )

    print_startup_banner(
        spec_hash256,
        config_hash256,
        run_id64=run_id64,
        run_id256=run_id256,
        run_label=run_label or ("" if resume_run_dir is None else run_dir_name),
        run_dir_name=run_dir_name,
        spec_mismatch_policy=_spec_mismatch_policy(stack),
    )
    print(
        _format_spec_bundle_status_message_impl(
            public_demo_enabled=public_demo_enabled,
            compatibility_hash=simulator_info.get("compatibility_hash", ""),
            spec_hash256=spec_hash256,
        )
    )
    print(_format_loaded_stack_config_message_impl(len(stack.components)))

    if ddp_context.enabled:
        try:
            device = resolve_distributed_learner_device(str(args.device), context=ddp_context)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        device = _resolve_device(stack, args.device)
    profile = _resolve_runtime_profile(stack, args.profile)
    seed = _resolve_seed(stack, args.seed)
    actor_device_layout = _manifest_actor_device_layout(
        stack=stack,
        num_envs=num_envs,
        unroll_length=unroll_length,
        profile=profile,
        seed=seed,
        pass_action_id=int(spec_bundle["action"]["pass_action_id"]),
        runtime_mode=cast(QueueRuntimeMode, args.runtime_mode),
        learner_device=device,
        resolved_topology=resolved_topology,
        rank_local_actor_devices=bool(ddp_context.enabled),
    )
    policy_set_selection, policy_set_selection_details = _resolve_policy_set_selection(
        stack,
        snapshot_registry_path=args.snapshot_registry_json,
        dev_eval_summaries_path=args.dev_eval_summaries_json,
    )
    manifest = RunManifest(
        run_id256=run_id256,
        run_id64=run_id64,
        start_nonce=start_nonce,
        git_commit=git_commit,
        git_dirty=_git_dirty(),
        spec_hash256=spec_hash256,
        config_hash256=config_hash256,
        simulator=simulator_info,
        spec_bundle=spec_bundle,
        config_canonical=canonical_config_dict(stack),
        seed_files=build_seed_file_manifest(stack.seed_sets, root=stack.root),
        hardware=_hardware_summary(
            device,
            actor_device=("cpu" if stack.config.system is None else stack.config.system.actor_device),
            actor_device_layout=actor_device_layout,
        ),
        evaluation_pinning=_evaluation_pinning(stack),
        policy_set_selection=policy_set_selection,
        policy_set_selection_details=policy_set_selection_details,
    )
    if resume_run_dir is None:
        if rank0:
            artifacts = write_run_artifacts(
                stack.root / "runs",
                manifest,
                run_label=run_label or None,
            )
            run_dir_text = artifacts.run_dir.as_posix()
        else:
            run_dir_text = ""
        run_dir_text = str(broadcast_object(run_dir_text, context=ddp_context))
        if not rank0:
            artifacts = _run_artifacts_from_existing_run_dir(Path(run_dir_text))
        distributed_barrier(ddp_context)
    else:
        artifacts = _run_artifacts_from_existing_run_dir(resume_run_dir)
    tensorboard_logger: TensorBoardLogger | None = None
    if rank0:
        run_summary_payload = _load_json_object(artifacts.run_summary_path, label="run summary")
        run_summary_payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(args.runtime_mode)
        run_summary_payload["policy_set_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
        run_summary_payload["distributed"] = ddp_context.to_dict()
        if resolved_topology is not None:
            run_summary_payload["autoscale_topology"] = resolved_topology.to_dict()
        if training_config is not None:
            run_summary_payload["training_controls"] = _training_controls_payload_impl(
                training_config,
                max_wall_clock_minutes=max_wall_clock_minutes,
                include_wall_clock_budget=True,
            )
        if args.b1_baseline_run_dir is not None:
            run_summary_payload["b1_baseline_run_dir"] = args.b1_baseline_run_dir.resolve().as_posix()
        if seed_snapshot_run_dir is not None:
            run_summary_payload["seed_snapshot_run_dir"] = seed_snapshot_run_dir.as_posix()
            run_summary_payload["seed_snapshot_run_dir_auto_inferred"] = seed_snapshot_run_dir_auto_inferred
        run_summary_payload["stack_config_path"] = args.stack_config.resolve().as_posix()
        if resume_checkpoint_path is not None:
            run_summary_payload["resume"] = {
                "enabled": True,
                "resume_run_dir": None if resume_run_dir is None else resume_run_dir.as_posix(),
                "resume_checkpoint_path": resume_checkpoint_path.as_posix(),
                "reset_optimizer": bool(args.resume_reset_optimizer),
            }
        _write_json(artifacts.run_summary_path, run_summary_payload)

        determinism_payload = _load_json_object(artifacts.determinism_report_path, label="determinism report")
        determinism_payload["runtime_mode"] = "public_demo" if public_demo_enabled else str(args.runtime_mode)
        determinism_payload["policy_selection_mode"] = policy_set_selection_details.get("mode", "unresolved")
        determinism_payload["distributed"] = ddp_context.to_dict()
        if resolved_topology is not None:
            determinism_payload["autoscale_topology"] = resolved_topology.to_dict()
        if training_config is not None:
            determinism_payload["training_controls"] = _training_controls_payload_impl(training_config)
        if args.b1_baseline_run_dir is not None:
            determinism_payload["b1_baseline_run_dir"] = args.b1_baseline_run_dir.resolve().as_posix()
        if seed_snapshot_run_dir is not None:
            determinism_payload["seed_snapshot_run_dir"] = seed_snapshot_run_dir.as_posix()
            determinism_payload["seed_snapshot_run_dir_auto_inferred"] = seed_snapshot_run_dir_auto_inferred
        if resume_checkpoint_path is not None:
            determinism_payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
            determinism_payload["resume_reset_optimizer"] = bool(args.resume_reset_optimizer)
        _write_json(artifacts.determinism_report_path, determinism_payload)

        environment_payload = _load_json_object(artifacts.environment_path, label="environment manifest")
        environment_payload["cwd"] = stack.root.as_posix()
        environment_payload["argv"] = sys.argv
        environment_payload["hardware"] = manifest.hardware
        environment_payload["distributed"] = ddp_context.to_dict()
        if resolved_topology is not None:
            environment_payload["autoscale_topology"] = resolved_topology.to_dict()
        if resume_checkpoint_path is not None:
            environment_payload["resume_checkpoint_path"] = resume_checkpoint_path.as_posix()
            environment_payload["resume_reset_optimizer"] = bool(args.resume_reset_optimizer)
        _write_json(artifacts.environment_path, environment_payload)
        tensorboard_logger = TensorBoardLogger(artifacts.layout.tensorboard_dir)
        if not tensorboard_logger.enabled:
            unavailable_reason = tensorboard_unavailable_reason()
            print(_format_tensorboard_disabled_message_impl(unavailable_reason), file=sys.stderr)
        else:
            tensorboard_logger.log_run_context(
                manifest=manifest.to_dict(),
                environment=environment_payload,
                run_summary=run_summary_payload,
                determinism_report=determinism_payload,
            )
        if resume_run_dir is None:
            print(_format_manifest_written_message_impl(artifacts.manifest_path))
        else:
            print(_format_resume_run_dir_message_impl(artifacts.run_dir))

    try:
        if public_demo_enabled:
            staged = stage_public_demo_run(artifacts.run_dir)
            print(
                _format_public_demo_staged_message_impl(
                    mode=PUBLIC_DEMO_MODE,
                    policy_count=len(staged.policy_ids),
                    catalog_path=staged.catalog_path,
                )
            )
            print(_format_public_demo_disclaimer_message_impl())
            return

        if manifest_only_reason is not None:
            _print_manifest_only_message(manifest_only_reason)
            return

        runtime_prerequisite_failure = _runtime_training_prerequisite_failure(stack)
        if runtime_prerequisite_failure is not None:
            _raise_runtime_prerequisite_failure(runtime_prerequisite_failure)

        assert training_config is not None
        checkpoint_interval_updates = _require_positive_int(
            "--checkpoint-interval-updates",
            args.checkpoint_interval_updates
            if args.checkpoint_interval_updates is not None
            else int(training_config.checkpoint_interval_updates),
        )

        profile_timers = bool(training_config.profile_timers)
        torch_profiler = bool(training_config.torch_profiler)
        if profile_timers or torch_profiler:
            print(_format_structured_profiling_enabled_message_impl(training_config))

        metrics = _run_minimal_training(
            stack=stack,
            contract=simulator_contract,
            artifacts=artifacts,
            num_envs=num_envs,
            unroll_length=unroll_length,
            max_updates=max_updates,
            max_wall_clock_minutes=max_wall_clock_minutes,
            profile=profile,
            device=device,
            seed=seed,
            checkpoint_interval_updates=checkpoint_interval_updates,
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=spec_hash256,
            runtime_mode=cast(QueueRuntimeMode, args.runtime_mode),
            b1_baseline_run_dir=None if args.b1_baseline_run_dir is None else args.b1_baseline_run_dir.resolve(),
            seed_snapshot_run_dir=seed_snapshot_run_dir,
            seed_snapshot_run_dir_auto_inferred=seed_snapshot_run_dir_auto_inferred,
            profile_timers=profile_timers,
            torch_profiler=torch_profiler,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_allow_config_mismatch=bool(args.resume_allow_config_mismatch),
            resume_reset_optimizer=bool(args.resume_reset_optimizer),
            tensorboard_logger=tensorboard_logger,
            resolved_topology=resolved_topology,
            distributed_context=ddp_context,
        )
        print(_format_training_completed_message_impl(metrics))
        if float(metrics.get("early_cutoff_triggered", 0.0)) >= 0.5:
            print(_format_training_stopped_by_early_cutoff_message_impl(metrics))
    finally:
        if tensorboard_logger is not None:
            tensorboard_logger.close()
        destroy_process_group_if_initialized()


if __name__ == "__main__":
    main()
