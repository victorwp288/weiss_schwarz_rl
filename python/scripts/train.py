from __future__ import annotations

# ruff: noqa: F401
# This path-based script intentionally re-exports helper names used by tests and
# by the compatibility-oriented training entrypoint hook modules.
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from weiss_rl.artifacts.manifest import (
    RunArtifacts,
    RunManifest,
    build_seed_file_manifest,
    write_run_artifacts,
)
from weiss_rl.config import (
    StackConfig,
    apply_stack_overrides,
    canonical_config_dict,
    compute_config_hash256,
    load_stack_config,
    parse_override_tokens,
)
from weiss_rl.core.simulator_contract import SimulatorContract, load_verified_simulator_contract
from weiss_rl.core.spec import assert_spec_bundle_contract
from weiss_rl.diagnostics.cli_banner import print_startup_banner
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger, tensorboard_unavailable_reason
from weiss_rl.envs.decision_env import DecisionBoundaryBatch as _DecisionBoundaryBatch
from weiss_rl.eval.harness import ScheduledGame as _ScheduledGame
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.experiments.toy_public_demo import (
    PUBLIC_DEMO_MODE,
    public_demo_simulator_info,
    public_demo_spec_bundle,
    public_demo_spec_hash256,
    stage_public_demo_run,
)
from weiss_rl.league import run_promotion_gate as run_promotion_gate
from weiss_rl.learners.impala_learner import ImpalaLearner
from weiss_rl.learners.ppo_lite_learner import PpoLiteLearner
from weiss_rl.model import PolicyValueModel
from weiss_rl.model import build_policy_value_model as build_policy_value_model
from weiss_rl.models.loading import load_snapshot_eval_model
from weiss_rl.runtime import QueueRuntime, QueueRuntimeMode
from weiss_rl.runtime import build_runtime_config as build_runtime_config
from weiss_rl.training import checkpoint_guard as _checkpoint_guard_helpers
from weiss_rl.training.algorithm_contracts import validate_algorithm_model_contract
from weiss_rl.training.batches import (
    IMPALA_ALGORITHMS,
    PPO_ALGORITHMS,
    bootstrap_values,
    build_learner_batch,
)
from weiss_rl.training.batches import (
    MinimalRollout as _TrainingMinimalRollout,
)
from weiss_rl.training.batches import (
    collect_training_batch as collect_training_batch,
)
from weiss_rl.training.checkpoints import (
    BEST_CHECKPOINT_FILENAME,
    CHECKPOINT_TRACKER_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    ResumeCheckpoint,
    append_checkpoint_guard_event,
    best_checkpoint_record,
    build_checkpoint_record,
    checkpoint_guard_log_path,
    checkpoint_path_for_update,
    current_focal_policy_id,
    extract_structured_guard_b2_anchor_score,
    initialize_model_from_checkpoint,
    load_checkpoint_tracker,
    maybe_log_structured_mainmove_guard,
    publish_checkpoint_aliases,
    relative_path_text,
    resolve_resume_checkpoint_path,
    restore_minimal_train_checkpoint,
    write_checkpoint_tracker,
    write_minimal_train_checkpoint,
    write_scalars_record,
)
from weiss_rl.training.checkpoints import (
    ensure_current_checkpoint as ensure_current_checkpoint,
)
from weiss_rl.training.checkpoints import (
    maybe_finalize_from_best_checkpoint as maybe_finalize_from_best_checkpoint,
)
from weiss_rl.training.checkpoints import (
    maybe_rollback_to_best_checkpoint as maybe_rollback_to_best_checkpoint,
)
from weiss_rl.training.cli import build_train_parser
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
from weiss_rl.training.dev_eval_opponents import periodic_dev_eval_opponents as periodic_dev_eval_opponents
from weiss_rl.training.dev_eval_runner import PeriodicDevEvalRunner
from weiss_rl.training.environments import (
    build_ids_eval_env,
    build_training_env,
    env_pool_config,
    spec_dimensions,
)
from weiss_rl.training.execution import resolve_training_execution_settings
from weiss_rl.training.guidance import (
    apply_guidance_schedule_for_next_update,
    entropy_coef_for_next_update,
    model_guidance_payload,
    public_heuristic_logit_bias_scale_for_next_update,
    restore_model_guidance_from_payload,
    teacher_public_heuristic_coef_for_next_update,
)
from weiss_rl.training.import_contracts import (
    assert_noleague_baseline_config,
    canonical_config_sections,
    config_marks_noleague_baseline,
    is_noleague_baseline_role,
    legacy_noleague_baseline_mode,
    read_optional_hash_file,
    role_from_config_canonical,
    validate_imported_snapshot_contract,
)
from weiss_rl.training.inputs import (
    expected_sha256,
    normalize_sha256,
    require_matching_hash,
    require_positive_int,
    resolve_run_label,
    spec_mismatch_policy,
)
from weiss_rl.training.learner_compile import maybe_compile_learner_model
from weiss_rl.training.learner_factory import build_training_learner
from weiss_rl.training.manifest_layout import manifest_actor_device_layout
from weiss_rl.training.minimal_entrypoint_hooks import run_minimal_training_with_script_hooks
from weiss_rl.training.minimal_loop import MinimalTrainingHooks as MinimalTrainingHooks
from weiss_rl.training.minimal_loop import run_minimal_training as run_minimal_training
from weiss_rl.training.noleague_anchor import ensure_noleague_baseline_anchor
from weiss_rl.training.paths import (
    TrainingPaths,
    run_artifacts_from_existing_run_dir,
    training_paths,
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
from weiss_rl.training.profiling import build_training_profiler as build_training_profiler
from weiss_rl.training.profiling import profile_block as profile_block
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
from weiss_rl.training.report_payloads import (
    augment_determinism_payload,
    augment_environment_payload,
    augment_run_summary_payload,
    profiling_enabled_message,
)
from weiss_rl.training.run_identity import new_run_identity, resume_run_identity
from weiss_rl.training.run_metadata import (
    evaluation_pinning,
    git_commit,
    git_dirty,
    git_output,
    hardware_summary,
    load_json_object,
    manifest_source_path,
    repo_root,
    start_nonce,
)
from weiss_rl.training.script_entrypoint_hooks import (
    ensure_current_checkpoint_with_script_hooks,
    maybe_finalize_from_best_checkpoint_with_script_hooks,
    maybe_rollback_to_best_checkpoint_with_script_hooks,
    periodic_dev_eval_opponents_with_script_hooks,
    run_periodic_dev_eval_with_script_hooks,
    run_snapshot_promotion_gate_with_script_hooks,
    update_stall_monitor_with_script_hooks,
)
from weiss_rl.training.seed_snapshots import (
    import_seed_snapshot_pool,
    validate_seed_snapshot_import_contract,
)
from weiss_rl.training.snapshots import (
    delete_pruned_snapshot_artifacts,
    demote_registry_champions_newer_than,
    persist_snapshot_registry_entry,
    save_snapshot_registry_with_retention,
    seed_snapshot_policy_id,
    sha256_file,
    snapshot_artifact_dir_for_prune,
    sync_snapshot_registry_retention,
    write_imported_snapshot_artifact,
    write_json_file,
    write_snapshot_artifact,
)
from weiss_rl.training.startup import (
    apply_training_flag_overrides,
    manifest_scaffold_only_reason,
    noleague_training_prerequisite_failure,
    print_manifest_only_message,
    raise_noleague_training_prerequisite_failure,
    raise_runtime_prerequisite_failure,
    resolve_device,
    resolve_runtime_profile,
    resolve_seed,
    runtime_training_prerequisite_failure,
)
from weiss_rl.training.torch_threads import (
    central_runtime_actor_torch_threads,
    configure_torch_threads,
    torch_num_threads_scope,
)
from weiss_rl.training.train_entrypoint_main import run_train_main
from weiss_rl.training.warmstart import run_structured_warmstart

_PROMOTION_GATE_RANDOMLEGAL_NAME = "B0 RandomLegal"
_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID = "b0_randomlegal"
_PROMOTION_GATE_NOLEAGUE_BASELINE_NAME = "B1 NoLeague baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID = "b1_noleague_baseline"
_PROMOTION_GATE_NOLEAGUE_BASELINE_CHECKPOINT = "baseline_checkpoint.pt"
MinimalRollout = _TrainingMinimalRollout
DecisionBoundaryBatch = _DecisionBoundaryBatch
ScheduledGame = _ScheduledGame
_LATEST_CHECKPOINT_FILENAME = LATEST_CHECKPOINT_FILENAME
_BEST_CHECKPOINT_FILENAME = BEST_CHECKPOINT_FILENAME
_CHECKPOINT_TRACKER_FILENAME = CHECKPOINT_TRACKER_FILENAME
_IMPALA_ALGORITHMS = IMPALA_ALGORITHMS
_PPO_ALGORITHMS = PPO_ALGORITHMS
_checkpoint_candidate_metric = _checkpoint_guard_helpers.checkpoint_candidate_metric
_confirmatory_dev_eval_request = _checkpoint_guard_helpers.confirmatory_dev_eval_request
_confirmatory_dev_eval_target_pairs = _checkpoint_guard_helpers.confirmatory_dev_eval_target_pairs
_dev_eval_aggregate_score = _checkpoint_guard_helpers.dev_eval_aggregate_score
_dev_eval_confidence_stats = _checkpoint_guard_helpers.dev_eval_confidence_stats
_dev_eval_ineligibility_reasons = _checkpoint_guard_helpers.dev_eval_ineligibility_reasons
_dev_eval_metric_eligible = _checkpoint_guard_helpers.dev_eval_metric_eligible
_dev_eval_worst_natural_timeout_rate = _checkpoint_guard_helpers.dev_eval_worst_natural_timeout_rate
_dev_eval_worst_no_progress_timeout_rate = _checkpoint_guard_helpers.dev_eval_worst_no_progress_timeout_rate
_dev_eval_worst_reason_rate = _checkpoint_guard_helpers.dev_eval_worst_reason_rate
_dev_eval_worst_stall_rate = _checkpoint_guard_helpers.dev_eval_worst_stall_rate
_dev_eval_worst_truncation_rate = _checkpoint_guard_helpers.dev_eval_worst_truncation_rate
_expand_periodic_dev_eval_paired_seeds = _checkpoint_guard_helpers.expand_periodic_dev_eval_paired_seeds
_should_promote_best_checkpoint = _checkpoint_guard_helpers.should_promote_best_checkpoint
_summary_rate = _checkpoint_guard_helpers.summary_rate
_apply_training_flag_overrides = apply_training_flag_overrides
_expected_sha256 = expected_sha256
_manifest_scaffold_only_reason = manifest_scaffold_only_reason
_normalize_sha256 = normalize_sha256
_print_manifest_only_message = print_manifest_only_message
_raise_noleague_training_prerequisite_failure = raise_noleague_training_prerequisite_failure
_raise_runtime_prerequisite_failure = raise_runtime_prerequisite_failure
_require_matching_hash = require_matching_hash
_require_positive_int = require_positive_int
_resolve_device = resolve_device
_resolve_run_label = resolve_run_label
_resolve_runtime_profile = resolve_runtime_profile
_resolve_seed = resolve_seed
_runtime_training_prerequisite_failure = runtime_training_prerequisite_failure
_noleague_training_prerequisite_failure = noleague_training_prerequisite_failure
_spec_mismatch_policy = spec_mismatch_policy
_delete_pruned_snapshot_artifacts = delete_pruned_snapshot_artifacts
_append_checkpoint_guard_event = append_checkpoint_guard_event
_assert_noleague_baseline_config = assert_noleague_baseline_config
_best_checkpoint_record = best_checkpoint_record
_bootstrap_values = bootstrap_values
_build_env = build_training_env
_build_ids_eval_env = build_ids_eval_env
_build_learner_batch = build_learner_batch
_canonical_config_sections = canonical_config_sections
_central_runtime_actor_torch_threads = central_runtime_actor_torch_threads
_checkpoint_guard_log_path = checkpoint_guard_log_path
_config_marks_noleague_baseline = config_marks_noleague_baseline
_configure_torch_threads = configure_torch_threads
_demote_registry_champions_newer_than = demote_registry_champions_newer_than
_env_pool_config = env_pool_config
_evaluation_pinning = evaluation_pinning
_extract_structured_guard_b2_anchor_score = extract_structured_guard_b2_anchor_score
_is_noleague_baseline_role = is_noleague_baseline_role
_legacy_noleague_baseline_mode = legacy_noleague_baseline_mode
_load_checkpoint_tracker = load_checkpoint_tracker
_load_dev_eval_summaries = load_dev_eval_summaries
_load_json_object = load_json_object
_load_snapshot_registry = load_snapshot_registry
_manifest_source_path = manifest_source_path
_maybe_log_structured_mainmove_guard = maybe_log_structured_mainmove_guard
_policy_set_selection = policy_set_selection
_read_optional_hash_file = read_optional_hash_file
_relative_path_text = relative_path_text
_resolve_policy_set_selection = resolve_policy_set_selection
_resolve_resume_checkpoint_path = resolve_resume_checkpoint_path
_role_from_config_canonical = role_from_config_canonical
_run_artifacts_from_existing_run_dir = run_artifacts_from_existing_run_dir
_save_snapshot_registry_with_retention = save_snapshot_registry_with_retention
_selection_requires_dev_eval_summaries = selection_requires_dev_eval_summaries
_selection_requires_snapshot_registry = selection_requires_snapshot_registry
_sha256_file = sha256_file
_snapshot_artifact_dir_for_prune = snapshot_artifact_dir_for_prune
_spec_dimensions = spec_dimensions
_start_nonce = start_nonce
_sync_snapshot_registry_retention = sync_snapshot_registry_retention
_torch_num_threads_scope = torch_num_threads_scope
_training_paths = training_paths
_validate_imported_snapshot_contract = validate_imported_snapshot_contract
_write_checkpoint_tracker = write_checkpoint_tracker
_write_json_file = write_json_file
_write_scalars_record = write_scalars_record
_write_snapshot_artifact = write_snapshot_artifact
_manifest_actor_device_layout = manifest_actor_device_layout
_hardware_summary = hardware_summary
_entropy_coef_for_next_update = entropy_coef_for_next_update
_teacher_public_heuristic_coef_for_next_update = teacher_public_heuristic_coef_for_next_update
_public_heuristic_logit_bias_scale_for_next_update = public_heuristic_logit_bias_scale_for_next_update
_apply_guidance_schedule_for_next_update = apply_guidance_schedule_for_next_update
_model_guidance_payload = model_guidance_payload
_restore_model_guidance_from_payload = restore_model_guidance_from_payload
_maybe_compile_learner_model = maybe_compile_learner_model
_validate_algorithm_model_contract = validate_algorithm_model_contract
_json_relative_path = json_relative_path
_write_json = write_json
_slug_policy_id = slug_policy_id
_promotion_anchor_policy_id_candidates = promotion_anchor_policy_id_candidates
_resolve_symbolic_promotion_anchor_policy_id = resolve_symbolic_promotion_anchor_policy_id
_find_noleague_baseline_snapshot = find_noleague_baseline_snapshot
_resolve_promotion_anchor_policy_ids = resolve_promotion_anchor_policy_ids
_snapshot_meta_by_policy_id = snapshot_meta_by_policy_id
_evaluation_config_or_raise = evaluation_config_or_raise
_validate_periodic_dev_eval_contract = validate_periodic_dev_eval_contract
_resolve_repo_path = resolve_repo_path
_resolve_periodic_dev_eval_seed_file = resolve_periodic_dev_eval_seed_file
_periodic_dev_eval_schedule = periodic_dev_eval_schedule
_legal_ids_for_env_row = legal_ids_for_env_row
_periodic_dev_eval_rng_seed = periodic_dev_eval_rng_seed
_promotion_gate_rng_seed = promotion_gate_rng_seed
_periodic_dev_eval_bootstrap_seed = periodic_dev_eval_bootstrap_seed
_promotion_gate_bootstrap_seed = promotion_gate_bootstrap_seed
_clone_cpu_eval_model = clone_cpu_eval_model
_current_focal_policy_id = current_focal_policy_id
_checkpoint_path_for_update = checkpoint_path_for_update
_should_run_periodic_dev_eval = should_run_periodic_dev_eval
_periodic_dev_eval_summaries_path = periodic_dev_eval_summaries_path
_stall_monitor_state_path = stall_monitor_state_path
_persist_periodic_dev_eval_summary = persist_periodic_dev_eval_summary


class _PeriodicDevEvalRunner(PeriodicDevEvalRunner):
    def __init__(
        self,
        *,
        stack: StackConfig,
        model: PolicyValueModel,
        opponent_policy_id: str,
        observation_dim: int,
        action_dim: int,
        pass_action_id: int,
        artifact_dir: Path,
        focal_policy_id: str,
        require_sorted_legal_ids: bool,
        opponent_model: PolicyValueModel | None = None,
        heuristic_policy: HeuristicPublicPolicy | None = None,
    ) -> None:
        super().__init__(
            stack=stack,
            model=model,
            opponent_policy_id=opponent_policy_id,
            observation_dim=observation_dim,
            action_dim=action_dim,
            pass_action_id=pass_action_id,
            artifact_dir=artifact_dir,
            focal_policy_id=focal_policy_id,
            require_sorted_legal_ids=require_sorted_legal_ids,
            build_eval_env=_build_ids_eval_env,
            opponent_model=opponent_model,
            heuristic_policy=heuristic_policy,
        )


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
    guidance_payload = _model_guidance_payload(model)
    return persist_snapshot_registry_entry(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        model_state_dict=model_state_dict,
        config_hash256=config_hash256,
        device=device,
        update=update,
        policy_version=policy_version,
        public_heuristic_logit_bias_scale=guidance_payload.get("public_heuristic_logit_bias_scale"),
        public_heuristic_actor_logit_bias_scale=guidance_payload.get("public_heuristic_actor_logit_bias_scale"),
    )


def _repo_root() -> Path:
    return repo_root(Path(__file__))


def _git_output(args: list[str]) -> str:
    return git_output(args, cwd=_repo_root())


def _git_commit() -> str:
    return git_commit(cwd=_repo_root())


def _git_dirty() -> bool:
    return git_dirty(cwd=_repo_root())


def _experiment_role(stack: StackConfig) -> str:
    experiment = stack.config.experiment
    return "" if experiment is None else str(experiment.role).strip()


def _write_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    return write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        device=device,
        config_hash256=compute_config_hash256(stack),
        spec_hash256=spec_hash256,
        algorithm=algorithm,
        recurrent_core=getattr(stack.config.model, "recurrent_core", None),
        guidance_payload=_model_guidance_payload(learner.model),
    )


def _build_checkpoint_record(
    *,
    alias_name: str,
    alias_path: Path,
    source_checkpoint_path: Path,
    artifacts: RunArtifacts,
    learner: ImpalaLearner,
    metric_kind: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    return build_checkpoint_record(
        alias_name=alias_name,
        alias_path=alias_path,
        source_checkpoint_path=source_checkpoint_path,
        run_dir=artifacts.run_dir,
        learner=learner,
        metric_kind=metric_kind,
        metric_value=metric_value,
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
    return publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        run_dir=artifacts.run_dir,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=dev_eval_summary,
    )


def _restore_learner_from_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
    restore_counters: bool = True,
) -> ResumeCheckpoint:
    expected_config_hash = compute_config_hash256(stack)
    allow_config_mismatch = os.environ.get("WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH", "").strip() == "1"
    return restore_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        device=device,
        expected_config_hash=expected_config_hash,
        expected_spec_hash256=expected_spec_hash256,
        algorithm=algorithm,
        restore_model_guidance=restore_model_guidance_from_payload,
        allow_config_mismatch=allow_config_mismatch,
        restore_counters=restore_counters,
    )


def _initialize_learner_from_checkpoint(
    *,
    checkpoint_path: Path,
    learner: ImpalaLearner,
    device: torch.device,
    expected_spec_hash256: str,
    algorithm: str,
) -> ResumeCheckpoint:
    return initialize_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        device=device,
        expected_spec_hash256=expected_spec_hash256,
        algorithm=algorithm,
        restore_model_guidance=restore_model_guidance_from_payload,
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
) -> ImpalaLearner | PpoLiteLearner:
    return build_training_learner(
        algorithm=algorithm,
        model=model,
        compiled_model=compiled_model,
        training_config=training_config,
        training_paths=training_paths,
        pass_action_id=pass_action_id,
        checkpoint_interval_updates=checkpoint_interval_updates,
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
    return run_structured_warmstart(
        learner=learner,
        runtime=runtime,
        algorithm=algorithm,
        training_config=training_config,
        rewards_config=rewards_config,
        training_paths=training_paths,
        tensorboard_logger=tensorboard_logger,
        start_time=start_time,
        profile_timers=profile_timers,
        actor_torch_threads=actor_torch_threads,
        learner_torch_threads=learner_torch_threads,
    )


def _build_heuristic_public_policy(
    spec_bundle: Mapping[str, object],
    *,
    scoring_profile: str,
) -> HeuristicPublicPolicy:
    return build_heuristic_public_policy(
        spec_bundle,
        scoring_profile=scoring_profile,
        policy_cls=HeuristicPublicPolicy,
    )


def _import_noleague_baseline_anchor(
    *,
    training_paths: TrainingPaths,
    run_dir: Path,
    baseline_run_dir: Path,
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> tuple[Path, str, int]:
    source_run_dir = Path(baseline_run_dir).resolve()
    source_snapshot = _find_noleague_baseline_snapshot(source_run_dir)
    if source_snapshot is None:
        raise FileNotFoundError(
            "Could not resolve the canonical B1 no-league baseline snapshot in "
            f"{source_run_dir}. Run a dedicated baseline_noleague training job first."
        )

    source_weights_path = source_run_dir / source_snapshot.path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"Resolved B1 baseline snapshot is missing its weights artifact: {source_weights_path}")

    payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload must be a dict: {source_weights_path}")
    _validate_imported_snapshot_contract(
        source_run_dir=source_run_dir,
        source_policy_id=source_snapshot.policy_id,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )
    weights_path, weights_sha256 = write_imported_snapshot_artifact(
        snapshots_dir=training_paths.snapshots_dir,
        run_dir=run_dir,
        source_payload=payload,
        source_run_dir=source_run_dir,
        source_policy_id=source_snapshot.policy_id,
        source_snapshot_path=source_snapshot.path,
        target_policy_id=_PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=int(source_snapshot.update),
        metadata_format="imported_train_snapshot_metadata_v1",
    )
    return weights_path, weights_sha256, int(source_snapshot.update)


def _validate_seed_snapshot_import_contract(
    *,
    source_run_dir: Path,
    payload: dict[str, Any],
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> None:
    validate_seed_snapshot_import_contract(
        source_run_dir=source_run_dir,
        payload=payload,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
    )


def _seed_snapshot_policy_id(*, source_run_dir: Path, source_policy_id: str) -> str:
    return seed_snapshot_policy_id(source_run_dir=source_run_dir, source_policy_id=source_policy_id)


def _import_seed_snapshot_pool(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    run_dir: Path,
    seed_snapshot_run_dir: Path,
    expected_model_state_dict: dict[str, Any],
    expected_config_canonical: dict[str, Any] | None,
    expected_spec_hash256: str | None,
) -> list[str]:
    return import_seed_snapshot_pool(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        expected_model_state_dict=expected_model_state_dict,
        expected_config_canonical=expected_config_canonical,
        expected_spec_hash256=expected_spec_hash256,
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
    return ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        learner=learner,
        device=device,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        baseline_run_dir=baseline_run_dir,
        permit_current_run_alias=permit_current_run_alias,
        source_checkpoint_path=source_checkpoint_path,
        update=update,
        write_checkpoint_fn=_write_checkpoint,
        import_noleague_baseline_anchor_fn=_import_noleague_baseline_anchor,
        model_guidance_payload_fn=_model_guidance_payload,
        write_snapshot_artifact_fn=_write_snapshot_artifact,
        experiment_role_fn=_experiment_role,
    )


def _load_snapshot_eval_model(
    *,
    run_dir: Path,
    snapshot_path: str,
    observation_dim: int,
    action_dim: int,
    stack: StackConfig,
    observation_spec: dict[str, Any] | None = None,
    spec_bundle: dict[str, Any] | None = None,
) -> PolicyValueModel:
    return load_snapshot_eval_model(
        run_dir=run_dir,
        snapshot_path=snapshot_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        stack=stack,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    )


class _PromotionGateRunner(PromotionGateRunner):
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
    ) -> None:
        super().__init__(
            stack=stack,
            focal_policy_id=focal_policy_id,
            focal_model=focal_model,
            anchor_models=anchor_models,
            heuristic_policies=heuristic_policies,
            observation_dim=observation_dim,
            action_dim=action_dim,
            pass_action_id=pass_action_id,
            artifact_dir=artifact_dir,
            require_sorted_legal_ids=require_sorted_legal_ids,
            build_eval_env=_build_ids_eval_env,
            random_legal_policy_id=_PROMOTION_GATE_RANDOMLEGAL_POLICY_ID,
        )


def _ensure_current_checkpoint(
    *,
    training_paths: TrainingPaths,
    learner: ImpalaLearner,
    stack: StackConfig,
    device: torch.device,
    spec_hash256: str | None = None,
    algorithm: str | None = None,
) -> Path:
    return ensure_current_checkpoint_with_script_hooks(sys.modules[__name__], **locals())


def _periodic_dev_eval_opponents(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    run_dir: Path,
    observation_dim: int,
    action_dim: int,
) -> list[tuple[str, str, PolicyValueModel | None, HeuristicPublicPolicy | None]]:
    return periodic_dev_eval_opponents_with_script_hooks(sys.modules[__name__], **locals())


def _update_stall_monitor(
    *,
    stack: StackConfig,
    training_paths: TrainingPaths,
    update_count: int,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    return update_stall_monitor_with_script_hooks(sys.modules[__name__], **locals())


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
    return maybe_rollback_to_best_checkpoint_with_script_hooks(sys.modules[__name__], **locals())


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
    return maybe_finalize_from_best_checkpoint_with_script_hooks(sys.modules[__name__], **locals())


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
) -> dict[str, Any]:
    return run_periodic_dev_eval_with_script_hooks(sys.modules[__name__], **locals())


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
    policy_version: int,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
) -> bool | None:
    return run_snapshot_promotion_gate_with_script_hooks(sys.modules[__name__], **locals())


def _run_minimal_training(
    *,
    stack: StackConfig,
    contract: SimulatorContract,
    artifacts: Any,
    num_envs: int,
    unroll_length: int,
    max_updates: int,
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
    profile_timers: bool = False,
    torch_profiler: bool = False,
    resume_checkpoint_path: Path | None = None,
    init_from_checkpoint_path: Path | None = None,
    init_schedule_offset_override_updates: int | None = None,
    tensorboard_logger: TensorBoardLogger | None = None,
) -> dict[str, float]:
    return run_minimal_training_with_script_hooks(
        sys.modules[__name__],
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        num_envs=num_envs,
        unroll_length=unroll_length,
        max_updates=max_updates,
        profile=profile,
        device=device,
        seed=seed,
        checkpoint_interval_updates=checkpoint_interval_updates,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        runtime_mode=runtime_mode,
        b1_baseline_run_dir=b1_baseline_run_dir,
        seed_snapshot_run_dir=seed_snapshot_run_dir,
        profile_timers=profile_timers,
        torch_profiler=torch_profiler,
        resume_checkpoint_path=resume_checkpoint_path,
        init_from_checkpoint_path=init_from_checkpoint_path,
        init_schedule_offset_override_updates=init_schedule_offset_override_updates,
        tensorboard_logger=tensorboard_logger,
    )


def main() -> None:
    run_train_main(sys.modules[__name__])


if __name__ == "__main__":
    main()
