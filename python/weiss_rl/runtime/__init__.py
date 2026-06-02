"""Queue-based single-node runtime for deterministic and throughput-aware training."""

from __future__ import annotations

import copy
import os
import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import StackConfig
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.envs.decision_env import DecisionBoundaryEnv
from weiss_rl.eval.heuristic_public import HeuristicPublicPolicy
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_POLICY_ID,
    heuristic_public_policy_ids,
    heuristic_public_profile_name_for_policy_id,
)
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.league.opponent_pool import OpponentPoolSampler
from weiss_rl.league.outcomes import OnlineOutcomeTracker
from weiss_rl.league.registry import REGISTRY_FILENAME
from weiss_rl.model import PolicyValueModel
from weiss_rl.models.loading import load_snapshot_model_from_path
from weiss_rl.runtime.components import shared as runtime_shared
from weiss_rl.runtime.components.actor_state import (
    _ActorState,
    build_runtime_env,
)
from weiss_rl.runtime.components.actor_state import (
    actor_seed as runtime_actor_seed,
)
from weiss_rl.runtime.components.actor_state import (
    build_actor_state as build_runtime_actor_state,
)
from weiss_rl.runtime.components.actor_unroll import QueueRuntimeActorUnrollMixin
from weiss_rl.runtime.components.batch_collection import collect_pending_runtime_batch
from weiss_rl.runtime.components.batching import (
    concat_batch_major_field,
    concat_optional_time_major_field,
    concat_time_major_field,
    gae_advantages,
)
from weiss_rl.runtime.components.central_collection import QueueRuntimeCentralCollectionMixin
from weiss_rl.runtime.components.central_rows import QueueRuntimeCentralRowsMixin
from weiss_rl.runtime.components.config import QueueRuntimeConfig
from weiss_rl.runtime.components.config import build_runtime_config as build_runtime_config
from weiss_rl.runtime.components.counters import (
    accumulate_timeout_counters,
    collector_counter_template,
    merge_simulator_timing_counters,
    optional_int,
    packed_legal_views_from_step_out,
    timeout_limits_for_env,
)
from weiss_rl.runtime.components.devices import (
    available_cuda_device_names,
    configured_learner_device_name,
    is_cuda_auto_request,
    normalize_device_name,
)
from weiss_rl.runtime.components.devices import (
    resolve_actor_device_layout as resolve_runtime_actor_device_layout,
)
from weiss_rl.runtime.components.hashing import hash_state_dict, hash_unroll
from weiss_rl.runtime.components.heuristic_actor_rows import QueueRuntimeHeuristicActorRowsMixin
from weiss_rl.runtime.components.heuristic_public_actions import QueueRuntimeHeuristicPublicActionsMixin
from weiss_rl.runtime.components.heuristic_rollouts import QueueRuntimeHeuristicRolloutMixin
from weiss_rl.runtime.components.ipc_shared import shared_transport as runtime_shared_transport
from weiss_rl.runtime.components.ipc_shared.collector_commands import handle_collector_commands
from weiss_rl.runtime.components.ipc_shared.ipc import deserialize_state_dict_from_ipc, serialize_state_dict_for_ipc
from weiss_rl.runtime.components.ipc_shared.logging import PerformanceLogger, process_debug_log
from weiss_rl.runtime.components.ipc_shared.threads import configure_runtime_actor_torch_threads
from weiss_rl.runtime.components.legal_batching import (
    concatenate_batch_legal_actions,
    concatenate_legal_actions,
    infer_packed_meta_width,
    optional_legal_action_meta,
    require_ids_offsets,
    require_mask,
    slice_packed_rows,
    slice_packed_rows_with_meta,
    structured_legal_batch_from_mask,
    structured_legal_batch_from_packed,
)
from weiss_rl.runtime.components.legal_meta import (
    action_catalog_indices,
)
from weiss_rl.runtime.components.lifecycle import QueueRuntimeLifecycleMixin
from weiss_rl.runtime.components.opponent_mixin import QueueRuntimeOpponentMixin
from weiss_rl.runtime.components.opponent_rows import QueueRuntimeOpponentRowsMixin
from weiss_rl.runtime.components.opponents.central_opponents import QueueRuntimeCentralOpponentMixin
from weiss_rl.runtime.components.opponents.episode_roles import QueueRuntimeEpisodeRolesMixin
from weiss_rl.runtime.components.pending_mixin import QueueRuntimePendingMixin
from weiss_rl.runtime.components.policy_ids import FIXED_OPPONENT_EXCLUSIONS, MIRROR_OPPONENT_POLICY_ID
from weiss_rl.runtime.components.policy_inference.actor_models import (
    actor_inference_model,
    maybe_compile_runtime_actor_model,
)
from weiss_rl.runtime.components.policy_outputs import QueueRuntimePolicyOutputMixin
from weiss_rl.runtime.components.policy_rows import QueueRuntimePolicyRowsMixin
from weiss_rl.runtime.components.process import collector_process_main, start_process_collectors
from weiss_rl.runtime.components.structured_warmstart import (
    restore_process_collector_fixed_opponents,
    set_process_collector_fixed_opponents,
)
from weiss_rl.runtime.components.structured_warmstart import (
    structured_warmstart_source_mix as runtime_structured_warmstart_source_mix,
)
from weiss_rl.runtime.components.support import QueueRuntimeSupportMixin
from weiss_rl.runtime.components.teacher_heuristic_mixin import QueueRuntimeTeacherHeuristicMixin
from weiss_rl.runtime.components.topology import QueueRuntimeMode, resolve_actor_topology
from weiss_rl.runtime.components.types import PendingUnroll, RuntimeBatch, RuntimeUnroll

__all__ = [
    "QueueRuntime",
    "QueueRuntimeMode",
    "build_runtime_config",
    "resolve_actor_device_layout",
]

_MIRROR_OPPONENT_POLICY_ID = MIRROR_OPPONENT_POLICY_ID
_NOLEAGUE_BASELINE_POLICY_ID = NOLEAGUE_BASELINE_POLICY_ID
_FIXED_OPPONENT_EXCLUSIONS = FIXED_OPPONENT_EXCLUSIONS
_PFSP_TIMEOUT_FILTER_MIN_SAMPLES = 32
_PROMOTION_GATED_RECENT_RESERVOIR_MIN_SIZE = 2
_PFSP_DIVERSITY_FLOOR_SIZE = 2
_DEFAULT_ACTION_META_WIDTH = runtime_shared.DEFAULT_ACTION_META_WIDTH
_SharedCollectorSlot = runtime_shared.SharedCollectorSlot
_SharedPendingUnroll = runtime_shared.SharedPendingUnroll
_HEURISTIC_PUBLIC_VARIANT_POLICY_IDS = heuristic_public_policy_ids(include_base=False)
_obs_numpy_dtype_for_profile = runtime_shared_transport.obs_numpy_dtype_for_profile
_shared_segment_spec = runtime_shared_transport.shared_segment_spec
_create_shared_collector_slot_config = runtime_shared_transport.create_shared_collector_slot_config
_open_shared_collector_slot = runtime_shared_transport.open_shared_collector_slot
_shared_unroll_metadata = runtime_shared_transport.shared_unroll_metadata
_write_unroll_to_shared_slot = runtime_shared_transport.write_unroll_to_shared_slot
_read_unroll_from_shared_slot = runtime_shared_transport.read_unroll_from_shared_slot
_serialize_state_dict_for_ipc = serialize_state_dict_for_ipc
_deserialize_state_dict_from_ipc = deserialize_state_dict_from_ipc
_collector_counter_template = collector_counter_template
_timeout_limits_for_env = timeout_limits_for_env
_optional_int = optional_int
_merge_simulator_timing_counters = merge_simulator_timing_counters
_accumulate_timeout_counters = accumulate_timeout_counters
_packed_legal_views_from_step_out = packed_legal_views_from_step_out
_process_debug_log = process_debug_log
_handle_collector_commands = handle_collector_commands
_is_cuda_auto_request = is_cuda_auto_request
_available_cuda_device_names = available_cuda_device_names
_normalize_device_name = normalize_device_name
_configured_learner_device_name = configured_learner_device_name
resolve_actor_device_layout = resolve_runtime_actor_device_layout


def _configure_runtime_actor_torch_threads(actor_torch_threads: int) -> None:
    configure_runtime_actor_torch_threads(actor_torch_threads)


def _resolve_runtime_actor_device(
    stack: StackConfig,
    *,
    learner_device: torch.device | str | None = None,
) -> torch.device:
    system = stack.config.system
    requested = "cpu" if system is None else str(system.actor_device).strip()
    prefer_process_collectors = "," in requested or _is_cuda_auto_request(requested)
    resolved = resolve_actor_device_layout(
        stack,
        actor_count=1,
        learner_device=learner_device,
        prefer_process_collectors=prefer_process_collectors,
    )[0]
    return torch.device(resolved)


def _maybe_compile_runtime_actor_model(model: PolicyValueModel, *, enabled: bool) -> Any | None:
    return maybe_compile_runtime_actor_model(model, enabled=enabled)


def _actor_inference_model(actor: _ActorState) -> Any:
    return actor_inference_model(actor)


def _collector_process_main(
    *,
    stack: StackConfig,
    config: QueueRuntimeConfig,
    model_state_dict: dict[str, Any],
    observation_dim: int,
    action_dim: int,
    observation_spec: dict[str, Any] | None,
    spec_bundle: dict[str, Any] | None,
    run_dir: str | None,
    actor_id: int,
    actor_device_name: str | None,
    learner_device_name: str | None,
    control_queue: Any,
    free_queue: Any | None,
    result_queue: Any,
    shared_slot_configs: list[dict[str, Any]] | None,
) -> None:
    collector_process_main(
        runtime_cls=QueueRuntime,
        stack=stack,
        config=config,
        model_state_dict=model_state_dict,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
        run_dir=run_dir,
        actor_id=actor_id,
        actor_device_name=actor_device_name,
        learner_device_name=learner_device_name,
        control_queue=control_queue,
        free_queue=free_queue,
        result_queue=result_queue,
        shared_slot_configs=shared_slot_configs,
    )


class QueueRuntime(
    QueueRuntimePendingMixin,
    QueueRuntimeOpponentMixin,
    QueueRuntimeOpponentRowsMixin,
    QueueRuntimePolicyOutputMixin,
    QueueRuntimeCentralRowsMixin,
    QueueRuntimeCentralOpponentMixin,
    QueueRuntimeCentralCollectionMixin,
    QueueRuntimeEpisodeRolesMixin,
    QueueRuntimeHeuristicActorRowsMixin,
    QueueRuntimeHeuristicPublicActionsMixin,
    QueueRuntimeHeuristicRolloutMixin,
    QueueRuntimeActorUnrollMixin,
    QueueRuntimeLifecycleMixin,
    QueueRuntimeTeacherHeuristicMixin,
    QueueRuntimePolicyRowsMixin,
    QueueRuntimeSupportMixin,
):
    """Single-node actor queue runtime with deterministic ordered mode."""

    def __init__(
        self,
        *,
        stack: StackConfig,
        config: QueueRuntimeConfig,
        model: Any,
        observation_dim: int,
        action_dim: int,
        observation_spec: dict[str, Any] | None = None,
        spec_bundle: dict[str, Any] | None = None,
        run_dir: Path | None = None,
        performance_log_path: Path | None = None,
        defer_initial_opponent_pool_refresh: bool = False,
        learner_device: torch.device | str | None = None,
    ) -> None:
        if config.actor_count < 1:
            raise ValueError("actor_count must be >= 1")
        if config.envs_per_actor < 1:
            raise ValueError("envs_per_actor must be >= 1")
        if config.batch_unrolls_per_update < 1:
            raise ValueError("batch_unrolls_per_update must be >= 1")
        if config.queue_capacity_unrolls < config.batch_unrolls_per_update:
            raise ValueError("queue_capacity_unrolls must be >= batch_unrolls_per_update")
        if config.mode == "train_ordered" and config.batch_unrolls_per_update < config.actor_count:
            raise ValueError("train_ordered requires batch_unrolls_per_update >= actor_count")

        self.stack = stack
        self.config = config
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self._observation_spec = None if observation_spec is None else dict(observation_spec)
        self._spec_bundle = None if spec_bundle is None else dict(spec_bundle)
        action_meta_spec = {} if self._spec_bundle is None else dict(self._spec_bundle.get("action_meta_v1", {}))
        self._action_meta_width = int(action_meta_spec.get("width", _DEFAULT_ACTION_META_WIDTH))
        system_config = stack.config.system
        self._learner_device = torch.device(_configured_learner_device_name(stack, learner_device=learner_device))
        self._requested_actor_device = (
            "cpu" if system_config is None else str(getattr(system_config, "actor_device", "cpu")).strip()
        )
        self._process_actor_device_names = resolve_actor_device_layout(
            stack,
            actor_count=int(config.actor_count),
            learner_device=self._learner_device,
            prefer_process_collectors=True,
        )
        self._device = _resolve_runtime_actor_device(stack, learner_device=self._learner_device)
        self._run_dir = None if run_dir is None else Path(run_dir)
        self._artifact_layout = None if self._run_dir is None else ArtifactLayout.from_run_dir(self._run_dir)
        training_config = stack.config.training
        experiment_config = stack.config.experiment
        self._experiment_role = "" if experiment_config is None else str(experiment_config.role).strip()
        self._actor_amp_enabled = bool(
            training_config is not None and bool(training_config.mixed_precision) and self._device.type == "cuda"
        )
        self._compile_actor_inference = bool(
            training_config is not None
            and bool(getattr(training_config, "compile_actor_inference", False))
            and self._device.type == "cpu"
        )
        self._league_config = stack.config.league
        self._league_enabled = bool(
            self._artifact_layout is not None
            and self._league_config is not None
            and self._league_config.enabled
            and self._experiment_role != "baseline_noleague"
        )
        self._registry_path = (
            None if self._artifact_layout is None else self._artifact_layout.training_snapshots_dir / REGISTRY_FILENAME
        )
        self._opponent_models: dict[str, PolicyValueModel] = {}
        self._opponent_model_locks: dict[str, threading.Lock] = {}
        self._opponent_heuristic_policies: dict[str, HeuristicPublicPolicy] = {}
        self._teacher_guidance_enabled = bool(
            training_config is not None and bool(getattr(training_config, "structured_aux_enabled", False))
        )
        self._teacher_aux_mode = (
            "always"
            if training_config is None
            else str(getattr(training_config, "teacher_aux_mode", "always")).strip().lower()
        )
        self._teacher_label_profiles = (
            ("base",)
            if training_config is None
            else tuple(getattr(training_config, "teacher_public_heuristic_profiles", ())) or ("base",)
        )
        self._teacher_label_profile_mode = (
            "mixture"
            if training_config is None
            else str(getattr(training_config, "teacher_public_heuristic_profile_mode", "mixture")).strip().lower()
        )
        self._teacher_label_profiles_end_updates = (
            -1
            if training_config is None
            else int(getattr(training_config, "teacher_public_heuristic_profiles_end_updates", -1))
        )
        structured_warmstart_cfg = (
            None if training_config is None else getattr(training_config, "structured_warmstart", None)
        )
        self._teacher_guidance_warmstart_updates = 0
        if structured_warmstart_cfg is not None and bool(getattr(structured_warmstart_cfg, "enabled", False)):
            self._teacher_guidance_warmstart_updates = max(0, int(getattr(structured_warmstart_cfg, "updates", 0)))
        self._actor_policy_backend = (
            "model"
            if training_config is None
            else str(getattr(training_config, "actor_policy_backend", "model")).strip().lower()
        )
        if self._actor_policy_backend not in {"model", "heuristic_public"}:
            raise ValueError("training.actor_policy_backend must be one of: model, heuristic_public")
        self._actor_heuristic_fraction = (
            1.0 if training_config is None else float(getattr(training_config, "actor_heuristic_fraction", 1.0))
        )
        if self._actor_heuristic_fraction < 0.0 or self._actor_heuristic_fraction > 1.0:
            raise ValueError("training.actor_heuristic_fraction must be between 0.0 and 1.0 inclusive")
        self._actor_heuristic_start_updates = (
            0 if training_config is None else int(getattr(training_config, "actor_heuristic_start_updates", 0))
        )
        if self._actor_heuristic_start_updates < 0:
            raise ValueError("training.actor_heuristic_start_updates must be >= 0")
        self._actor_heuristic_end_updates = (
            -1 if training_config is None else int(getattr(training_config, "actor_heuristic_end_updates", -1))
        )
        if self._actor_heuristic_end_updates < -1:
            raise ValueError("training.actor_heuristic_end_updates must be >= -1")
        self._actor_heuristic_final_fraction = (
            self._actor_heuristic_fraction
            if training_config is None
            else float(getattr(training_config, "actor_heuristic_final_fraction", self._actor_heuristic_fraction))
        )
        if self._actor_heuristic_final_fraction < 0.0 or self._actor_heuristic_final_fraction > 1.0:
            raise ValueError("training.actor_heuristic_final_fraction must be between 0.0 and 1.0 inclusive")
        if (
            self._actor_heuristic_end_updates >= 0
            and self._actor_heuristic_end_updates < self._actor_heuristic_start_updates
        ):
            raise ValueError("training.actor_heuristic_end_updates must be >= training.actor_heuristic_start_updates")
        self._train_on_heuristic_actor_rows = (
            True if training_config is None else bool(getattr(training_config, "train_on_heuristic_actor_rows", True))
        )
        requested_diverse_actor_count = (
            0 if training_config is None else int(getattr(training_config, "diverse_opponent_actor_count", 0))
        )
        if requested_diverse_actor_count < 0:
            raise ValueError("training.diverse_opponent_actor_count must be >= 0")
        self._diverse_opponent_actor_count = min(int(self.config.actor_count), requested_diverse_actor_count)
        requested_diverse_model_actor_count = (
            0 if training_config is None else int(getattr(training_config, "diverse_model_actor_count", 0))
        )
        if requested_diverse_model_actor_count < 0:
            raise ValueError("training.diverse_model_actor_count must be >= 0")
        self._diverse_model_actor_count = min(
            int(self._diverse_opponent_actor_count), requested_diverse_model_actor_count
        )
        self._diverse_opponent_batch_fraction = (
            0.0 if training_config is None else float(getattr(training_config, "diverse_opponent_batch_fraction", 0.0))
        )
        if self._diverse_opponent_batch_fraction < 0.0 or self._diverse_opponent_batch_fraction > 1.0:
            raise ValueError("training.diverse_opponent_batch_fraction must be between 0.0 and 1.0 inclusive")
        self._diverse_opponent_batch_wait_ms = (
            0 if training_config is None else int(getattr(training_config, "diverse_opponent_batch_wait_ms", 0))
        )
        if self._diverse_opponent_batch_wait_ms < 0:
            raise ValueError("training.diverse_opponent_batch_wait_ms must be >= 0")
        self._heuristic_actor_hidden_state_tracking = (
            True
            if training_config is None
            else bool(getattr(training_config, "heuristic_actor_hidden_state_tracking", True))
        )
        self._trajectory_retention_enabled = bool(
            training_config is not None and float(getattr(training_config, "trajectory_retention_coef", 0.0)) > 0.0
        )
        self._trajectory_retention_policy_ids = (
            ()
            if training_config is None
            else tuple(
                str(policy_id).strip()
                for policy_id in getattr(training_config, "trajectory_retention_policy_ids", ())
                if str(policy_id).strip()
            )
        )
        self._trajectory_retention_sources = (
            ()
            if training_config is None
            else tuple(
                str(source).strip().lower()
                for source in getattr(training_config, "trajectory_retention_sources", ())
                if str(source).strip()
            )
        )
        algorithm_name = (
            "" if training_config is None else str(getattr(training_config, "algorithm", "")).strip().lower()
        )
        self._actor_behavior_values_required = "ppo" in algorithm_name
        self._action_catalog: ActionCatalog | None = None
        self._action_family_index: dict[str, int] = {}
        self._action_attack_type_index: dict[str, int] = {}
        if self._spec_bundle is not None:
            with suppress(Exception):
                self._action_catalog = ActionCatalog.from_spec_bundle(self._spec_bundle)
        if self._action_catalog is not None:
            self._action_family_index, self._action_attack_type_index = action_catalog_indices(self._action_catalog)
        self._last_action_arg0_obs_index = -1
        if self._spec_bundle is not None:
            observation_spec = self._spec_bundle.get("observation", {})
            if isinstance(observation_spec, dict):
                for field in observation_spec.get("header_fields", []):
                    if isinstance(field, dict) and field.get("name") == "last_action_arg0":
                        self._last_action_arg0_obs_index = int(field.get("index", -1))
                        break
        self._teacher_policy: HeuristicPublicPolicy | None = None
        self._teacher_policy_by_profile: dict[str, HeuristicPublicPolicy] = {}
        self._teacher_action_catalog: ActionCatalog | None = None
        self._teacher_family_index: dict[str, int] = {}
        self._teacher_attack_type_index: dict[str, int] = {}
        if self._teacher_guidance_enabled:
            if self._spec_bundle is None:
                raise RuntimeError("structured_aux.enabled requires the runtime spec bundle")
            try:
                self._teacher_policy = HeuristicPublicPolicy.from_spec_bundle(self._spec_bundle)
                self._teacher_policy_by_profile["base"] = self._teacher_policy
                for profile_name in self._teacher_label_profiles:
                    normalized_profile = str(profile_name).strip().lower()
                    if not normalized_profile or normalized_profile == "base":
                        continue
                    self._teacher_policy_by_profile[normalized_profile] = HeuristicPublicPolicy.from_spec_bundle(
                        self._spec_bundle,
                        scoring_profile=normalized_profile,
                    )
                self._teacher_action_catalog = self._action_catalog or ActionCatalog.from_spec_bundle(self._spec_bundle)
            except Exception as exc:
                raise RuntimeError(
                    "Structured teacher guidance requires a heuristic-compatible simulator contract"
                ) from exc
            self._teacher_family_index = {
                family.name: index for index, family in enumerate(self._teacher_action_catalog.families)
            }
            self._teacher_attack_type_index = {
                name: index for index, name in enumerate(self._teacher_action_catalog.attack_type_names)
            }
        if self._actor_policy_backend == "heuristic_public" and self._teacher_policy is None:
            if self._spec_bundle is None:
                raise RuntimeError("training.actor_policy_backend=heuristic_public requires the runtime spec bundle")
            self._teacher_policy = HeuristicPublicPolicy.from_spec_bundle(self._spec_bundle)
        self._opponent_sampler: OpponentPoolSampler | None = None
        self._opponent_candidate_ids: tuple[str, ...] = ()
        self._outcomes = OnlineOutcomeTracker(
            window_size=(50_000 if self._league_config is None else int(self._league_config.pfsp_window_episodes))
        )
        self._pfsp_epoch = int(self._outcomes.current_epoch)
        self._current_learner_update = 0
        self._effective_learner_update = 0
        self._published_snapshot_update_by_fingerprint: dict[str, int] = {}
        self._pfsp_pool_size = 0
        self._pfsp_quarantined_opponents = 0
        self._pfsp_champion_pool_size = 0
        self._pfsp_recent_pool_size = 0
        self._pfsp_hard_negative_pool_size = 0
        self._pfsp_last_sampled_envs = 0
        self._pfsp_last_mirror_envs = 0
        self._pfsp_last_heuristic_public_envs = 0
        self._pfsp_last_heuristic_public_variant_envs = 0
        self._pfsp_last_noleague_baseline_envs = 0
        self._pfsp_last_champion_envs = 0
        self._pfsp_last_recent_envs = 0
        self._pfsp_last_hard_negative_envs = 0
        self._pfsp_last_warmup_snapshot_envs = 0
        self._pfsp_last_sampled_policy_envs: dict[str, int] = {}
        self._pfsp_last_heuristic_public_policy_envs: dict[str, int] = {}
        self._pfsp_last_heuristic_public_variant_policy_envs: dict[str, int] = {}
        self._pfsp_last_noleague_baseline_policy_envs: dict[str, int] = {}
        self._pfsp_last_champion_policy_envs: dict[str, int] = {}
        self._pfsp_last_recent_policy_envs: dict[str, int] = {}
        self._pfsp_last_hard_negative_policy_envs: dict[str, int] = {}
        self._pfsp_last_warmup_snapshot_policy_envs: dict[str, int] = {}
        self._disable_mirror_policy_fusion = False
        self._opponent_champion_ids: tuple[str, ...] = ()
        self._opponent_recent_ids: tuple[str, ...] = ()
        self._opponent_hard_negative_ids: tuple[str, ...] = ()
        heuristic_public_mix_fraction = 0.0
        heuristic_public_variant_mix_fraction = 0.0
        if self._league_config is not None:
            sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
            heuristic_public_mix_fraction = float(getattr(sampling_cfg, "heuristic_public_mix_fraction", 0.0))
            heuristic_public_variant_mix_fraction = max(
                float(getattr(sampling_cfg, "heuristic_public_variant_mix_fraction", 0.0)),
                float(
                    getattr(
                        sampling_cfg,
                        "heuristic_public_variant_final_mix_fraction",
                        getattr(sampling_cfg, "heuristic_public_variant_mix_fraction", 0.0),
                    )
                ),
            )
        base_heuristic_required = bool(
            heuristic_public_mix_fraction > 0.0
            or (
                int(getattr(self, "_diverse_opponent_actor_count", 0)) > 0
                and int(getattr(self, "_diverse_opponent_actor_count", 0)) < int(self.config.actor_count)
            )
        )
        if base_heuristic_required:
            if self._spec_bundle is None:
                raise RuntimeError("heuristic-public opponent lanes require the runtime spec bundle")
            try:
                self._opponent_heuristic_policies[HEURISTIC_PUBLIC_POLICY_ID] = HeuristicPublicPolicy.from_spec_bundle(
                    self._spec_bundle
                )
            except Exception as exc:
                raise RuntimeError(
                    "Training-time B2 HeuristicPublic requires a heuristic-compatible simulator contract"
                ) from exc
        if heuristic_public_variant_mix_fraction > 0.0:
            if self._spec_bundle is None:
                raise RuntimeError(
                    "league.sampling.heuristic_public_variant_mix_fraction > 0 requires the runtime spec bundle"
                )
            try:
                for policy_id in _HEURISTIC_PUBLIC_VARIANT_POLICY_IDS:
                    profile_name = heuristic_public_profile_name_for_policy_id(policy_id)
                    if profile_name is None:
                        continue
                    self._opponent_heuristic_policies[policy_id] = HeuristicPublicPolicy.from_spec_bundle(
                        self._spec_bundle,
                        scoring_profile=profile_name,
                    )
            except Exception as exc:
                raise RuntimeError(
                    "Training-time heuristic-public variant baselines require a heuristic-compatible simulator contract"
                ) from exc
        self._heuristic_public_reserved_envs_per_actor = 0
        self._noleague_baseline_reserved_envs_per_actor = 0
        if self._league_config is not None:
            sampling_cfg = getattr(self._league_config, "sampling", self._league_config)
            self._heuristic_public_reserved_envs_per_actor = int(
                getattr(sampling_cfg, "heuristic_public_reserved_envs_per_actor", 0)
            )
            self._noleague_baseline_reserved_envs_per_actor = int(
                getattr(sampling_cfg, "noleague_baseline_reserved_envs_per_actor", 0)
            )
        if self._heuristic_public_reserved_envs_per_actor + self._noleague_baseline_reserved_envs_per_actor > int(
            config.envs_per_actor
        ):
            raise ValueError("league.sampling reserved env counts per actor cannot exceed training.envs_per_actor")
        model_kind = "" if stack.config.model is None else str(stack.config.model.encoder_kind).strip().lower()
        structured_fixed_opponents_expected = bool(
            model_kind == "structured_v2"
            and (
                bool(getattr(structured_warmstart_cfg, "enabled", False))
                or self._heuristic_public_reserved_envs_per_actor > 0
                or self._noleague_baseline_reserved_envs_per_actor > 0
            )
        )
        self._structured_fixed_opponents_expected = structured_fixed_opponents_expected
        collection_backend = (
            "auto"
            if system_config is None
            else str(getattr(system_config, "collection_backend", "auto")).strip().lower()
        )
        if collection_backend not in {"auto", "central", "process"}:
            raise ValueError("system.collection_backend must be one of: auto, central, process")
        self._collection_backend = collection_backend
        process_collectors_supported = bool(
            config.mode == "train_async_fast"
            and int(config.actor_count) > 1
            and model_kind != "typed_v1"
            and all(
                torch.device(device_name).type in {"cpu", "cuda"} for device_name in self._process_actor_device_names
            )
        )
        auto_use_process_collectors = bool(process_collectors_supported and not self._league_enabled)
        central_batched_collection_supported = bool(
            config.mode == "train_async_fast"
            and (
                (self._device.type == "cpu" and model_kind in {"typed_v1", "structured_v2"})
                or (self._device.type == "cuda" and model_kind == "structured_v2")
            )
        )
        auto_use_central_batched_collection = bool(central_batched_collection_supported)
        auto_prefers_process_collectors = bool(
            auto_use_process_collectors
            and (
                (self._actor_policy_backend == "model" and self._device.type == "cpu" and model_kind == "structured_v2")
                or (
                    (_is_cuda_auto_request(self._requested_actor_device) or "," in self._requested_actor_device)
                    and len(dict.fromkeys(self._process_actor_device_names)) > 1
                )
            )
        )
        if auto_prefers_process_collectors:
            auto_use_central_batched_collection = False
        elif auto_use_central_batched_collection:
            auto_use_process_collectors = False
        if collection_backend == "auto":
            self._use_process_collectors = auto_use_process_collectors
            self._use_central_batched_collection = auto_use_central_batched_collection
        elif collection_backend == "central":
            if not central_batched_collection_supported:
                raise ValueError("system.collection_backend=central is not supported for the current runtime setup")
            self._use_process_collectors = False
            self._use_central_batched_collection = True
        else:
            if not process_collectors_supported:
                raise ValueError("system.collection_backend=process is not supported for the current runtime setup")
            self._use_process_collectors = True
            self._use_central_batched_collection = False
        self._use_shared_collector_transport = bool(self._use_process_collectors)
        self._use_simulator_fused_logits_step = bool(
            config.mode == "train_async_fast" and str(config.profile).strip().lower() == "fast" and model_kind == "mlp"
        )
        self._process_context: Any | None = None
        self._collector_processes: list[Any] = []
        self._collector_control_queues: list[Any] = []
        self._collector_free_queues: list[Any] = []
        self._collector_result_queue: Any | None = None
        self._collector_shared_slots: dict[int, tuple[_SharedCollectorSlot, ...]] = {}
        self._shared_actor_model = None
        self._shared_compiled_actor_model = None
        fixed_opponent_backend = (
            str(getattr(stack.config.training, "fixed_opponent_backend", "python_scalar")).strip().lower()
        )
        if fixed_opponent_backend not in {"python_scalar", "python_batched", "simulator_native"}:
            raise ValueError(
                "training.fixed_opponent_backend must be one of: python_scalar, python_batched, simulator_native"
            )
        self._fixed_opponent_backend = fixed_opponent_backend
        self._profile_timers = bool(getattr(stack.config.training, "profile_timers", False))
        self._debug_validate_sampled_packed_actions = (
            os.environ.get("WEISS_DEBUG_VALIDATE_SAMPLED_PACKED_ACTIONS", "").strip() == "1"
        )
        self._batch_timer_metrics: dict[str, float] = {}
        self._bootstrap_model_devices: list[torch.device] = (
            [torch.device(device_name) for device_name in self._process_actor_device_names]
            if self._use_process_collectors
            else []
        )
        if self._use_central_batched_collection:
            self._shared_actor_model = copy.deepcopy(model).to(self._device)
            self._shared_actor_model.eval()
            self._shared_compiled_actor_model = _maybe_compile_runtime_actor_model(
                self._shared_actor_model,
                enabled=self._compile_actor_inference,
            )
        self._bootstrap_models = (
            [
                copy.deepcopy(model).to(self._bootstrap_model_devices[actor_id])
                for actor_id in range(int(config.actor_count))
            ]
            if self._use_process_collectors
            else None
        )
        self._actors = (
            []
            if self._use_process_collectors
            else [
                self._build_actor_state(model=model, actor_id=actor_id) for actor_id in range(int(config.actor_count))
            ]
        )
        self._pending_unrolls: deque[PendingUnroll] = deque()
        self._next_actor_index = 0
        self._collector_executor = (
            None
            if self._use_process_collectors or self._use_central_batched_collection or len(self._actors) <= 1
            else ThreadPoolExecutor(
                max_workers=len(self._actors),
                thread_name_prefix="weiss-runtime-actor",
            )
        )
        if self._collector_executor is not None and stack.config.system is not None:
            _configure_runtime_actor_torch_threads(int(stack.config.system.actor_torch_threads))
        self._last_published_snapshot_version = 0
        self._performance_logger = None if performance_log_path is None else PerformanceLogger(performance_log_path)
        if self._performance_logger is not None:
            rows_per_actor_unroll = int(self.config.unroll_length) * int(self.config.envs_per_actor)
            batch_env_steps = int(self.config.batch_unrolls_per_update) * rows_per_actor_unroll
            self._performance_logger.log(
                {
                    "kind": "runtime_startup_v1",
                    "actor_device": self._device.type,
                    "actor_device_layout": list(dict.fromkeys(self._process_actor_device_names))
                    if self._use_process_collectors
                    else [str(self._device)],
                    "compile_actor_inference": bool(self._compile_actor_inference),
                    "fixed_opponent_backend": self._fixed_opponent_backend,
                    "actor_policy_backend": self._actor_policy_backend,
                    "actor_heuristic_fraction": float(self._actor_heuristic_fraction),
                    "actor_sampling_temperature": float(self.config.actor_sampling_temperature),
                    "runtime_actor_count": int(self.config.actor_count),
                    "runtime_envs_per_actor": int(self.config.envs_per_actor),
                    "runtime_total_envs": int(self.config.total_envs),
                    "runtime_unroll_length": int(self.config.unroll_length),
                    "runtime_rows_per_actor_unroll": int(rows_per_actor_unroll),
                    "runtime_batch_unrolls_per_update": int(self.config.batch_unrolls_per_update),
                    "runtime_batch_env_steps": int(batch_env_steps),
                    "runtime_queue_capacity_unrolls": int(self.config.queue_capacity_unrolls),
                    "collection_backend": self._collection_backend,
                    "league_enabled": bool(self._league_enabled),
                    "model_kind": model_kind,
                    "structured_fixed_opponents_expected": bool(self._structured_fixed_opponents_expected),
                    "structured_warmstart_enabled": bool(
                        training_config is not None
                        and bool(getattr(training_config, "structured_warmstart_enabled", False))
                    ),
                    "structured_warmstart_flag_enabled": bool(
                        structured_warmstart_cfg is not None
                        and bool(getattr(structured_warmstart_cfg, "enabled", False))
                    ),
                    "use_central_batched_collection": bool(self._use_central_batched_collection),
                    "use_process_collectors": bool(self._use_process_collectors),
                }
            )
        self._runtime_start = time.time()
        self._runtime_last_metrics_time = self._runtime_start
        self._runtime_cumulative_env_steps = 0
        if self._use_process_collectors:
            self._start_process_collectors(model)
            self.refresh_opponent_pool()
        elif not bool(defer_initial_opponent_pool_refresh):
            self.refresh_opponent_pool()

    def _reset_batch_timer_metrics(self) -> None:
        self._batch_timer_metrics = {}

    def _record_batch_timer_ms(self, name: str, elapsed_seconds: float) -> None:
        if not bool(getattr(self, "_profile_timers", False)):
            return
        if not hasattr(self, "_batch_timer_metrics"):
            self._batch_timer_metrics = {}
        key = f"timer_runtime_{name}_ms"
        self._batch_timer_metrics[key] = self._batch_timer_metrics.get(key, 0.0) + (float(elapsed_seconds) * 1000.0)

    def _record_batch_counter(self, name: str, value: float) -> None:
        if not bool(getattr(self, "_profile_timers", False)):
            return
        if not hasattr(self, "_batch_timer_metrics"):
            self._batch_timer_metrics = {}
        key = f"runtime_{name}"
        self._batch_timer_metrics[key] = self._batch_timer_metrics.get(key, 0.0) + float(value)

    def _set_process_collector_fixed_opponents(
        self,
        *,
        slots: np.ndarray | None,
        forced_policy_ids: Sequence[str],
        activate_teacher_heuristic: bool,
    ) -> None:
        set_process_collector_fixed_opponents(
            self,
            slots=slots,
            forced_policy_ids=forced_policy_ids,
            activate_teacher_heuristic=activate_teacher_heuristic,
            noleague_policy_id=_NOLEAGUE_BASELINE_POLICY_ID,
        )

    def _restore_process_collector_fixed_opponents(self) -> None:
        restore_process_collector_fixed_opponents(self)

    @contextmanager
    def structured_warmstart_source_mix(self) -> Any:
        with runtime_structured_warmstart_source_mix(
            self,
            heuristic_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
            noleague_policy_id=_NOLEAGUE_BASELINE_POLICY_ID,
        ) as metrics:
            yield metrics

    @contextmanager
    def disable_mirror_policy_fusion(self) -> Any:
        previous = bool(getattr(self, "_disable_mirror_policy_fusion", False))
        self._disable_mirror_policy_fusion = True
        try:
            yield
        finally:
            self._disable_mirror_policy_fusion = previous

    def collect_update_batch(
        self,
        *,
        gamma: float,
        truncation_reward: float,
        truncation_bootstrap_value: bool,
        vtrace_rho_bar: float,
        vtrace_c_bar: float,
    ) -> RuntimeBatch:
        return collect_pending_runtime_batch(
            self,
            target_count=int(self.config.batch_unrolls_per_update),
            build_batch=lambda selected: self._build_learner_batch(
                selected,
                gamma=gamma,
                truncation_reward=truncation_reward,
                truncation_bootstrap_value=truncation_bootstrap_value,
                vtrace_rho_bar=vtrace_rho_bar,
                vtrace_c_bar=vtrace_c_bar,
            ),
            build_timer_name="build_learner_batch",
            total_timer_name="collect_update_batch_total",
        )

    def collect_policy_batch(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        truncation_reward: float,
        truncation_bootstrap_value: bool,
    ) -> RuntimeBatch:
        return collect_pending_runtime_batch(
            self,
            target_count=int(self.config.batch_unrolls_per_update),
            build_batch=lambda selected: self._build_ppo_batch(
                selected,
                gamma=gamma,
                gae_lambda=gae_lambda,
                truncation_reward=truncation_reward,
                truncation_bootstrap_value=truncation_bootstrap_value,
            ),
            build_timer_name="build_ppo_batch",
            total_timer_name="collect_policy_batch_total",
        )

    def _read_unroll_from_shared_slot(self, slot: Any, metadata: dict[str, Any]) -> RuntimeUnroll:
        return _read_unroll_from_shared_slot(slot, metadata)

    def _start_process_collectors(self, model: PolicyValueModel) -> None:
        start_process_collectors(
            runtime=self,
            model=model,
            collector_process_target=_collector_process_main,
        )

    def _build_actor_state(self, *, model: PolicyValueModel, actor_id: int) -> _ActorState:
        env, layout_name = self._build_env(seed=_actor_seed(self.config.base_seed, actor_id), actor_id=actor_id)
        return build_runtime_actor_state(
            actor_state_cls=_ActorState,
            model=model,
            actor_id=int(actor_id),
            env=env,
            layout_name=layout_name,
            base_seed=int(self.config.base_seed),
            envs_per_actor=int(self.config.envs_per_actor),
            device=self._device,
            shared_actor_model=self._shared_actor_model,
            shared_compiled_actor_model=self._shared_compiled_actor_model,
            maybe_compile_actor_model=lambda actor_model: _maybe_compile_runtime_actor_model(
                actor_model,
                enabled=bool(self._compile_actor_inference),
            ),
            legal_action_meta_from_ids=self._legal_action_meta_from_ids,
            fixed_opponent_policy_slots=self._fixed_opponent_policy_slots,
            diverse_opponent_actor_count=int(getattr(self, "_diverse_opponent_actor_count", 0)),
            diverse_model_actor_count=int(getattr(self, "_diverse_model_actor_count", 0)),
            assign_episode_roles=lambda actor, done: self._assign_episode_roles(actor, done, initial=True),
        )

    def _build_env(self, *, seed: int, actor_id: int) -> tuple[DecisionBoundaryEnv, str]:
        return build_runtime_env(
            stack=self.stack,
            profile=str(self.config.profile),
            envs_per_actor=int(self.config.envs_per_actor),
            pass_action_id=int(self.config.pass_action_id),
            seed=int(seed),
            actor_id=int(actor_id),
            profile_timers=bool(self._profile_timers),
        )

    def _load_snapshot_model(self, snapshot_path: str) -> PolicyValueModel:
        if self._run_dir is None:
            raise RuntimeError("QueueRuntime cannot load opponent snapshots without a canonical run_dir")
        return load_snapshot_model_from_path(
            run_dir=self._run_dir,
            snapshot_path=snapshot_path,
            stack=self.stack,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            observation_spec=self._observation_spec,
            spec_bundle=self._spec_bundle,
            device=self._device,
        )


_resolve_actor_topology = resolve_actor_topology
_actor_seed = runtime_actor_seed
_concat_time_major_field = concat_time_major_field
_concat_optional_time_major_field = concat_optional_time_major_field
_concat_batch_major_field = concat_batch_major_field
_concatenate_legal_actions = concatenate_legal_actions
_require_ids_offsets = require_ids_offsets
_optional_legal_action_meta = optional_legal_action_meta
_require_mask = require_mask
_concatenate_batch_legal_actions = concatenate_batch_legal_actions
_slice_packed_rows = slice_packed_rows
_slice_packed_rows_with_meta = slice_packed_rows_with_meta
_structured_legal_batch_from_mask = structured_legal_batch_from_mask
_structured_legal_batch_from_packed = structured_legal_batch_from_packed
_infer_packed_meta_width = infer_packed_meta_width
_hash_unroll = hash_unroll
_hash_state_dict = hash_state_dict
_gae_advantages = gae_advantages
