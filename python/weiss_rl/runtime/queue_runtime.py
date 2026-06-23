"""Queue-based single-node runtime for deterministic and throughput-aware training."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weiss_rl.config import StackConfig
from weiss_rl.envs.decision_env import DecisionBoundaryEnv
from weiss_rl.eval.policies.set import (
    HEURISTIC_PUBLIC_POLICY_ID,
)
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.model import PolicyValueModel
from weiss_rl.models.loading import load_snapshot_model_from_path
from weiss_rl.runtime.components.actor_startup import build_runtime_actor_startup_state
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
from weiss_rl.runtime.components.lifecycle import QueueRuntimeLifecycleMixin
from weiss_rl.runtime.components.opponent_mixin import QueueRuntimeOpponentMixin
from weiss_rl.runtime.components.opponent_rows import QueueRuntimeOpponentRowsMixin
from weiss_rl.runtime.components.opponents.central_opponents import QueueRuntimeCentralOpponentMixin
from weiss_rl.runtime.components.opponents.episode_roles import QueueRuntimeEpisodeRolesMixin
from weiss_rl.runtime.components.pending_mixin import QueueRuntimePendingMixin
from weiss_rl.runtime.components.policy_inference.actor_models import (
    maybe_compile_runtime_actor_model,
)
from weiss_rl.runtime.components.policy_outputs import QueueRuntimePolicyOutputMixin
from weiss_rl.runtime.components.policy_rows import QueueRuntimePolicyRowsMixin
from weiss_rl.runtime.components.process import collector_process_main, start_process_collectors
from weiss_rl.runtime.components.runtime_configuration import (
    initialize_runtime_configuration,
    resolve_runtime_actor_device,
)
from weiss_rl.runtime.components.shared_memory.slots import SharedCollectorSlot, SharedPendingUnroll
from weiss_rl.runtime.components.startup_logging import log_runtime_startup
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

_NOLEAGUE_BASELINE_POLICY_ID = NOLEAGUE_BASELINE_POLICY_ID
_SharedCollectorSlot = SharedCollectorSlot
_SharedPendingUnroll = SharedPendingUnroll
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
    return resolve_runtime_actor_device(stack, learner_device=learner_device)


def _maybe_compile_runtime_actor_model(model: PolicyValueModel, *, enabled: bool) -> Any | None:
    return maybe_compile_runtime_actor_model(model, enabled=enabled)


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
        runtime_config_context = initialize_runtime_configuration(
            self,
            stack=stack,
            config=config,
            observation_dim=observation_dim,
            action_dim=action_dim,
            observation_spec=observation_spec,
            spec_bundle=spec_bundle,
            run_dir=run_dir,
            learner_device=learner_device,
        )
        runtime_actor_startup = build_runtime_actor_startup_state(
            model=model,
            config=config,
            process_actor_device_names=self._process_actor_device_names,
            actor_device=self._device,
            use_process_collectors=self._use_process_collectors,
            use_central_batched_collection=self._use_central_batched_collection,
            compile_actor_inference=self._compile_actor_inference,
            build_actor_state=lambda actor_id, shared_actor_model, shared_compiled_actor_model: self._build_actor_state(
                model=model,
                actor_id=actor_id,
                shared_actor_model=shared_actor_model,
                shared_compiled_actor_model=shared_compiled_actor_model,
            ),
            maybe_compile_actor_model=_maybe_compile_runtime_actor_model,
            actor_torch_threads=runtime_config_context.actor_torch_threads,
            configure_actor_torch_threads=_configure_runtime_actor_torch_threads,
        )
        self._install_actor_startup_state(
            runtime_actor_startup,
            performance_log_path=performance_log_path,
        )
        log_runtime_startup(
            self,
            model_kind=runtime_config_context.model_kind,
            training_config=runtime_config_context.training_config,
            structured_warmstart_cfg=runtime_config_context.structured_warmstart_cfg,
        )
        self._initialize_runtime_metrics()
        self._start_collectors_or_refresh_pool(
            model,
            defer_initial_opponent_pool_refresh=defer_initial_opponent_pool_refresh,
        )

    def _install_actor_startup_state(self, runtime_actor_startup: Any, *, performance_log_path: Path | None) -> None:
        self._bootstrap_model_devices = runtime_actor_startup.bootstrap_model_devices
        self._shared_actor_model = runtime_actor_startup.shared_actor_model
        self._shared_compiled_actor_model = runtime_actor_startup.shared_compiled_actor_model
        self._bootstrap_models = runtime_actor_startup.bootstrap_models
        self._actors = runtime_actor_startup.actors
        self._pending_unrolls: deque[PendingUnroll] = deque()
        self._next_actor_index = 0
        self._collector_executor = runtime_actor_startup.collector_executor
        self._last_published_snapshot_version = 0
        self._performance_logger = None if performance_log_path is None else PerformanceLogger(performance_log_path)

    def _initialize_runtime_metrics(self) -> None:
        self._runtime_start = time.time()
        self._runtime_last_metrics_time = self._runtime_start
        self._runtime_cumulative_env_steps = 0

    def _start_collectors_or_refresh_pool(
        self,
        model: PolicyValueModel,
        *,
        defer_initial_opponent_pool_refresh: bool,
    ) -> None:
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

    def _build_actor_state(
        self,
        *,
        model: PolicyValueModel,
        actor_id: int,
        shared_actor_model: Any | None = None,
        shared_compiled_actor_model: Any | None = None,
    ) -> _ActorState:
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
            shared_actor_model=shared_actor_model,
            shared_compiled_actor_model=shared_compiled_actor_model,
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
_concat_batch_major_field = concat_batch_major_field
_hash_unroll = hash_unroll
_hash_state_dict = hash_state_dict
_gae_advantages = gae_advantages
