"""Process collector child-loop helpers for queue runtime."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from weiss_rl.config import StackConfig
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.model import build_policy_value_model
from weiss_rl.runtime.components.config import QueueRuntimeConfig
from weiss_rl.runtime.components.ipc_shared.collector_commands import handle_collector_commands
from weiss_rl.runtime.components.ipc_shared.ipc import deserialize_state_dict_from_ipc, serialize_state_dict_for_ipc
from weiss_rl.runtime.components.ipc_shared.logging import process_debug_log
from weiss_rl.runtime.components.ipc_shared.shared_transport import (
    create_shared_collector_slot_config,
    open_shared_collector_slot,
    shared_unroll_metadata,
    write_unroll_to_shared_slot,
)
from weiss_rl.runtime.components.ipc_shared.threads import configure_runtime_actor_torch_threads


def start_process_collectors(
    *,
    runtime: Any,
    model: torch.nn.Module,
    collector_process_target: Any,
) -> None:
    """Start process-backed actor collectors for a QueueRuntime instance."""

    system_config = runtime.stack.config.system
    start_method = "spawn" if system_config is None else str(system_config.mp_start_method).strip()
    runtime._process_context = mp.get_context(start_method)
    runtime._collector_result_queue = runtime._process_context.Queue(maxsize=int(runtime.config.queue_capacity_unrolls))
    model_state_dict = serialize_state_dict_for_ipc(
        {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    )
    slot_configs: dict[int, list[dict[str, Any]]] = {}
    if runtime._use_shared_collector_transport:
        hidden_size = int(getattr(model, "hidden_size", 1))
        slot_count = max(
            2,
            min(
                int(runtime.config.queue_capacity_unrolls),
                8,
                (
                    (int(runtime.config.batch_unrolls_per_update) + int(runtime.config.actor_count) - 1)
                    // int(runtime.config.actor_count)
                )
                + 1,
            ),
        )
        for actor_id in range(int(runtime.config.actor_count)):
            actor_slot_configs: list[dict[str, Any]] = []
            actor_slots: list[Any] = []
            for slot_id in range(slot_count):
                slot_config = create_shared_collector_slot_config(
                    actor_id=int(actor_id),
                    slot_id=int(slot_id),
                    profile=str(runtime.config.profile),
                    unroll_length=int(runtime.config.unroll_length),
                    envs_per_actor=int(runtime.config.envs_per_actor),
                    observation_dim=int(runtime.observation_dim),
                    action_dim=int(runtime.action_dim),
                    hidden_size=hidden_size,
                    layout_name=("i16_legal_ids" if str(runtime.config.profile) == "fast" else "mask"),
                    legal_action_meta_width=int(runtime._action_meta_width),
                )
                actor_slot_configs.append(slot_config)
                actor_slots.append(open_shared_collector_slot(slot_config, create=True))
            runtime._collector_shared_slots[int(actor_id)] = tuple(actor_slots)
            slot_configs[int(actor_id)] = actor_slot_configs
    for actor_id in range(int(runtime.config.actor_count)):
        control_queue = runtime._process_context.Queue()
        free_queue = None
        if runtime._use_shared_collector_transport:
            free_queue = runtime._process_context.Queue(maxsize=len(slot_configs[int(actor_id)]))
            for slot_id in range(len(slot_configs[int(actor_id)])):
                free_queue.put(slot_id)
        process = cast(Any, runtime._process_context).Process(
            target=collector_process_target,
            kwargs={
                "stack": runtime.stack,
                "config": runtime.config,
                "model_state_dict": model_state_dict,
                "observation_dim": int(runtime.observation_dim),
                "action_dim": int(runtime.action_dim),
                "observation_spec": runtime._observation_spec,
                "spec_bundle": runtime._spec_bundle,
                "run_dir": (None if runtime._run_dir is None else str(runtime._run_dir)),
                "actor_id": int(actor_id),
                "actor_device_name": str(runtime._process_actor_device_names[int(actor_id)]),
                "learner_device_name": str(runtime._learner_device),
                "control_queue": control_queue,
                "free_queue": free_queue,
                "result_queue": runtime._collector_result_queue,
                "shared_slot_configs": (
                    None if not runtime._use_shared_collector_transport else slot_configs[int(actor_id)]
                ),
            },
            daemon=True,
        )
        process.start()
        runtime._collector_control_queues.append(control_queue)
        if runtime._use_shared_collector_transport:
            runtime._collector_free_queues.append(free_queue)
        runtime._collector_processes.append(process)


def collector_process_main(
    *,
    runtime_cls: Any,
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
    try:
        _collector_process_main_impl(
            runtime_cls=runtime_cls,
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
    except Exception as exc:
        _report_collector_process_error(
            result_queue=result_queue,
            actor_id=int(actor_id),
            run_dir=run_dir,
            exc=exc,
        )
        raise


def _collector_process_main_impl(
    *,
    runtime_cls: Any,
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
    stack_for_child = _stack_for_child_device_config(
        stack=stack,
        actor_device_name=actor_device_name,
        learner_device_name=learner_device_name,
    )
    system_config = stack_for_child.config.system
    if system_config is not None:
        configure_runtime_actor_torch_threads(int(getattr(system_config, "actor_torch_threads", 1)))

    model_config = stack_for_child.config.model
    if model_config is None:
        raise RuntimeError("stack config is missing model config")
    model = build_policy_value_model(
        observation_dim=int(observation_dim),
        config=model_config,
        action_dim=int(action_dim),
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
    ).to(torch.device("cpu"))
    model.load_state_dict(deserialize_state_dict_from_ipc(model_state_dict))
    model.eval()

    local_config = QueueRuntimeConfig(
        mode="train_async_fast",
        actor_count=1,
        envs_per_actor=int(config.envs_per_actor),
        unroll_length=int(config.unroll_length),
        batch_unrolls_per_update=1,
        queue_capacity_unrolls=1,
        profile=str(config.profile),
        base_seed=int(config.base_seed),
        pass_action_id=int(config.pass_action_id),
        actor_reload_interval_updates=int(config.actor_reload_interval_updates),
        pass_with_nonpass_penalty=float(getattr(config, "pass_with_nonpass_penalty", 0.0)),
        mulligan_select_with_confirm_penalty=float(getattr(config, "mulligan_select_with_confirm_penalty", 0.0)),
        terminal_outcome_backfill_reward=float(getattr(config, "terminal_outcome_backfill_reward", 0.0)),
        terminal_outcome_trace_backfill_reward=float(getattr(config, "terminal_outcome_trace_backfill_reward", 0.0)),
        actor_sampling_temperature=float(getattr(config, "actor_sampling_temperature", 1.0)),
        mulligan_force_confirm_after_select=bool(getattr(config, "mulligan_force_confirm_after_select", False)),
        force_pass_over_main_move_only=bool(getattr(config, "force_pass_over_main_move_only", False)),
        main_move_only_max_consecutive=int(getattr(config, "main_move_only_max_consecutive", 0)),
        force_attack_over_pass_when_attack_legal=bool(
            getattr(config, "force_attack_over_pass_when_attack_legal", False)
        ),
    )
    shared_slots = (
        None
        if shared_slot_configs is None
        else tuple(open_shared_collector_slot(shared_slot_config) for shared_slot_config in shared_slot_configs)
    )
    runtime = runtime_cls(
        stack=stack_for_child,
        config=local_config,
        model=model,
        observation_dim=int(observation_dim),
        action_dim=int(action_dim),
        observation_spec=observation_spec,
        spec_bundle=spec_bundle,
        run_dir=(None if run_dir is None else Path(run_dir)),
        performance_log_path=None,
        defer_initial_opponent_pool_refresh=True,
        learner_device=(None if learner_device_name is None else learner_device_name),
    )
    _restore_parent_actor_lane_counts(
        runtime=runtime, stack=stack_for_child, parent_actor_count=int(config.actor_count)
    )
    process_debug_log(
        run_dir=(None if run_dir is None else Path(run_dir)),
        actor_id=int(actor_id),
        message="collector runtime initialized",
    )
    if int(actor_id) != 0:
        runtime._actors[0].env.close()
        runtime._actors[0] = runtime._build_actor_state(model=model, actor_id=int(actor_id))
    actor = runtime._actors[0]
    process_debug_log(
        run_dir=(None if run_dir is None else Path(run_dir)),
        actor_id=int(actor_id),
        message="collector actor ready",
    )
    default_fixed_slots = (
        None
        if actor.fixed_opponent_policy_id_by_env is None
        else np.asarray(actor.fixed_opponent_policy_id_by_env, dtype=object).copy()
    )
    default_forced_policy_ids = tuple(getattr(runtime, "_forced_fixed_opponent_policy_ids", ()))
    default_teacher_active = HEURISTIC_PUBLIC_POLICY_ID in runtime._opponent_heuristic_policies
    default_has_noleague_baseline = NOLEAGUE_BASELINE_POLICY_ID in runtime._opponent_models
    try:
        _collector_loop(
            runtime=runtime,
            actor=actor,
            actor_id=int(actor_id),
            run_dir=run_dir,
            control_queue=control_queue,
            free_queue=free_queue,
            result_queue=result_queue,
            shared_slots=shared_slots,
            default_fixed_slots=default_fixed_slots,
            default_forced_policy_ids=default_forced_policy_ids,
            default_teacher_active=default_teacher_active,
            default_has_noleague_baseline=default_has_noleague_baseline,
        )
    finally:
        if shared_slots is not None:
            for shared_slot in shared_slots:
                shared_slot.close(unlink=False)
        runtime.close()


def _report_collector_process_error(
    *,
    result_queue: Any,
    actor_id: int,
    run_dir: str | None,
    exc: Exception,
) -> None:
    payload = {
        "kind": "collector_error_v1",
        "actor_id": int(actor_id),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    process_debug_log(
        run_dir=(None if run_dir is None else Path(run_dir)),
        actor_id=int(actor_id),
        message=f"collector error {type(exc).__name__}: {exc}",
    )
    try:
        result_queue.put_nowait(payload)
    except Exception:
        try:
            result_queue.put(payload, timeout=1.0)
        except Exception:
            return


def _stack_for_child_device_config(
    *,
    stack: StackConfig,
    actor_device_name: str | None,
    learner_device_name: str | None,
) -> StackConfig:
    system_config = stack.config.system
    stack_for_child = stack
    if system_config is not None:
        child_system = system_config
        if actor_device_name is not None:
            child_system = replace(child_system, actor_device=str(actor_device_name))
        if learner_device_name is not None:
            child_system = replace(child_system, learner_device=str(learner_device_name))
        if child_system is not system_config:
            stack_for_child = replace(
                stack,
                config=replace(
                    stack.config,
                    system=child_system,
                ),
            )
            system_config = stack_for_child.config.system
    if (
        system_config is not None
        and str(getattr(system_config, "collection_backend", "auto")).strip().lower() == "process"
    ):
        stack_for_child = replace(
            stack_for_child,
            config=replace(
                stack_for_child.config,
                system=replace(system_config, collection_backend="auto"),
            ),
        )
    return stack_for_child


def _restore_parent_actor_lane_counts(
    *,
    runtime: Any,
    stack: StackConfig,
    parent_actor_count: int,
) -> None:
    """Preserve parent global actor-lane limits inside a single-actor child runtime."""

    training_config = getattr(getattr(stack, "config", None), "training", None)
    requested_diverse_actor_count = (
        0 if training_config is None else int(getattr(training_config, "diverse_opponent_actor_count", 0))
    )
    requested_diverse_model_actor_count = (
        0 if training_config is None else int(getattr(training_config, "diverse_model_actor_count", 0))
    )
    global_actor_count = max(0, int(parent_actor_count))
    diverse_actor_count = min(global_actor_count, max(0, requested_diverse_actor_count))
    runtime._diverse_opponent_actor_count = diverse_actor_count
    runtime._diverse_model_actor_count = min(diverse_actor_count, max(0, requested_diverse_model_actor_count))


def _collector_loop(
    *,
    runtime: Any,
    actor: Any,
    actor_id: int,
    run_dir: str | None,
    control_queue: Any,
    free_queue: Any | None,
    result_queue: Any,
    shared_slots: tuple[Any, ...] | None,
    default_fixed_slots: np.ndarray | None,
    default_forced_policy_ids: tuple[str, ...],
    default_teacher_active: bool,
    default_has_noleague_baseline: bool,
) -> None:
    while True:
        if _should_stop_collector(
            runtime=runtime,
            actor=actor,
            control_queue=control_queue,
            default_fixed_slots=default_fixed_slots,
            default_forced_policy_ids=default_forced_policy_ids,
            default_teacher_active=default_teacher_active,
            default_has_noleague_baseline=default_has_noleague_baseline,
        ):
            return
        process_debug_log(
            run_dir=(None if run_dir is None else Path(run_dir)),
            actor_id=int(actor_id),
            message="collector collect start",
        )
        unroll = runtime._collect_actor_unroll(actor)
        process_debug_log(
            run_dir=(None if run_dir is None else Path(run_dir)),
            actor_id=int(actor_id),
            message="collector collect done",
        )
        if shared_slots is None or free_queue is None:
            result_queue.put(unroll)
            process_debug_log(
                run_dir=(None if run_dir is None else Path(run_dir)),
                actor_id=int(actor_id),
                message="collector result queued direct",
            )
            continue

        slot_id = _next_free_slot_id(
            runtime=runtime,
            actor=actor,
            actor_id=actor_id,
            control_queue=control_queue,
            free_queue=free_queue,
            shared_slot_count=len(shared_slots),
            default_fixed_slots=default_fixed_slots,
            default_forced_policy_ids=default_forced_policy_ids,
            default_teacher_active=default_teacher_active,
            default_has_noleague_baseline=default_has_noleague_baseline,
        )
        if slot_id is None:
            return
        write_unroll_to_shared_slot(shared_slots[slot_id], unroll)
        result_queue.put(shared_unroll_metadata(unroll, slot_id=slot_id))
        process_debug_log(
            run_dir=(None if run_dir is None else Path(run_dir)),
            actor_id=int(actor_id),
            message=f"collector result queued shared slot={slot_id}",
        )


def _next_free_slot_id(
    *,
    runtime: Any,
    actor: Any,
    actor_id: int,
    control_queue: Any,
    free_queue: Any,
    shared_slot_count: int,
    default_fixed_slots: np.ndarray | None,
    default_forced_policy_ids: tuple[str, ...],
    default_teacher_active: bool,
    default_has_noleague_baseline: bool,
) -> int | None:
    while True:
        if _should_stop_collector(
            runtime=runtime,
            actor=actor,
            control_queue=control_queue,
            default_fixed_slots=default_fixed_slots,
            default_forced_policy_ids=default_forced_policy_ids,
            default_teacher_active=default_teacher_active,
            default_has_noleague_baseline=default_has_noleague_baseline,
        ):
            return None
        try:
            token = free_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if token == "stop":
            return None
        slot_id = int(token)
        if slot_id < 0 or slot_id >= int(shared_slot_count):
            raise RuntimeError(f"collector {actor_id} received invalid shared slot token {slot_id}")
        return slot_id


def _should_stop_collector(
    *,
    runtime: Any,
    actor: Any,
    control_queue: Any,
    default_fixed_slots: np.ndarray | None,
    default_forced_policy_ids: tuple[str, ...],
    default_teacher_active: bool,
    default_has_noleague_baseline: bool,
) -> bool:
    return handle_collector_commands(
        runtime=runtime,
        actor=actor,
        control_queue=control_queue,
        default_fixed_slots=default_fixed_slots,
        default_forced_policy_ids=default_forced_policy_ids,
        default_teacher_active=default_teacher_active,
        default_has_noleague_baseline=default_has_noleague_baseline,
    )
