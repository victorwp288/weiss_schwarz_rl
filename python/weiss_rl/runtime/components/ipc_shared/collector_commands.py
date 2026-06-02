"""Process-collector command handling for the queue runtime."""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np

from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.model import build_policy_value_model
from weiss_rl.models.loading import restore_model_guidance_from_payload
from weiss_rl.models.state_dict_compat import load_model_state_dict_with_context_compat
from weiss_rl.runtime.components.ipc_shared.ipc import deserialize_state_dict_from_ipc
from weiss_rl.runtime.components.ipc_shared.logging import process_debug_log

_NOLEAGUE_BASELINE_POLICY_ID = NOLEAGUE_BASELINE_POLICY_ID


def handle_collector_commands(
    *,
    runtime: Any,
    actor: Any,
    control_queue: Any,
    default_fixed_slots: np.ndarray | None,
    default_forced_policy_ids: tuple[str, ...],
    default_teacher_active: bool,
    default_has_noleague_baseline: bool,
) -> bool:
    """Apply pending process-collector commands.

    Returns True when the collector should stop. Unknown command kinds are ignored,
    preserving the historical runtime behavior.
    """

    while True:
        try:
            command = control_queue.get_nowait()
        except queue.Empty:
            return False
        kind = str(command.get("kind", ""))
        _debug(runtime=runtime, actor=actor, message=f"command kind={kind}")
        if kind == "stop":
            return True
        if kind == "reload":
            _handle_reload(runtime=runtime, actor=actor, command=command)
            continue
        if kind == "set_update":
            _handle_set_update(runtime=runtime, actor=actor, command=command)
            continue
        if kind == "refresh_opponent_pool":
            _handle_refresh_opponent_pool(runtime=runtime, actor=actor, command=command)
            continue
        if kind == "set_fixed_opponents":
            _handle_set_fixed_opponents(
                runtime=runtime,
                actor=actor,
                command=command,
                default_fixed_slots=default_fixed_slots,
                default_forced_policy_ids=default_forced_policy_ids,
                default_teacher_active=default_teacher_active,
                default_has_noleague_baseline=default_has_noleague_baseline,
            )


def _handle_reload(*, runtime: Any, actor: Any, command: dict[str, Any]) -> None:
    actor.model.load_state_dict(deserialize_state_dict_from_ipc(command["model_state_dict"]))
    actor.model.eval()
    update = int(command.get("update", actor.snapshot_version))
    actor.snapshot_version = update
    runtime._current_learner_update = update
    if "effective_update" in command:
        runtime._effective_learner_update = int(command["effective_update"])
    if bool(command.get("refresh_opponent_pool", False)):
        _refresh_opponent_pool(runtime=runtime, actor=actor, command_name="reload")


def _handle_set_update(*, runtime: Any, actor: Any, command: dict[str, Any]) -> None:
    runtime._current_learner_update = int(command.get("update", getattr(runtime, "_current_learner_update", 0)))
    if "effective_update" in command:
        runtime._effective_learner_update = int(command["effective_update"])
    if bool(command.get("refresh_opponent_pool", False)):
        _refresh_opponent_pool(runtime=runtime, actor=actor, command_name="set_update")


def _handle_refresh_opponent_pool(*, runtime: Any, actor: Any, command: dict[str, Any]) -> None:
    if "update" in command:
        runtime._current_learner_update = int(command["update"])
    if "effective_update" in command:
        runtime._effective_learner_update = int(command["effective_update"])
    _refresh_opponent_pool(runtime=runtime, actor=actor, command_name="refresh_opponent_pool")


def _handle_set_fixed_opponents(
    *,
    runtime: Any,
    actor: Any,
    command: dict[str, Any],
    default_fixed_slots: np.ndarray | None,
    default_forced_policy_ids: tuple[str, ...],
    default_teacher_active: bool,
    default_has_noleague_baseline: bool,
) -> None:
    restore_defaults = bool(command.get("restore_defaults", False))
    activate_teacher = (
        default_teacher_active if restore_defaults else bool(command.get("activate_teacher_heuristic", False))
    )
    if activate_teacher and runtime._teacher_policy is not None:
        runtime._opponent_heuristic_policies[HEURISTIC_PUBLIC_POLICY_ID] = runtime._teacher_policy
    elif not default_teacher_active:
        runtime._opponent_heuristic_policies.pop(HEURISTIC_PUBLIC_POLICY_ID, None)

    if restore_defaults:
        runtime._forced_fixed_opponent_policy_ids = tuple(default_forced_policy_ids)
    else:
        runtime._forced_fixed_opponent_policy_ids = tuple(
            str(policy_id) for policy_id in command.get("forced_policy_ids", ())
        )

    baseline_state_dict = None if restore_defaults else command.get("noleague_baseline_state_dict")
    if baseline_state_dict is not None:
        baseline_model = build_policy_value_model(
            observation_dim=int(runtime.observation_dim),
            config=runtime.stack.config.model,
            action_dim=int(runtime.action_dim),
            observation_spec=runtime._observation_spec,
            spec_bundle=runtime._spec_bundle,
        ).to(runtime._device)
        load_model_state_dict_with_context_compat(
            baseline_model,
            deserialize_state_dict_from_ipc(baseline_state_dict),
            context="collector B1 baseline",
        )
        guidance_payload = command.get("noleague_baseline_guidance_payload")
        if isinstance(guidance_payload, dict):
            restore_model_guidance_from_payload(baseline_model, guidance_payload)
        baseline_model.eval()
        runtime._opponent_models[_NOLEAGUE_BASELINE_POLICY_ID] = baseline_model
        runtime._opponent_model_locks[_NOLEAGUE_BASELINE_POLICY_ID] = threading.Lock()
    elif restore_defaults and not default_has_noleague_baseline:
        runtime._opponent_models.pop(_NOLEAGUE_BASELINE_POLICY_ID, None)
        runtime._opponent_model_locks.pop(_NOLEAGUE_BASELINE_POLICY_ID, None)

    if restore_defaults:
        actor.fixed_opponent_policy_id_by_env = (
            None if default_fixed_slots is None else np.asarray(default_fixed_slots, dtype=object).copy()
        )
    else:
        fixed_slots = command.get("fixed_opponent_policy_id_by_env")
        actor.fixed_opponent_policy_id_by_env = None if fixed_slots is None else np.asarray(fixed_slots, dtype=object)
    runtime._reset_actor_state_for_fixed_opponents(actor)


def _refresh_opponent_pool(*, runtime: Any, actor: Any, command_name: str) -> None:
    _debug(runtime=runtime, actor=actor, message=f"command {command_name} refresh_opponent_pool start")
    runtime.refresh_opponent_pool()
    _debug(runtime=runtime, actor=actor, message=f"command {command_name} refresh_opponent_pool done")


def _debug(*, runtime: Any, actor: Any, message: str) -> None:
    process_debug_log(
        run_dir=getattr(runtime, "_run_dir", None),
        actor_id=getattr(actor, "actor_id", -1),
        message=message,
    )
