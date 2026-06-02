"""Actor-state and environment construction helpers for queue runtime."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from weiss_rl.artifacts.reproducibility import derive_actor_seed
from weiss_rl.envs.decision_env import DecisionBoundaryBatch, DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config
from weiss_rl.runtime.components.opponent_context import initial_seat_hidden_for_opponents
from weiss_rl.runtime.components.policy_ids import MIRROR_OPPONENT_POLICY_ID


@dataclass(slots=True)
class _ActorState:
    actor_id: int
    env: DecisionBoundaryEnv
    model: Any
    compiled_model: Any | None
    rng: np.random.Generator
    seat_hidden: torch.Tensor
    current_batch: DecisionBoundaryBatch
    layout_name: str
    focal_seat_by_env: np.ndarray
    opponent_policy_id_by_env: np.ndarray
    opponent_hidden: torch.Tensor
    diverse_opponent_lane: bool
    force_model_policy_lane: bool
    fixed_opponent_policy_id_by_env: np.ndarray | None = None
    snapshot_version: int = 0
    next_unroll_seq: int = 0


def build_runtime_env(
    *,
    stack: Any,
    profile: str,
    envs_per_actor: int,
    pass_action_id: int,
    seed: int,
    actor_id: int,
    profile_timers: bool,
) -> tuple[DecisionBoundaryEnv, str]:
    env_config = build_env_config_from_stack(stack, seed=int(seed), actor_id=int(actor_id))
    pool, layout_name = make_env_pool_from_config(
        env_config,
        profile=profile,  # type: ignore[arg-type]
        num_envs=int(envs_per_actor),
    )
    legality = "ids_offsets" if layout_name == "i16_legal_ids" else "mask"
    max_no_progress_decisions = _max_no_progress_decisions(stack)
    env = DecisionBoundaryEnv(
        pool,
        legality=legality,  # type: ignore[arg-type]
        pass_action_id=int(pass_action_id),
        engine_status_policy="hard_fail",
        # QueueRuntime copies step outputs into unroll buffers before the next
        # simulator call, so views avoid an extra allocation per reset/step.
        copy_arrays=False,
        max_decisions=int(env_config["max_decisions"]),
        max_ticks=int(env_config["max_ticks"]),
        max_no_progress_decisions=max_no_progress_decisions,
        profile_timers=bool(profile_timers),
    )
    return env, str(layout_name)


def build_actor_state(
    *,
    actor_state_cls: Any,
    model: Any,
    actor_id: int,
    env: DecisionBoundaryEnv,
    layout_name: str,
    base_seed: int,
    envs_per_actor: int,
    device: torch.device,
    shared_actor_model: Any | None,
    shared_compiled_actor_model: Any | None,
    maybe_compile_actor_model: Callable[[Any], Any | None],
    legal_action_meta_from_ids: Callable[[np.ndarray], np.ndarray | None],
    fixed_opponent_policy_slots: Callable[[], np.ndarray | None],
    diverse_opponent_actor_count: int,
    diverse_model_actor_count: int,
    assign_episode_roles: Callable[[Any, np.ndarray], None],
) -> Any:
    seed = actor_seed(base_seed, actor_id)
    if shared_actor_model is not None:
        actor_model = shared_actor_model
        compiled_model = shared_compiled_actor_model
    else:
        actor_model = copy.deepcopy(model).to(device)
        actor_model.eval()
        compiled_model = maybe_compile_actor_model(actor_model)

    current_batch = env.reset(seed=seed)
    if current_batch.ids_offsets is not None and current_batch.legal_action_meta is None:
        legal_action_meta = legal_action_meta_from_ids(current_batch.ids_offsets[0])
        if legal_action_meta is not None:
            current_batch = replace(current_batch, legal_action_meta=legal_action_meta)
            env._last_batch = current_batch

    state = actor_state_cls(
        actor_id=actor_id,
        env=env,
        model=actor_model,
        compiled_model=compiled_model,
        rng=np.random.default_rng(seed),
        seat_hidden=actor_model.initial_seat_hidden(envs_per_actor, device=device),
        current_batch=current_batch,
        layout_name=layout_name,
        focal_seat_by_env=np.zeros((int(envs_per_actor),), dtype=np.int64),
        opponent_policy_id_by_env=np.full(
            (int(envs_per_actor),),
            MIRROR_OPPONENT_POLICY_ID,
            dtype=object,
        ),
        opponent_hidden=actor_model.initial_seat_hidden(envs_per_actor, device=device),
        diverse_opponent_lane=(int(actor_id) < int(diverse_opponent_actor_count)),
        force_model_policy_lane=(int(actor_id) < int(diverse_model_actor_count)),
        fixed_opponent_policy_id_by_env=fixed_opponent_policy_slots(),
    )
    assign_episode_roles(state, np.ones((int(envs_per_actor),), dtype=np.bool_))
    state.seat_hidden = initial_seat_hidden_for_opponents(
        actor_model,
        envs_per_actor,
        device=device,
        opponent_policy_ids=state.opponent_policy_id_by_env,
    )
    return state


def actor_seed(base_seed: int, actor_id: int) -> int:
    return derive_actor_seed(int(base_seed), actor_id=int(actor_id))


def _max_no_progress_decisions(stack: Any) -> int | None:
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    raw_limit = curriculum.simulator.get("max_no_progress_decisions")
    if raw_limit is None:
        return None
    return int(raw_limit)
