from __future__ import annotations

from typing import Any

from weiss_rl.envs.decision_env import DecisionBoundaryEnv
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config


def spec_dimensions(contract: Any) -> tuple[int, int]:
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    return observation_dim, action_dim


def env_pool_config(stack: Any, *, seed: int) -> dict[str, Any]:
    return build_env_config_from_stack(stack, seed=int(seed))


def _max_no_progress_decisions(stack: Any) -> int | None:
    curriculum = stack.config.curriculum
    if curriculum is None:
        return None
    raw_limit = curriculum.simulator.get("max_no_progress_decisions")
    if raw_limit is None:
        return None
    return int(raw_limit)


def build_training_env(
    stack: Any,
    *,
    profile: str,
    num_envs: int,
    seed: int,
) -> DecisionBoundaryEnv:
    config = env_pool_config(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(
        config,
        profile=profile,  # type: ignore[arg-type]
        num_envs=num_envs,
    )
    if layout_name != "mask":
        raise RuntimeError(
            "The compatibility training path expects mask legality because ImpalaLearner consumes legal_mask. "
            f"Profile {profile!r} resolved to layout {layout_name!r}."
        )
    return DecisionBoundaryEnv(
        pool,
        legality="mask",
        engine_status_policy="hard_fail",
        max_decisions=int(config["max_decisions"]),
        max_ticks=int(config["max_ticks"]),
        max_no_progress_decisions=_max_no_progress_decisions(stack),
    )


def build_ids_eval_env(
    stack: Any,
    *,
    seed: int,
    pass_action_id: int,
) -> DecisionBoundaryEnv:
    config = env_pool_config(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(
        config,
        profile="fast",
        num_envs=1,
    )
    if layout_name != "i16_legal_ids":
        raise RuntimeError(
            "Periodic dev eval requires ids-based legality for the pinned eval protocol. "
            f"Profile 'fast' resolved to layout {layout_name!r}."
        )
    return DecisionBoundaryEnv(
        pool,
        legality="ids_offsets",
        pass_action_id=pass_action_id,
        engine_status_policy="hard_fail",
        max_decisions=int(config["max_decisions"]),
        max_ticks=int(config["max_ticks"]),
        max_no_progress_decisions=_max_no_progress_decisions(stack),
    )
