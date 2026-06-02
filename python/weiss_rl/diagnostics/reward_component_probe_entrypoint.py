from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from weiss_rl.config import load_stack_config
from weiss_rl.envs.pool_factory import build_env_config_from_stack, make_env_pool_from_config

COMPONENT_NAMES = ("terminal", "damage", "level", "board", "no_progress")


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "abs_mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0,
            "nonzero_fraction": 0.0,
            "positive_fraction": 0.0,
            "negative_fraction": 0.0,
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "abs_mean": float(np.abs(values).mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "sum": float(values.sum()),
        "nonzero_fraction": float(np.mean(values != 0.0)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "negative_fraction": float(np.mean(values < 0.0)),
    }


def summarize_reward_samples(
    *,
    rewards: np.ndarray,
    reward_components: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    engine_status: np.ndarray,
) -> dict[str, Any]:
    rewards = np.asarray(rewards, dtype=np.float32)
    components = np.asarray(reward_components, dtype=np.float32)
    if components.ndim != 2 or int(components.shape[1]) != len(COMPONENT_NAMES):
        raise ValueError(f"reward_components must have shape (N, {len(COMPONENT_NAMES)}), got {components.shape}")
    component_sums = components.sum(axis=1)
    return {
        "transition_count": int(rewards.size),
        "reward": _stats(rewards),
        "components": {name: _stats(components[:, index]) for index, name in enumerate(COMPONENT_NAMES)},
        "component_sum_error_max_abs": float(np.max(np.abs(rewards.reshape(-1) - component_sums)))
        if rewards.size
        else 0.0,
        "terminated_fraction": float(np.mean(np.asarray(terminated, dtype=np.bool_))) if rewards.size else 0.0,
        "truncated_fraction": float(np.mean(np.asarray(truncated, dtype=np.bool_))) if rewards.size else 0.0,
        "engine_fault_fraction": float(np.mean(np.asarray(engine_status, dtype=np.uint8) != 0))
        if rewards.size
        else 0.0,
    }


def _legal_actions_from_debug_out(out: Any, *, pass_action_id: int, rng: np.random.Generator) -> np.ndarray:
    masks = np.asarray(out.masks, dtype=np.bool_)
    if masks.ndim != 2:
        raise ValueError(f"debug masks must be 2D, got {masks.shape}")
    actions = np.full((masks.shape[0],), int(pass_action_id), dtype=np.uint32)
    for row_index, row in enumerate(masks):
        legal = np.flatnonzero(row)
        if legal.size:
            actions[row_index] = np.uint32(legal[int(rng.integers(0, legal.size))])
    return actions


def run_reward_component_probe(
    *,
    stack_config: Path,
    num_envs: int,
    steps: int,
    seed: int,
    event_capacity: int,
) -> dict[str, Any]:
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    stack = load_stack_config(stack_config)
    env_config = build_env_config_from_stack(stack, seed=seed)
    pool, layout_name = make_env_pool_from_config(env_config, profile="debug", num_envs=num_envs)
    if layout_name != "mask":
        raise RuntimeError(f"debug reward probe expects mask layout, got {layout_name!r}")

    import weiss_sim

    out = weiss_sim.BatchOutDebug(num_envs, int(event_capacity))
    pool.reset_debug_into(out)
    rng = np.random.default_rng(seed)
    reward_rows: list[np.ndarray] = []
    component_rows: list[np.ndarray] = []
    terminated_rows: list[np.ndarray] = []
    truncated_rows: list[np.ndarray] = []
    engine_status_rows: list[np.ndarray] = []
    pass_action_id = int(getattr(weiss_sim, "PASS_ACTION_ID", 51))

    for _ in range(steps):
        actions = _legal_actions_from_debug_out(out, pass_action_id=pass_action_id, rng=rng)
        pool.step_debug_into(actions, out)
        reward_rows.append(np.asarray(out.rewards, dtype=np.float32).copy())
        component_rows.append(np.asarray(out.reward_components, dtype=np.float32).copy())
        terminated = np.asarray(out.terminated, dtype=np.bool_).copy()
        truncated = np.asarray(out.truncated, dtype=np.bool_).copy()
        terminated_rows.append(terminated)
        truncated_rows.append(truncated)
        engine_status_rows.append(np.asarray(out.engine_status, dtype=np.uint8).copy())
        if bool(np.any(terminated | truncated)):
            pool.reset_debug_into(out)

    close_fn = getattr(pool, "close", None)
    if callable(close_fn):
        close_fn()

    reward_array = np.concatenate(reward_rows, axis=0)
    component_array = np.concatenate(component_rows, axis=0)
    summary = summarize_reward_samples(
        rewards=reward_array,
        reward_components=component_array,
        terminated=np.concatenate(terminated_rows, axis=0),
        truncated=np.concatenate(truncated_rows, axis=0),
        engine_status=np.concatenate(engine_status_rows, axis=0),
    )
    summary.update(
        {
            "stack_config": str(stack_config),
            "num_envs": int(num_envs),
            "steps": int(steps),
            "seed": int(seed),
            "reward_json": env_config.get("reward_json"),
            "curriculum_json": env_config.get("curriculum_json"),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe simulator reward components through BatchOutDebug")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--event-capacity", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_reward_component_probe(
        stack_config=args.stack_config,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        event_capacity=args.event_capacity,
    )
    output_path = args.output_json
    if output_path is None:
        output_path = Path("runs") / "diagnostics" / f"reward_components_{args.stack_config.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
