from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import weiss_sim

from config_example import load_example_config, repo_root
from policy_example import sample_actions_for_policy


def run_minimal_loop_example(*, stack_config: Path, loop_config: Path, steps_override: int | None) -> None:
    config = load_example_config(stack_config_path=stack_config, loop_config_path=loop_config)
    if steps_override is not None:
        config.num_steps = int(steps_override)

    print("Loaded config:")
    print(
        f" mode={config.mode} num_envs={config.num_envs} num_steps={config.num_steps} "
        f"seed={config.seed} policy={config.action_policy}"
    )

    total_rewards = np.zeros(config.num_envs, dtype=np.float64)
    completed_episodes = 0
    engine_error_events = 0

    # New high-level API: use `make()` and batched `ResetBatch` / `StepBatch` helpers.
    with weiss_sim.make(
        mode=config.mode,
        num_envs=config.num_envs,
        seed=config.seed,
        max_decisions=config.max_decisions,
        max_ticks=config.max_ticks,
        observation_visibility=config.observation_visibility,
        error_policy=config.error_policy,
        card_pool="all",
    ) as sim:
        batch = sim.reset(seed=config.seed)
        spec = sim.spec()
        print(f" spec_hash={spec.get('spec_hash')} action_space={sim.action_space}")

        for step_index in range(1, config.num_steps + 1):
            actions = sample_actions_for_policy(
                policy_name=config.action_policy,
                legal_actions=batch.legal,
                base_seed=config.seed,
                step_index=step_index,
            )

            step, _, _ = sim.step_auto(
                actions=actions,
                reset_done=config.auto_reset_done,
                reset_engine_errors=config.auto_reset_engine_errors,
            )

            total_rewards += step.reward.astype(np.float64, copy=False)
            completed_episodes += int(np.count_nonzero(step.done))
            engine_error_events += int(np.count_nonzero(step.engine_status != 0))

            # `step_auto()` updates `sim.latest_batch` after optional resets.
            latest = sim.latest_batch
            if latest is None:
                raise RuntimeError("Simulator did not expose a latest batch after stepping")
            batch = latest

            if step_index % config.log_every == 0 or step_index == config.num_steps:
                mean_reward = float(np.mean(total_rewards))
                done_rate = float(np.count_nonzero(step.done)) / float(config.num_envs)
                print(
                    f"step={step_index:5d} mean_reward={mean_reward:+.4f} "
                    f"done_rate={done_rate:.3f} engine_errors_total={engine_error_events} "
                    f"completed_episodes={completed_episodes}"
                )

    print("Loop completed successfully.")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run the minimal weiss_sim loop example")
    parser.add_argument(
        "--stack-config",
        type=Path,
        default=root / "configs" / "rl_stack_locked.yaml",
        help="Path to the consolidated config index",
    )
    parser.add_argument(
        "--loop-config",
        type=Path,
        default=root / "configs" / "minimal_loop.yaml",
        help="Path to minimal loop overrides",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional override for number of loop steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_minimal_loop_example(
        stack_config=args.stack_config,
        loop_config=args.loop_config,
        steps_override=args.steps,
    )


if __name__ == "__main__":
    main()
