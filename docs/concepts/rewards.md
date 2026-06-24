# Rewards

This page explains the reward path. Use it when the question is "what exactly
is the learner optimizing?"

## Objective

The thesis reward objective is terminal win/loss from the learner perspective,
with optional shaping for earlier credit assignment.

The active config block is `rewards:` in the thesis YAML files. The main B1
config uses:

- `objective: terminal_pm1`
- terminal win: `+1`
- terminal loss: `-1`
- terminal draw: `0`
- timeout/truncation reward from config
- optional damage, level, board, and no-progress shaping

`terminal_only_pm1` is retained as the reward ablation and disables shaping.

## Where Rewards Are Configured

| Concept | Files |
| --- | --- |
| Reward config records | `python/weiss_rl/config/schemas/environment_models.py` |
| Reward config parsing | `python/weiss_rl/config/sections/sections_environment.py` |
| Simulator environment config | `python/weiss_rl/envs/env_config.py` |
| Simulator reward contract | `python/weiss_rl/envs/simulator_reward_contract.py`, `python/weiss_rl/envs/reward_payload.py` |
| Reward flow map | `python/weiss_rl/runtime/components/rewards/reward_flow.py` |
| Runtime reward shaping | `python/weiss_rl/runtime/components/rewards/reward_shaping.py`, `python/weiss_rl/runtime/components/rewards/reward_shaping_plan.py`, `python/weiss_rl/runtime/components/rewards/reward_shaping_pass.py`, `python/weiss_rl/runtime/components/rewards/reward_shaping_mulligan.py`, `python/weiss_rl/runtime/components/rewards/reward_shaping_counters.py` |
| Learner-batch terminal backfill | `python/weiss_rl/runtime/components/batching/reward_backfill.py` |
| Reward component probe | `python/weiss_rl/diagnostics/probes/reward_component_probe_entrypoint.py` |
| B1 reward config | `configs/thesis/b1_noleague.yaml` |
| Reward ablation | `configs/thesis/ablations/terminal_only_reward.yaml` |

## Simulator Payload

`env_config.py` adds `reward_json` to the simulator environment config.
`simulator_reward_contract.py` translates the stack config into the exact
payload accepted by the simulator. This keeps the simulator-facing reward
definition next to environment construction rather than scattering constants
through the learner.

For `terminal_only_pm1`, the payload forces shaping terms to zero even if a
shared base config contains shaping defaults.

## Learner Perspective

The learner receives rewards on focal decision rows. The runtime keeps the
perspective aligned with the learner seat so a positive terminal reward means a
win for the focal policy, not simply the acting player in the raw simulator
step.

Relevant files:

- `python/weiss_rl/envs/learner_turn_env.py`
- `python/weiss_rl/actors/actor_worker.py`
- `python/weiss_rl/actors/action_accounting.py`
- `python/weiss_rl/runtime/components/rewards/reward_shaping.py`
- `python/weiss_rl/runtime/components/rewards/reward_shaping_pass.py`
- `python/weiss_rl/runtime/components/rewards/reward_shaping_mulligan.py`
- `python/weiss_rl/runtime/components/rewards/reward_shaping_counters.py`

## Runtime Shaping

Runtime-side shaping is deliberately narrow. The retained shaping path covers
two local action-quality penalties:

- pass-with-nonpass-available discourages choosing pass when a meaningful
  nonpass option is legal;
- mulligan-select-with-confirm discourages extra mulligan selection when confirm
  is already legal.

Both rules subtract from the learner reward and write counters so the effect is
visible in metrics.

`reward_shaping.py` is the collector composer. The pass and mulligan modules
own the rule masks and reward edits, while `reward_shaping_counters.py` owns
the metric counter updates.
`reward_shaping_plan.py` names the rule order and data requirements so the
collector-side shaping path can be reviewed as a small checklist.

## Terminal Backfill

The learner-batch builder also handles terminal outcome backfill settings. These
settings are explicit config values and are tested so truncation rewards are not
applied twice.

There are two separate backfill modes:

- `terminal_outcome_backfill_reward` credits the last trainable row when a
  terminal win/loss lands on a non-train row;
- `terminal_outcome_trace_backfill_reward` spreads terminal win/loss credit to
  earlier trainable rows in the same in-batch episode suffix.

Relevant files:

- `python/weiss_rl/runtime/components/rewards/reward_shaping.py`
- `python/weiss_rl/runtime/components/rewards/reward_shaping_plan.py`
- `python/weiss_rl/runtime/components/rewards/reward_shaping_pass.py`
- `python/weiss_rl/runtime/components/rewards/reward_shaping_mulligan.py`
- `python/weiss_rl/runtime/components/rewards/reward_shaping_counters.py`
- `python/weiss_rl/runtime/components/batching/reward_backfill.py`
- `tests/weiss_rl/test_runtime_reward_shaping.py`
- `tests/weiss_rl/test_runtime_reward_backfill.py`
- `tests/weiss_rl/test_runtime_queue_impala_discounts.py`

## Discounting

Discount settings come from `rewards.discount.gamma`. Nonterminal learner rows
use gamma; terminal rows use zero discount. Timeout and truncation behavior is
handled separately so a timeout reward is not silently mixed with normal
terminal win/loss semantics.

## Why Shaping Exists

Weiss Schwarz games can have long delayed outcomes. Sparse terminal rewards are
clean but can give weak update signals early in training. Shaping provides
intermediate information while the terminal result remains the real objective.

The reward ablation keeps this honest: it allows a terminal-only comparison
against the shaped default.

## Reward Flow Map

`python/weiss_rl/runtime/components/rewards/reward_flow.py` names the end-to-end path:
simulator reward payload, learner-perspective rows, collector shaping, terminal
backfill, reward-component probing, and final-evaluation outcome scoring. The
reward component probe writes this map beside the component checks so a retained
diagnostic artifact explains which part of the reward path it is validating.

## What To Check

When reward behavior changes, check:

- the YAML `rewards:` block;
- the simulator reward payload;
- the reward component probe `probe_plan` and component sums;
- learner-row perspective;
- terminal and timeout rows;
- reward backfill metrics;
- V-trace targets and discount masks;
- whether final evaluation still uses game outcome, not training shaping.
