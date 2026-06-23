# Architecture

This repository is a behavior-sensitive thesis RL pipeline. The code is
organized around explicit contracts: config hashes, simulator spec bundles,
run-tree layouts, seed files, checkpoint schemas, and deterministic evaluation
outputs.

## Pipeline

1. `python -m weiss_rl.cli` selects one of the six public thesis commands.
2. The training workflow validates config and simulator inputs, writes the run
   manifest, builds the learner, and starts training.
3. `weiss_rl.runtime.QueueRuntime` collects simulator-backed decision-boundary
   rollouts and assembles learner batches.
4. `weiss_rl.learners` updates policy/value models and writes metrics,
   checkpoints, snapshots, and TensorBoard logs.
5. The evaluation workflow resolves deterministic policy sets, runs pinned final
   evaluation, and writes reporting artifacts.

## Public Surface

`python -m weiss_rl.cli` is the operator surface. Its public commands are:

- `train-b1`
- `train-main`
- `smoke-eval`
- `eval-final`
- `figures`
- `b2-audit`

Verifiers, diagnostics, and implementation entrypoints may remain runnable, but
they are not the main operator interface. Path-based `python/scripts/*.py`
entrypoints are retired.

## Package Map

| Package | Owns |
| --- | --- |
| `weiss_rl.cli` and `weiss_rl.workflows` | Public commands, profiles, dry-run plans, verification, and command routing. |
| `weiss_rl.config` | Strict config parsing, preset inheritance, overrides, canonical hashes, and seed-file resolution. |
| `weiss_rl.core` and `weiss_rl.envs` | Simulator-facing contracts, legal-action batches, observation layout, action catalog decoding, and env wrappers. |
| `weiss_rl.runtime` | Queue runtime, actor collection, learner-batch assembly, opponent sampling, process collectors, IPC/shared transport, and metrics. |
| `weiss_rl.learners` | IMPALA/V-trace, PPO-lite, learner batch validation, losses, action log-probability helpers, and structured auxiliary metrics. |
| `weiss_rl.models` and `weiss_rl.model` | Policy/value model internals plus the compatibility facade used by training and evaluation. |
| `weiss_rl.eval` | Deterministic evaluation, policy resolution, payoff folding, uncertainty, diagnostics, and paper-readiness reporting. |
| `weiss_rl.league` | Snapshot registry, PFSP sampling, promotion gates, opponent pools, and online outcomes. |
| `weiss_rl.artifacts` and `weiss_rl.diagnostics` | Run layout, manifests, hygiene checks, telemetry, training-log helpers, and diagnostics. |

## Runtime Components

`weiss_rl.runtime.components` is grouped by user-level concept:

- `collection/`: actor scheduling, pending queues, and central collection
  step/action contexts.
- `batching/`: bootstrap fields, reward backfills, and IMPALA/PPO learner
  payload builders.
- `opponents/`: fixed-opponent grouping, heuristic/model overwrites, and
  episode-role accounting.
- `policy_inference/`: actor model selection, central policy outputs,
  deterministic logits, heuristic outputs, and debug validation.
- `ipc_shared/`: collector commands, state-dict IPC, process logging,
  shared transport, and thread setup.
- `shared_memory/`: low-level shared-memory slot configuration and IO.

## Simulator Contract

Simulator-backed training and evaluation target `weiss-sim>=1.2.0,<2`.

Required simulator surfaces include:

- `weiss_sim.OBS_LEN == 378`
- `weiss_sim.ACTION_SPACE_SIZE == 527`
- `weiss_sim.SPEC_HASH == 8590000130`
- `weiss_sim.export_spec_bundle()`
- `weiss_sim.fast(...)`, `weiss_sim.inspect(...)`
- `weiss_sim.make_pool(...)`, `weiss_sim.EnvPoolBuffers`
- `weiss_sim.rl.reset_rl(...)`, `weiss_sim.rl.step_rl(...)`
- fused logit sampling and packed legal-id buffer paths

Published deck presets:

- `starter_deck_ws02_v1`
- `main_deck_5hy_yotsuba_v1`
- `aggro_deck_5hy_nino_v1`
- `control_deck_jj_s66_v1`

## Behavior Boundaries

Treat these as public behavior:

- legal action ID order and packed legality metadata
- observation layout, reward semantics, and done/truncation handling
- rollout packing, learner-batch shapes, and V-trace masks
- checkpoint payloads and snapshot registry paths
- policy ordering, seat swaps, paired seeds, payoff folding, and uncertainty
  schemas
- CLI defaults, config override behavior, and run manifest fields

If a refactor changes one of these surfaces, assume behavior changed until a
test and log entry prove otherwise.

## Reproducibility

Canonical runs record simulator spec hash, spec bundle, canonical config hash,
run ID, run label, git commit, dirty flag, seed-file hashes, hardware summary,
runtime mode, policy-set details, and checkpoint tracker state.

`train_async_fast` records provenance and seeds, but host scheduling can affect
collection order. Use `train_ordered` for order-sensitive debugging and
regression isolation.

If a refactor changes a hash, output order, seed use, checkpoint schema, run
manifest field, or summary shape, assume behavior changed until a test and log
entry prove otherwise.
