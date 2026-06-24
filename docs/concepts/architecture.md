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

`python/weiss_rl/workflows/command_surface.py` owns command names, help text,
and the training/evaluation command groups.
`python/weiss_rl/workflows/workflow_route_explanation.py` renders the checked
command-to-dispatch-to-plan-builder map and the public workflow lifecycle:
register command, parse arguments, build plan, dispatch, retain outputs. Runtime
dispatch still lives in `workflow_dispatch.py`; training and evaluation
workflow packages build the actual plans.

## Package Map

| Package | Owns |
| --- | --- |
| `weiss_rl.cli` and `weiss_rl.workflows` | Public command registry, command-group dispatch, profiles, dry-run plans, and verification. |
| `weiss_rl.config` | Strict config parsing, preset inheritance, overrides, canonical hashes, and seed-file resolution. |
| `weiss_rl.core` and `weiss_rl.envs` | Simulator-facing contracts, legal-action batches, observation layout, action catalog decoding, and env wrappers. |
| `weiss_rl.runtime` | Queue runtime, actor collection, learner-batch assembly, opponent sampling, process collectors, IPC/shared transport, and metrics. |
| `weiss_rl.learners` | IMPALA/V-trace, PPO-lite, learner batch validation, losses, action log-probability helpers, and structured auxiliary metrics. |
| `weiss_rl.models` and `weiss_rl.model` | Policy/value model internals plus the compatibility facade used by training and evaluation. |
| `weiss_rl.eval` | Deterministic evaluation, policy resolution, payoff folding, uncertainty, diagnostics, and paper-readiness reporting. |
| `weiss_rl.league` | Snapshot registry, PFSP sampling, promotion gates, opponent pools, and online outcomes. |
| `weiss_rl.artifacts` and `weiss_rl.diagnostics` | Run layout, manifests, hygiene checks, telemetry, training-log helpers, and diagnostics. |

## Runtime Components

`weiss_rl.runtime.components` is grouped by runtime responsibility:

- `actors/`: actor startup, state, routing, unroll execution, and actor-side
  policy rows.
- `central/`: centralized actor collection, policy phases, row partitioning,
  unroll assembly, and finalization.
- `collection/`: collector state, action execution, legal-action steps, unroll
  storage, batch collection, and terminal resets.
- `batching/`: bootstrap fields, counters, legal batching, metrics, reward
  backfills, and IMPALA/PPO learner payload builders.
- `actions/`: action catalog setup, dense/packed action surfaces, pass/mulligan
  guards, legal metadata, and policy row/output records.
- `rewards/`: reward flow maps and reward-shaping plans/counters.
- `opponents/`: fixed-opponent grouping, heuristic/model overwrites, residency,
  and episode-role accounting.
- `policy_inference/`: actor model selection, central policy outputs,
  deterministic logits, heuristic outputs, and debug validation.
- `ipc_shared/` and `shared_memory/`: collector commands, state-dict IPC,
  process logging, shared transport, thread setup, and low-level shared-memory
  slot IO.

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

The code-level section map lives in
`python/weiss_rl/core/simulator_contract.py`:

| Section | Source | Why It Matters |
| --- | --- | --- |
| Runtime identity | `__version__`, module path, build info, db info | Proves which simulator build produced the run. |
| Spec hash | `SPEC_HASH`, `export_spec_bundle()` | Guards startup, checkpoints, and readiness checks against layout drift. |
| Observation layout | `spec_bundle.observation` | Keeps encoders, model trunks, and replay diagnostics on the same vector layout. |
| Action catalog | `spec_bundle.action` | Keeps dense logits, legal masks, and packed candidate IDs aligned. |
| Pass action | `spec_bundle.action.pass_action_id` | Gives env wrappers, samplers, and evaluation policies one fallback action ID. |
| Thesis deck presets | `weiss_sim.cards` preset APIs | Confirms retained runs use the expected deck presets under the approx profile. |

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
