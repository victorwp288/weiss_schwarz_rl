# Architecture

This repository is a behavior-sensitive thesis RL pipeline. The code is
organized around explicit contracts: config hashes, simulator spec bundles,
run-tree layouts, seed files, checkpoint schemas, and deterministic evaluation
outputs.

## Pipeline

1. `python -m weiss_rl.cli` selects a standard thesis command.
2. `weiss_rl.training.train_entrypoint` validates config/simulator inputs,
   writes the run manifest, builds the learner, and starts training.
3. `weiss_rl.runtime.QueueRuntime` collects simulator-backed decision-boundary
   rollouts and assembles learner batches.
4. `weiss_rl.learners` updates policy/value models and writes metrics,
   checkpoints, snapshots, and TensorBoard logs.
5. `weiss_rl.workflows.eval_entrypoint` resolves deterministic policy sets,
   runs pinned final evaluation, and writes reporting artifacts.

## Package Map

| Package | Owns |
| --- | --- |
| `weiss_rl.cli` and `weiss_rl.workflows` | Public thesis command surface, profiles, dry-run plans, verification, and command routing. |
| `weiss_rl.config` | Strict config parsing, preset inheritance, overrides, canonical hashes, and seed-file resolution. |
| `weiss_rl.core` and `weiss_rl.envs` | Simulator-facing domain contracts, legal-action batches, observation layout, action catalog decoding, and environment wrappers. |
| `weiss_rl.runtime` | Queue runtime, actor collection, learner-batch assembly, opponent sampling, process collectors, IPC/shared transport, and runtime metrics. |
| `weiss_rl.learners` | IMPALA/V-trace, PPO-lite, learner batch validation, losses, action log-probability helpers, and structured auxiliary metrics. |
| `weiss_rl.models` and `weiss_rl.model` | Policy/value model internals plus the compatibility facade used by training and evaluation. |
| `weiss_rl.eval` | Deterministic evaluation, policy resolution, payoff folding, uncertainty, diagnostics, and paper-readiness reporting. |
| `weiss_rl.league` | Snapshot registry, PFSP sampling, promotion gates, opponent pools, and online outcomes. |
| `weiss_rl.artifacts` and `weiss_rl.diagnostics` | Run artifact layout, manifests, hygiene checks, telemetry, training-log helpers, and diagnostic entrypoints. |

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

Files still at the top of `runtime/components/` are central runtime surfaces or
cross-cutting helpers. Move them only with adjacent tests and import-path
cleanup.

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

## Retired Script Surface

The package CLI is the standard thesis surface. The old path-based
`python/scripts/*` wrappers and dated campaign, rescue, sweep, guided-bootstrap,
and paired-probe scripts were removed from the active checkout.

Use package modules under `weiss_rl.*` for lower-level diagnostics. Future
extractions should preserve public commands and keep compatibility shims only
where a live caller still needs them.
