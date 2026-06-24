# Training

This page explains how a policy is trained and where the training code lives.
Use it when the question is "what happens after I run `train-b1` or
`train-main`?"

## Public Entry Points

Training starts through `python -m weiss_rl.cli`:

- `train-b1` trains the no-league B1 baseline.
- `train-main` trains the main league policy from an explicit B1 anchor.

The CLI is intentionally the public surface. Lower-level modules are importable
for tests and diagnostics, but thesis runs should use the public commands.
Dry-run payloads include `evidence_targets`, the retained files to inspect
after the command finishes.

## Training Stages

1. The CLI resolves the profile, config, run label, and output directory.
2. The config loader expands the thesis YAML stack and computes canonical
   config identity.
3. The simulator contract is checked before training starts.
4. The run manifest, seed files, and runtime settings are written.
5. The runtime collects simulator-backed decision-boundary rollouts.
6. The learner consumes batches, updates the model, and writes metrics.
7. Checkpoint and promotion logic publishes retained snapshots.
8. Periodic dev eval and promotion gates decide whether a checkpoint is worth
   keeping.

## Startup Handoff

`python/weiss_rl/training/train_entrypoint/startup_state.py` owns the first
hard boundary after CLI parsing. Its route map names five steps:

| Step | Output |
| --- | --- |
| Select spec source | `spec_bundle`, `spec_hash256`, `simulator_info` |
| Verify config identity | `config_hash256` |
| Capture run provenance | `git_commit`, `start_nonce` |
| Assign run identity | `run_id256`, `run_id64`, `run_dir_name`, `resume_artifacts` |
| Announce startup | startup banner and spec/config summary |

Public-demo runs stop at the synthetic bundle. Retained runs load the active
simulator contract before any run directory is materialized.

## Owner Files

| Concept | Files |
| --- | --- |
| Public command surface | `python/weiss_rl/workflows/command_surface.py`, `python/weiss_rl/workflows/command_help.py`, `python/weiss_rl/workflows/workflow_route_explanation.py`, `python/weiss_rl/workflows/parser_argument_helpers.py`, `python/weiss_rl/workflows/training_workflow/` |
| Training config schema and parser | `python/weiss_rl/config/schemas/training_models.py`, `python/weiss_rl/config/schemas/training_aux_models.py`, `python/weiss_rl/config/schemas/training_compat_accessors.py`, `python/weiss_rl/config/sections/sections_training_schema.py`, `python/weiss_rl/config/sections/sections_training_sections.py`, `python/weiss_rl/config/sections/sections_training.py`, `python/weiss_rl/config/sections/sections_training_structured_aux.py`, `python/weiss_rl/config/sections/sections_training_replay_aux.py`, `python/weiss_rl/config/sections/sections_training_public_teacher.py`, `python/weiss_rl/config/sections/sections_training_aux_helpers.py` |
| Training CLI state | `python/weiss_rl/training/train_entrypoint/cli_state.py` |
| Simulator and run identity checks | `python/weiss_rl/training/train_entrypoint/startup_state.py` |
| Manifest and run metadata | `python/weiss_rl/training/train_entrypoint/manifest_state.py` |
| Training execution dispatch | `python/weiss_rl/training/train_entrypoint/run_execution.py`, `python/weiss_rl/training/train_entrypoint/training_run_preflight.py` |
| Entrypoint lifecycle wrappers | `python/weiss_rl/training/train_entrypoint/lifecycle.py`, `python/weiss_rl/training/train_entrypoint/lifecycle_checkpoint_wrappers.py`, `python/weiss_rl/training/train_entrypoint/lifecycle_training_wrapper.py` |
| Runtime collection | `python/weiss_rl/runtime/queue_runtime.py`, `python/weiss_rl/runtime/components/process.py`, `python/weiss_rl/runtime/components/process_child_config.py`, `python/weiss_rl/runtime/components/` |
| Runtime opponent pool | `python/weiss_rl/runtime/components/opponent_mixin.py`, `python/weiss_rl/runtime/components/opponent_pool_refresh_log.py`, `python/weiss_rl/runtime/components/opponents/`, `python/weiss_rl/runtime/components/opponents/residency.py` |
| Runtime action-surface guards | `python/weiss_rl/runtime/components/actions/action_surface.py`, `python/weiss_rl/runtime/components/actions/action_surface_mulligan_guard.py`, `python/weiss_rl/runtime/components/actions/action_surface_pass_guards.py`, `python/weiss_rl/runtime/components/actions/action_surface_packed.py`, `python/weiss_rl/eval/sampling/model_action_surface.py` |
| Runtime rewards and discounts | `python/weiss_rl/runtime/components/rewards/reward_shaping.py`, `python/weiss_rl/runtime/components/rewards/reward_shaping_counters.py`, `python/weiss_rl/runtime/components/batching/reward_backfill.py`, `python/weiss_rl/runtime/components/batching/bootstrap_values.py` |
| One learner update | `python/weiss_rl/training/loop/update.py`, `python/weiss_rl/training/loop/update_step.py`, `python/weiss_rl/training/loop/update_stage_pipeline.py`, `python/weiss_rl/training/loop/update_schedule.py`, `python/weiss_rl/training/loop/update_batch.py`, `python/weiss_rl/training/loop/update_completion.py` |
| Post-update checkpoint and dev eval | `python/weiss_rl/training/loop/post_update.py`, `python/weiss_rl/training/loop/loop_progress.py`, `python/weiss_rl/training/checkpointing/guards/snapshot_promotion.py`, `python/weiss_rl/training/checkpointing/guards/periodic_dev_eval.py` |
| Checkpoint promotion plan | `python/weiss_rl/training/checkpointing/guards/snapshot_promotion.py`, `python/weiss_rl/training/checkpointing/aliases/alias_publication.py`, `python/weiss_rl/training/checkpointing/lifecycle/tracker.py`, `python/weiss_rl/training/checkpointing/guards/guard.py` |
| Learner construction and batch support | `python/weiss_rl/training/learner_factory.py`, `python/weiss_rl/learners/impala/learner.py`, `python/weiss_rl/learners/impala/batching/batch_support.py`, `python/weiss_rl/learners/impala/batching/batch_field_support.py`, `python/weiss_rl/learners/packed_forward_metrics.py`, `python/weiss_rl/learners/` |
| Replay trajectory BC datasets | `python/weiss_rl/replay/trajectory_bc.py`, `python/weiss_rl/replay/trajectory_bc_dataset_schema.py`, `python/weiss_rl/replay/trajectory_bc_dataset.py`, `python/weiss_rl/replay/trajectory_bc_batching.py`, `python/weiss_rl/training/replay_data/` |
| Checkpoints and snapshots | `python/weiss_rl/training/checkpointing/`, `python/weiss_rl/training/train_entrypoint/checkpoints.py`, `python/weiss_rl/training/train_entrypoint/checkpoint_io_hooks.py`, `python/weiss_rl/training/train_entrypoint/checkpoint_requests.py`, `python/weiss_rl/training/train_entrypoint/snapshot_hooks.py`, `python/weiss_rl/training/train_entrypoint/snapshots.py` |
| Checkpoint guard decisions | `python/weiss_rl/training/checkpointing/guards/dev_eval_metrics.py`, `python/weiss_rl/training/checkpointing/guards/guard.py`, `python/weiss_rl/training/checkpointing/guards/guard_events.py`, `python/weiss_rl/training/checkpointing/lifecycle/lifecycle_decisions.py` |
| Learner and warmstart hooks | `python/weiss_rl/training/train_entrypoint/learner_hooks.py`, `python/weiss_rl/training/learner_factory.py`, `python/weiss_rl/learners/` |
| Periodic dev eval | `python/weiss_rl/training/dev_eval/`, `python/weiss_rl/training/dev_eval/matchup_artifacts.py`, `python/weiss_rl/training/dev_eval/policy_alignment.py`, `python/weiss_rl/training/periodic_dev_eval_run.py`, `python/weiss_rl/training/train_entrypoint/dev_eval_wrappers.py` |
| Promotion gates | `python/weiss_rl/training/promotion_gate_runner.py` |
| Training log diagnostics | `python/weiss_rl/diagnostics/progress/learning_progress_sections.py`, `python/weiss_rl/diagnostics/progress/learning_progress_metrics.py`, `python/weiss_rl/diagnostics/progress/learning_progress_teacher_guidance.py`, `python/weiss_rl/diagnostics/progress/learning_progress_sync.py`, `python/weiss_rl/diagnostics/progress/learning_progress_math.py`, `python/weiss_rl/diagnostics/progress/learning_progress_warnings.py` |

`python/weiss_rl/workflows/command_surface.py` owns each public workflow
command's purpose, evidence role, inputs, outputs, and next step. The parser
renders that metadata in `--help`, so the command registry is the source of
truth for the public workflow surface.
`python/weiss_rl/workflows/workflow_route_explanation.py` names the dispatch
target and plan builder for each public command.

Inside the package entrypoint, `cli_state.py` resolves CLI intent,
`startup_state.py` verifies identity and simulator contracts, `manifest_state.py`
writes run metadata, and `run_execution.py` starts the selected training path.
`training_run_preflight.py` keeps the final runtime prerequisite checks and
execution-setting resolution separate from the actual trainer call.
When evaluation inputs are available, `python/weiss_rl/training/policy_selection.py`
records the deterministic policy-set selection and `selection_trace` in the run
manifest so each retained policy has an auditable reason.
`python/weiss_rl/training/report_payloads.py` mirrors the compact policy
selection status into `run_summary.json` and `determinism.json`; use those files
for the quick "what did this run select?" answer, and the manifest trace for the
full evidence.

## B1 Training

B1 is the clean no-league baseline. It uses the same simulator and model family
as the main run, but it does not sample from the online league. Its job is to
produce a stable baseline anchor that can later be used by main training and
final evaluation.

The retained B1 run must be chosen explicitly. `train-main --b1-run` should
point at the intended B1 run directory rather than relying on whichever run is
chronologically newest.

## Training Profiles

Workflow profiles live in
`python/weiss_rl/workflows/training_workflow/profiles.py`. Each profile declares
its purpose, environment count, unroll length, update count, runtime mode,
simulator profile, device, checkpoint interval, and config overrides. The public
commands use these profiles to turn names like `smoke`, `gpu-probe`, and
`thesis-local` into concrete training entrypoint arguments.
`python/weiss_rl/config/sections/sections_training_sections.py` names the
nested training config sections and validates their keys, keeping rollout,
optimizer, V-trace, teacher auxiliary, and action-surface controls in explicit
groups.

## Main Training

Main training starts from a selected B1 anchor and then trains against the
league policy set. The league can include fixed heuristic opponents, imported
snapshots, recent learner checkpoints, and hard-negative opponents. Promotion is
not raw score chasing: checkpoint confidence, guardrails, seed coverage, and
regression checks decide which snapshots are retained.

## Runtime Shape

The runtime collects trajectories at decision boundaries. A learner row is a
state where the focal policy has a legal decision to make. Opponent-internal
steps can happen between focal decisions, but the learner batch is assembled
around focal decision rows.

The important runtime contracts are:

- observations keep the simulator spec layout;
- legal actions preserve simulator action IDs and packed candidate metadata;
- rewards and discounts are aligned to learner rows;
- terminal and timeout rows are not double counted;
- hidden state is tracked per seat, not only per environment.

## Learning Signals

The learner sees the policy action, selected log probability, value prediction,
reward, discount, bootstrap value, legal-action information, and optional
structured supervision. IMPALA/V-trace is the main thesis learner path; PPO-lite
is retained as an ablation.
`python/weiss_rl/learners/impala/losses/loss_plan.py` names the learner objective
components: V-trace targets, policy gradient, value regression, entropy,
trajectory retention, policy anchor, teacher auxiliary terms, and structured
metrics.

## Update Step

One learner update follows the stage plan in
`python/weiss_rl/training/loop/update_stage_pipeline.py`:

- schedule guidance and entropy for the next update;
- collect a runtime learner batch;
- apply the learner update;
- run optional post-update replay phases;
- merge runtime, schedule, replay, and snapshot metrics.

After each update, `python/weiss_rl/training/loop/post_update.py` runs the
checkpoint stage before the periodic dev-eval/checkpoint-guard stage. The
ordering is explicit because dev eval may need the current checkpoint to exist.
`python/weiss_rl/training/dev_eval/plan.py` names the dev-eval stages that show
up in each update summary: contract validation, eval-model snapshotting, anchor
panel resolution, matchup artifact writing, quality aggregation, and diagnostics.

When a checkpoint interval lands,
`python/weiss_rl/training/checkpointing/guards/snapshot_promotion.py` follows its
promotion plan: write the checkpoint and aliases, persist the snapshot registry
candidate, then refresh opponents and run the promotion gate.
`python/weiss_rl/training/checkpointing/lifecycle/lifecycle_plans.py` names the
guard decision questions used before rollback or finalization side effects run.
`python/weiss_rl/training/promotion.py` can trace promotion-anchor resolution so
required, optional, symbolic, built-in, and missing anchors are auditable.

At training shutdown,
`python/weiss_rl/training/checkpointing/lifecycle/finalization.py` publishes
the current checkpoint aliases first, then records whether the final tracker
came from those aliases or from a guard-selected best checkpoint reload.

## What To Check When Training Looks Good

Stable training does not automatically mean better play. Keep these checks
separate:

- throughput: how fast samples are collected;
- stability: whether losses, values, and checkpoints stay finite;
- learning quality: whether selected checkpoints improve against the policy
  panel;
- transfer: whether gains hold against B2 and other harder public opponents;
- readiness: whether the retained run tree satisfies the artifact contract.

## Fast Smoke Versus Thesis Evidence

Smoke profiles are plumbing checks. They confirm that configs, simulator import,
runtime wiring, and artifact paths work. They do not establish model quality.

Thesis evidence comes from named runs with retained manifests, logs, checkpoints,
final eval outputs, diagnostics, and artifact-contract validation.
