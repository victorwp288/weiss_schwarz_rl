# Evaluation

This page explains how policies are evaluated. Use it when the question is
"how do we know the trained checkpoint is better?"

## Evaluation Surfaces

There are three evaluation levels:

- smoke eval: fast plumbing check against a tiny fixed panel;
- periodic dev eval: training-time checkpoint signal;
- final eval: retained thesis-grade policy panel, paired seeds, payoff folding,
  uncertainty, metagame outputs, and paper-readiness checks.

Do not use smoke eval as model-quality evidence.

## Public Commands

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/main_thesis_seed1 --b1-run runs/b1_thesis_seed1
```

Dry-run payloads include `evidence_targets`, the retained files to inspect
after the command finishes.
`python/weiss_rl/workflows/evaluation_workflow/stages.py` names the shared
dry-run stage order for eval, figures, and B2 audit commands.

## Startup Handoff

`python/weiss_rl/workflows/eval_support/startup/eval_startup_prepare.py`
verifies the evaluation stack before dispatch:

| Step | Output |
| --- | --- |
| Load stack | `stack` |
| Verify config identity | `config_hash256` |
| Select spec source | `reported_spec_hash`, optional runtime `contract` |
| Announce startup | startup banner with hard-fail spec mismatch policy |

Public-demo eval uses the synthetic bundle. Retained eval loads the active
simulator contract before selecting policies or running matchups.

## Owner Files

| Concept | Files |
| --- | --- |
| Public eval workflow routing | `python/weiss_rl/workflows/command_surface.py`, `python/weiss_rl/workflows/workflow_route_explanation.py`, `python/weiss_rl/workflows/workflow_dispatch.py`, `python/weiss_rl/workflows/evaluation_workflow/`, `python/weiss_rl/workflows/evaluation_workflow/payloads.py` |
| Eval records and result schemas | `python/weiss_rl/eval/simulator/records.py` |
| Completed game records and episode identity | `python/weiss_rl/eval/simulator/completed_games.py` |
| Seat-swapped matchup schedule | `python/weiss_rl/eval/simulator/seat_swapped.py` |
| Terminal result decoding | `python/weiss_rl/eval/simulator/terminal_result.py`, `python/weiss_rl/eval/simulator/terminal_step.py` |
| Simulator-backed eval loop | `python/weiss_rl/eval/simulator/simulator_runner.py`, `python/weiss_rl/eval/simulator/simulator_game_lifecycle.py`, `python/weiss_rl/eval/simulator/harness.py` |
| Simulator replay capture | `python/weiss_rl/eval/replay/simulator_replay.py`, `python/weiss_rl/eval/replay/simulator_replay_capture.py`, `python/weiss_rl/replay/bundles.py` |
| Eval action selection | `python/weiss_rl/eval/simulator/simulator_action_selection.py`, `python/weiss_rl/eval/simulator/simulator_policy_step.py`, `python/weiss_rl/eval/search/simulator_god_search.py`, `python/weiss_rl/eval/search/simulator_god_search_rollouts.py`, `python/weiss_rl/eval/search/simulator_god_search_selection.py`, `python/weiss_rl/eval/search/simulator_god_search_outcomes.py` |
| Pinned eval sampling | `python/weiss_rl/eval/sampling/action_sampling.py`, `python/weiss_rl/eval/sampling/sampling_helpers.py` |
| Final eval orchestration | `python/weiss_rl/eval/final/run.py`, `python/weiss_rl/eval/final/run_plan.py`, `python/weiss_rl/eval/final/matchup_jobs.py`, `python/weiss_rl/eval/final_eval.py` |
| Final eval summary payload | `python/weiss_rl/eval/final/payload.py`, `python/weiss_rl/eval/final/payload_sections.py`, `python/weiss_rl/eval/final/matrices.py` |
| Final eval artifact writers | `python/weiss_rl/eval/final/artifacts.py`, `python/weiss_rl/eval/final/matrix_artifacts.py`, `python/weiss_rl/eval/final/matchup_manifest.py`, `python/weiss_rl/eval/final/matchup_outputs.py`, `python/weiss_rl/eval/final/run_diagnostics.py` |
| Policy panel IDs and deck mapping | `python/weiss_rl/eval/policies/fixed_panel.py` |
| Policy set resolution | `python/weiss_rl/eval/policies/set.py`, `python/weiss_rl/eval/policies/focal_recommendation.py`, `python/weiss_rl/eval/policies/registry_view.py`, `python/weiss_rl/eval/policies/dev_eval_summaries.py`, `python/weiss_rl/eval/policies/training_policy_ids.py`, `python/weiss_rl/eval/policies/resolution.py`, `python/weiss_rl/eval/snapshots/snapshot_registry_resolution.py` |
| Policy alignment and replay distribution metrics | `python/weiss_rl/eval/policies/alignment.py`, `python/weiss_rl/replay/inspection_step_diffs.py`, `python/weiss_rl/core/action_distribution_metrics.py` |
| B2 disagreement audit | `python/weiss_rl/diagnostics/b2_audit/b2_disagreement_audit.py`, `python/weiss_rl/diagnostics/b2_audit/b2_audit_source.py`, `python/weiss_rl/diagnostics/b2_audit/b2_audit_reports.py`, `python/weiss_rl/diagnostics/b2_audit/b2_audit_aggregation.py`, `python/weiss_rl/diagnostics/b2_audit/b2_audit_summary_math.py` |
| Trajectory policy drift | `python/weiss_rl/diagnostics/trajectory/trajectory_policy_drift.py`, `python/weiss_rl/diagnostics/trajectory/trajectory_policy_drift_stats.py` |
| Parallel final eval | `python/weiss_rl/eval/parallel/parallel_final_eval_plan.py`, `python/weiss_rl/eval/parallel/parallel_final_eval_core.py` |
| Payoff folding and uncertainty | `python/weiss_rl/eval/analysis/payoff_folding.py`, `python/weiss_rl/eval/analysis/uncertainty.py`, `python/weiss_rl/eval/analysis/stage2.py` |
| Metagame sensitivity reports | `python/weiss_rl/metagame/sensitivity.py`, `python/weiss_rl/metagame/sensitivity_inputs.py`, `python/weiss_rl/metagame/sensitivity_outputs.py` |
| Readiness checks | `python/weiss_rl/eval/paper_readiness.py`, `python/weiss_rl/eval/readiness/check_runtime.py`, `python/weiss_rl/eval/readiness/check_cli.py` |
| Readiness artifact inventory | `python/weiss_rl/eval/readiness/specs.py`, `python/weiss_rl/eval/readiness/run_directory_contract.py` |
| Readiness manifest contract | `python/weiss_rl/eval/readiness/manifest_contract.py`, `python/weiss_rl/eval/readiness/contracts.py` |
| Final-eval artifact contract | `python/weiss_rl/eval/readiness/final_eval_contract.py`, `python/weiss_rl/eval/readiness/final_eval_matchup_contract.py`, `python/weiss_rl/eval/readiness/final_eval_summary.py` |
| Sensitivity artifact contract | `python/weiss_rl/eval/readiness/sensitivity_contract.py` |
| Final-eval guardrails | `python/weiss_rl/eval/readiness/guardrails.py`, `python/weiss_rl/eval/readiness/baseline_guardrail.py` |

## Policy Panel

The thesis panel is deliberately fixed and named. It includes the focal model,
B0/B1/B2 guardrail policies, and B3/B4 deck-style opponents. The purpose is to
separate real improvement from overfitting to one narrow matchup.
`python/weiss_rl/eval/policies/fixed_panel.py` owns the B0-B4 role map:
policy ID, deck binding, policy source, and the evidence question each anchor
answers.

B1 resolution is explicit. The main run should point at the selected
`b1_noleague_baseline` source instead of resolving a chronological latest run.

`python/weiss_rl/eval/final/run_plan.py` owns the final-eval plan: selected
policy IDs, the selection payload recorded in artifacts, and the canonical
upper-triangle matchup jobs. `run.py` executes that plan, builds the summary
payload, and writes artifacts.

`python/weiss_rl/eval/final/policy_selection.py` documents the two selection
paths: an explicit policy panel from `policy_ids`, or deterministic artifact
selection from the snapshot registry, dev-eval summaries, selection config, and
target panel size. Final-eval metadata includes a `selection_trace` entry for
each selected policy, recording whether it came from an explicit request, a
fixed baseline, a champion snapshot, a spaced snapshot, or the dev-eval ranking.

`python/weiss_rl/eval/final/artifacts.py` owns the final-eval write plan:
core JSON, posterior samples, matrix exports, matchup manifest, aggregate
episodes, canonical diagnostics, and artifact hashes.
The write plan is payloadable through `final_eval_artifact_write_plan_payload()`
and lists the output paths created by each stage.
`summary.json` includes `summary_sections`, a compact reader map for the policy
selection, matchup-artifact convention, payoff matrices, posterior samples, and
per-matchup evidence paths.
`python/weiss_rl/workflows/eval_support/reports/eval_report_update_payloads.py`
mirrors a compact `canonical_eval_evidence` block into the run summary and
determinism report so the retained run directory points back to its final-eval
proof files.

## Paired Seeds

Final eval uses paired seeds so first/second-seat and stochastic effects are
balanced. Paired-seed folding is part of the contract: changing seed pairing,
policy order, or payoff folding changes the meaning of the reported score.

## Payoff Folding

Raw game records are folded into policy-vs-policy payoff summaries. The folding
scheme controls how paired games become one comparison unit. The retained
schemes exist so robustness and uncertainty can be reported without pretending
that every single game row is independent evidence.

## Uncertainty

Uncertainty summaries use paired-seed scores. This keeps confidence intervals
aligned with the experimental unit used by final evaluation.

## Readiness

Paper-readiness checks make sure a run tree contains the expected outputs:

- manifest and config identity;
- training logs and metrics;
- final eval records and summaries;
- metagame outputs;
- replay or diagnostic artifacts;
- figure outputs.

The inventory is grouped in
`python/weiss_rl/eval/readiness/specs.py`. The groups answer the first review
questions before the checker gets into individual files:
The final-eval artifact contract also reports `summary_section_keys` when
`summary.json` includes reader sections.

| Group | What It Proves |
| --- | --- |
| Run identity | The run, config, environment, and determinism metadata are pinned. |
| Training evidence | The selected policy has either training metrics or interpolation provenance. |
| Final evaluation | The retained policy panel, matchup schedule, payoff matrix, posterior samples, and hashes exist. |
| Diagnostics | Seat bias, truncation, replay, and related checks were written. |
| Metagame sensitivity | S0-S2 payoff, Nash, and AlphaRank robustness outputs exist. |
| Paper figures | Rendered figure outputs are tied to the same run tree. |

The emitted `paper_readiness_summary.json` also includes a `section_plan`
field for the top-level checks: run-directory artifacts, manifest contract,
final-eval artifact contract, and final-eval guardrails.

The readiness fixture is synthetic and lives under
`runs/paper_readiness_fixture_ci` when generated by the artifact contract. Smoke
output is not a substitute for a paper-grade run.

## What To Check Before Claiming A Result

Before treating a checkpoint as thesis evidence, confirm:

- the policy IDs and B1 anchor are the intended ones;
- paired seed count and folding scheme match the final-eval contract;
- B0-B4 guardrails are present;
- uncertainty summaries are generated;
- readiness checks pass;
- the result is described as scoped to its policy panel, deck set, and seed
  budget.
