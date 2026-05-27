# B1 Learning Rebuild Report

Date: 2026-05-17

Repo: `C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl`

Scope: B1 NoLeague learning-quality investigation, structural repair, checkpoint selection, and final paper-ready B0-B4 evaluation.

This report summarizes the B1 rebuild arc. The raw chronological record remains in `docs/rebuild_log.md`; this document is the curated "what changed, what helped, what failed, and what to do next" version.

## Current Status

B1 NoLeague is no longer the active blocker.

The current thesis-usable B1 artifact is:

- run: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- selected policy id: `selected_candidate`
- selected source policy id: `policy_000003`
- selected update: `15`
- chronological latest in the same run: `policy_000004`, update `20`
- selected weights hash:
  `66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685`

The important semantics are now correct:

- `latest` remains chronological latest.
- `selected_candidate` is the best-confirmed checkpoint.
- The selected checkpoint is not silently replaced by latest after training.

Final paper-ready eval was completed with:

```powershell
$env:PYTHONHASHSEED='0'; uv run --extra dev --extra sim python python/scripts/eval.py --stack-config configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml --run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 --snapshot-registry-json runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/training/snapshots/registry.json --b1-baseline-run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 --policy-id selected_candidate --policy-id "B0 RandomLegal" --policy-id "B2 HeuristicPublic" --policy-id "B3 HeuristicPublicAggro" --policy-id "B4 HeuristicPublicControl" --paired-seed-limit 256 --stage1-paired-seeds 256 --max-paired-seeds 256 --bootstrap-samples 4000
```

Final selected B1 row:

| Opponent | Wins | Games | Win rate | 95% CI | Posterior P(p > 0.5) |
|---|---:|---:|---:|---|---:|
| B0 RandomLegal | 512 | 512 | 1.000000 | [1.000000, 1.000000] | 1.0 |
| B2 HeuristicPublic | 341 | 512 | 0.666016 | [0.627164, 0.704415] | 1.0 |
| B3 HeuristicPublicAggro | 330 | 512 | 0.644531 | [0.603244, 0.685072] | 1.0 |
| B4 HeuristicPublicControl | 321 | 512 | 0.626953 | [0.584267, 0.670526] | 1.0 |

Artifacts:

- final eval summary: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/eval/final_eval/summary.json`
- final eval matrices: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/eval/final_eval/matrices/`
- policy set: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/eval/final_eval/policy_set.json`
- metagame summary: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/eval/metagame/summary.json`
- replay verification: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/eval/diagnostics/replay_verification.json`
- paper readiness: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/paper_readiness_summary.json`
- paper figures: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/figures/paper/`

Paper readiness passed:

- `passed: true`
- `alarms: []`
- final-eval artifact contract: passed
- final-eval guardrails: passed
- replay verification: `7/7` sampled episodes verified, `0` failed
- figures rendered:
  - `fig_learning_curves.{pdf,png}`
  - `fig_matchup_heatmap.{pdf,png}`
  - `fig_seat_bias.{pdf,png}`
  - `fig_truncation_heatmap.{pdf,png}`

Focused validation at the end:

```powershell
uv run --extra dev ruff check python/weiss_rl/replay/trajectory_bc.py python/scripts/merge_replay_trajectory_bc_datasets.py python/weiss_rl/training/checkpoint_interpolation.py python/scripts/interpolate_checkpoints.py python/weiss_rl/experiments/b1_candidate_selection.py python/scripts/select_b1_candidate.py python/weiss_rl/tests/test_replay_trajectory_bc.py python/weiss_rl/tests/test_checkpoint_interpolation.py python/weiss_rl/tests/test_b1_candidate_selection.py python/weiss_rl/tests/test_config_loader.py

uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_replay_trajectory_bc.py python/weiss_rl/tests/test_checkpoint_interpolation.py python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_direct_b2b3b4_trajbc_anchor_nopublic_continuation python/weiss_rl/tests/test_b1_candidate_selection.py
```

Results:

- ruff: passed
- pytest: `21 passed`

## Original Problem

The starting concern was not just "hyperparameters might be bad." The evidence suggested structural or workflow problems:

- Throughput was acceptable but learning did not improve cleanly.
- Scalar losses and teacher proxy metrics improved while policy quality did not improve monotonically.
- A prior B1 run, `b1_medium64_thesis_local_20260512`, reached update 181 in scalar logs but only had durable checkpoint/snapshot coverage to update 175.
- In that run, training proxies improved:
  - first 20 mean loss around `5.53848`
  - last 20 near pause around `3.63941`
  - teacher family accuracy around `0.477 -> 0.554`
  - teacher slot accuracy around `0.199 -> 0.259`
  - teacher action accuracy mostly flat around `0.36-0.37`
- But paired checkpoint eval did not show monotonic improvement:
  - update 100 vs B2/B3/B4: `10/16`, `5/16`, `8/16`
  - update 150 vs B2/B3/B4: `8/16`, `7/16`, `6/16`
  - update 175 vs B2/B3/B4: `10/16`, `7/16`, `7/16`
  - update 100 beat update 150 by `13/16`
  - update 100 beat update 175 by `10/16`
  - update 175 beat update 150 by `9/16`

That made "train longer" unsafe. The working hypothesis became:

- proxies were not enough;
- best/latest semantics were broken or underdeveloped;
- B1 was likely contaminated by inherited teacher or public heuristic behavior;
- reward/value perspective could be wrong;
- eval/action scoring could be mismatched;
- learning might be collapsing into superficial or brittle action patterns.

## Fixed Policy Constraints

The fixed deck policy was preserved:

- focal/B0/B1/B2/main: `preset:main_deck_5hy_yotsuba_v1`
- B3 aggro: `preset:aggro_deck_5hy_nino_v1`
- B4 control: `preset:control_deck_jj_s66_v1`

The simulator was treated as the gold-standard foundation. We inspected simulator contracts where needed, but did not edit the simulator repo.

`PYTHONHASHSEED=0` is required for reproducible model-load/train/eval diagnostics in this repo until the salted trait hashing issue is removed or fully isolated.

## Major Code And Workflow Changes

This is the curated list of changes that mattered most for B1. See `docs/rebuild_log.md` for every command and every intermediate run.

### 1. Standard Thesis Workflow And Fixed-Deck Surface

We built up the standard thesis CLI/config path around:

- `python/weiss_rl/cli.py`
- `python/scripts/train.py`
- `python/scripts/eval.py`
- `python/scripts/thesis_workflow.py`
- `configs/thesis/*.yaml`
- `docs/thesis_workflow.md`
- `docs/artifact_contract.md`
- `docs/rebuild_log.md`

What this helped:

- made B1 train/eval commands reproducible;
- made fixed deck policy explicit;
- made B0-B4 eval policy resolution testable;
- reduced hidden flag soup;
- gave final eval, metagame, figures, and readiness a canonical artifact layout.

### 2. Medium64 Model Surface

The tiny32 surface was not adequate for the thesis B1 path. We standardized B1/main/final-eval compatibility around medium64:

- `gru_hidden_size: 64`
- `encoder_mlp_width: 64`
- `typed_feature_width: 16`

What helped:

- fixed a real model-load mismatch where final eval inherited tiny32 and could not load medium64 checkpoints;
- kept local throughput stable enough for experiments.

What did not help by itself:

- medium64 alone did not solve non-monotonic learning;
- scalar proxy improvements still failed to guarantee policy-quality improvement.

Rejected path:

- 96/128-wide probes were not suitable for local long runs because they approached the RTX 5080 VRAM ceiling around `15.5-15.8 GB`.

### 3. Clean B1 NoLeague Route

We repaired B1 so it is actually a clean NoLeague model-actor RL path:

- B1 `league.enabled=false`
- B1/main `actor_policy_backend=model`
- B1/main `actor_heuristic_fraction=0.0`
- B1/main `teacher_public_heuristic_coef=0.0`
- B1 `heuristic_public_mix_fraction=0.0`
- teacher auxiliary made off or warmstart-only depending on the experiment, then removed for clean B1
- startup guards added so canonical B1 rejects inherited heuristic actor or teacher/warmstart contamination

What helped:

- removed a major source of misleading proxy learning;
- made B1 evaluation mean "the model policy learned this" rather than "the actor or teacher leaked heuristic behavior";
- forced policy quality to be measured by eval, not teacher imitation metrics.

What did not help by itself:

- clean route initially had weak B2 performance and lower throughput;
- removing contamination exposed the real learning problem rather than solving it.

### 4. Value/Reward Perspective Repair

We audited reward/value perspective and fixed the B1 value route:

- simulator rewards are actor-seat perspective;
- learner value bootstrapping can accidentally mix perspectives when turn ownership alternates;
- the learner now handles actor-perspective rewards and alternating actor-perspective values correctly;
- timeout/truncation discount semantics were tested;
- hidden state reset behavior was guarded by tests.

What helped:

- removed a structural reason for unstable targets;
- made reward/advantage scale diagnostics trustworthy;
- reduced the chance that "learning stalls" was caused by sign/perspective corruption.

What did not help by itself:

- fixing perspective made the learning signal cleaner, but B2 transfer still needed stronger behavioral supervision and checkpoint selection.

### 5. Packed Candidate Scoring And Eval Surface Repair

We changed structured candidate scoring to use the packed-plan scoring path instead of dense candidate-mask scoring.

Key area:

- model/eval candidate scoring path around structured model sampling and packed legal-candidate scoring

What helped:

- aligned train/eval legal-candidate scoring more tightly;
- reduced packed scoring overhead;
- made eval log-prob/action-surface behavior less suspicious;
- gave us confidence that legal masking/candidate ordering were not the main remaining issue.

What did not help by itself:

- B2 score in early clean probes remained weak, so the issue was not just packed scoring.

### 6. Learning Progress Diagnostics

We added diagnostics so "is it learning?" stopped being a manual TensorBoard guess:

- `python/scripts/learning_progress_diagnostic.py`
- `python/scripts/learning_run_compare.py`
- reward/advantage scale metrics in learner logging
- final-eval matrix summaries
- warnings when best and latest diverge
- latest-minus-best and non-monotonic drop reporting
- per-anchor summaries for B0/B2/B3/B4

What helped:

- made non-monotonicity visible;
- showed full shaping could improve local dev eval without solving B2;
- showed some runs peaked early and then declined;
- made the "update 15 beats update 20/latest" pattern explicit.

### 7. Reward Shaping Diagnostics And Ablations

We audited simulator reward components without editing the simulator:

- `python/scripts/reward_component_probe.py`
- reward ablation configs under `configs/thesis/ablations/`
- `terminal_only_reward.yaml`
- `damage_only_reward.yaml`
- `damage_level_reward.yaml`
- `full_shaping_reward.yaml`
- later variants with actor sync and entropy changes

Findings:

- `weiss-sim 1.1.0` exposes debug reward components in fixed order.
- Current damage-only reward geometry had reward abs mean around `0.005695`.
- Full shaping increased reward abs mean to around `0.010161` in the debug probe.
- In 25-update probes, fuller shaping improved aggregate dev eval:
  - terminal only: aggregate dev around `0.3906`
  - damage only: aggregate dev around `0.4219`
  - damage+level: aggregate dev around `0.4531`
  - full shaping: aggregate dev around `0.4844`

What helped:

- full shaping was clearly better than terminal-only or damage-only as a local B1 signal;
- reward scale diagnostics made ablation comparison safer;
- full shaping became part of later stronger guided/bootstrap work.

What did not help:

- reward shaping alone did not solve B2;
- 100-update reward probes remained non-monotonic;
- B2 remained weak in absolute terms in pure reward-only experiments;
- actor sync and lower entropy variants did not remove the structural plateau.

Conclusion:

Reward shaping was necessary cleanup, not the whole solution.

### 8. Best/Latest Checkpoint Semantics

We fixed and then repeatedly relied on the distinction between:

- chronological latest;
- selected best;
- published aliases.

Important behavior:

- `latest.pt` must remain latest.
- `best.pt` or `selected_candidate` must represent selected quality.
- downstream eval must not assume the latest checkpoint is best.

What helped:

- prevented later checkpoints from overwriting stronger earlier policies;
- explained why training appeared to "stop improving" after update 50 or 100;
- turned non-monotonic learning from an invisible failure into a manageable selection problem.

What did not help:

- selection alone cannot make a bad run good;
- it only becomes powerful once candidate generation and confirmation evidence are good.

### 9. B2/B4 Action Diagnostics And Guard Experiments

We investigated action-level behavior:

- B2 disagreement audit surface
- replay inspector fixes
- exact-action diagnosis
- eval stochastic vs argmax split
- temperature calibration
- mulligan surface guard
- main-move guard
- attack-pass collapse diagnosis
- public anti-pass loss
- sharp-teacher and exact-target variants
- same-family margin probes
- per-family B2 audit diagnostics

What helped:

- showed the model often failed on specific public tactical choices, not just value estimates;
- exposed pass/attack/main-move collapse patterns;
- made it clear that the model needed better action-conditioned behavior, not just more reward;
- helped motivate trajectory and state-matched replay approaches.

What did not help enough:

- guards and anti-pass losses could improve local symptoms but were brittle;
- exact-target and teacher-heavy variants often overfit or drifted late;
- temperature/argmax calibration changed the eval surface but did not create a robust B1 by itself;
- some candidates beat local checks but did not survive multi-anchor B2/B3/B4 confirmation.

### 10. Guided Bootstrap And League-Pivot Experiments

We tried guided/bootstrap paths before the final B1 was solved:

- strict B1 sync/temperature probes;
- guided-seed candidates;
- guided-factorized league pivot;
- clean continuation reproduction;
- B4 audits;
- observed-best alias repair;
- teacherfade candidates;
- rollback early-stop guards;
- profile-floor and public-floor probes;
- live-league and mirror-lane ideas;
- reset-offset and control-profile guarded transfer probes.

What helped:

- showed that pure B1 NoLeague was not the only useful path;
- produced stronger candidates than pure reward-only B1;
- localized recurring B4 regressions;
- motivated guardrails around rollback, reference drop, and required anchor floors.

What did not help enough:

- several guided/league candidates improved B2/B3 but regressed B4;
- profile-floor/public-floor and live-league style changes were not sufficient as simple fixes;
- B4 remained the hardest anchor to preserve until trajectory data was targeted more directly.

### 11. Trajectory Drift, Retention, And Replay BC

This became the decisive area.

Added or used tooling:

- `python/scripts/trajectory_audit_compare.py`
- `python/scripts/trajectory_policy_drift.py`
- `python/scripts/replay_trajectory_bc_dataset.py`
- `python/scripts/trajectory_bc_warmstart.py`
- `python/scripts/merge_replay_trajectory_bc_datasets.py`
- `python/weiss_rl/replay/trajectory_bc.py`
- tests in `python/weiss_rl/tests/test_replay_trajectory_bc.py`

What we found:

- B4 regressions were not random noise; state-matched drift showed concrete decision drift.
- Simple trajectory retention was not enough as a B4 fix.
- Direct trajectory BC on winning states was much more useful.

Important datasets/runs:

- B2 direct-win dataset:
  - audit: `runs/trajectory_bc_direct_vs_b2_replay_audit64_20260516`
  - dataset: `runs/trajectory_bc_direct_b2_win_64_20260516/trajectory_bc_direct_b2_win_64.npz`
  - `88` bundles, `7359` train rows
- B3 direct-win dataset:
  - audit: `runs/trajectory_bc_direct_vs_b3_replay_audit64_20260516`
  - dataset: `runs/trajectory_bc_direct_b3_win_64_20260516/trajectory_bc_direct_b3_win_64.npz`
  - `82` bundles, `6902` train rows
- B4 direct-win dataset:
  - dataset: `runs/trajectory_bc_direct_b4_win_64_20260516/trajectory_bc_direct_b4_win_64.npz`
  - `83` bundles, `6970` train rows
- merged balanced B2/B3/B4 direct-win dataset:
  - `runs/trajectory_bc_direct_b2_b3_b4_win_64_20260516/trajectory_bc_direct_b2_b3_b4_win_64.npz`
  - `253` bundles, `21231` train rows, `56166` padded rows

Warmstart result:

- run: `runs/trajectory_bc_warmstart_direct_b2_b3_b4_win_e1_20260516`
- policy: `trajectory_bc_latest`
- confirm256:
  - B2: `350/512 = 0.683594`
  - B3: `312/512 = 0.609375`
  - B4: `310/512 = 0.605469`
  - overall: `972/1536 = 0.632813`

What helped:

- balanced direct-win trajectory BC improved overall strength compared with B4-only direct data;
- it gave a much better seed for controlled continuation;
- it directly attacked the state/action mismatch rather than hoping reward would indirectly solve it.

What did not help:

- B4-only trajectory BC was useful but imbalanced;
- simple trajectory retention did not fix B4 regression;
- more replay retention without good selection was not enough.

### 12. No-Public Strong-Anchor Continuation

Config:

- `configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml`

This extended the strong-anchor continuation path while retaining balanced direct B2/B3/B4 trajectory BC.

What helped:

- removing public heuristic pressure reduced misalignment;
- strong anchor floors prevented candidates that were good on one opponent but bad on another from being promoted;
- bounded continuation from selected checkpoints produced stable improvements.

Key progression:

- old direct B4-only seed: `960/1536`
- balanced direct B2/B3/B4 seed: `972/1536`
- first selected continuation: `981/1536`
- first re-continuation: `991/1536`
- second re-continuation: `992/1536`

Final selected update:

- run: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- update: `15`
- confirm256:
  - B2: `341/512 = 0.666016`
  - B3: `330/512 = 0.644531`
  - B4: `321/512 = 0.626953`
  - overall: `992/1536 = 0.645833`

What did not help:

- continuing recursively after this showed plateauing; the last improvement was only `+1/1536`;
- update 20/latest was repeatedly weaker than update 15;
- this branch should not be extended blindly without a new structural idea.

### 13. Candidate Selection Repair

Added/changed:

- `python/weiss_rl/experiments/b1_candidate_selection.py`
- `python/scripts/select_b1_candidate.py`
- `python/weiss_rl/tests/test_b1_candidate_selection.py`

Bug fixed:

- the selector could let a confirm64 summary drive selection even when the command requested confirm256 selection.

New behavior:

- `_confirmation_scores(...)` supports `min_paired_seeds`;
- targeted-confirm-only records respect requested `confirm_paired_seeds`;
- low-seed exploratory confirms can suggest follow-up evals but cannot displace a higher-depth confirmed candidate.

What helped:

- made candidate selection match the evidence standard;
- allowed automatic publication of `selected_candidate`;
- preserved best/latest semantics in the snapshot registry.

What did not help:

- it does not generate better candidates by itself;
- it depends on targeted confirm artifacts existing at the requested depth.

### 14. Checkpoint Interpolation Attempt

Added:

- `python/weiss_rl/training/checkpoint_interpolation.py`
- `python/scripts/interpolate_checkpoints.py`
- `python/weiss_rl/tests/test_checkpoint_interpolation.py`

Attempt:

- alpha `0.5` interpolation between old direct and balanced direct checkpoints.

Result:

- artifact: `runs/checkpoint_interp_olddirect_balanceddirect_a050_20260516`
- confirm64:
  - B2: `94/128`
  - B3: `83/128`
  - B4: `73/128`
  - overall: `250/384`

Conclusion:

- worse than the selected branch;
- not promoted;
- useful tooling, but this interpolation was not a quality improvement.

### 15. Confirm512 Attempt

We attempted to extend targeted confirmation to 512 paired seeds.

Result:

- failed immediately because `configs/seeds/report_eval_seeds.txt` has only `256` non-empty paired seeds;
- this is correct behavior;
- the code did not silently reuse or invent seeds.

Conclusion:

- confirm256 is the current maximum report-depth contract;
- extend the seed file deliberately if deeper confirmation is needed.

## What Helped Most

The biggest wins were not a single hyperparameter.

1. Clean B1 route

   Removing inherited heuristic actor/teacher/public guidance made B1 claims meaningful.

2. Reward/value perspective repair

   Actor-perspective reward and value consistency removed a real structural risk.

3. Best/latest checkpoint separation

   This stopped the workflow from treating weaker late checkpoints as canonical.

4. Packed scoring and eval surface repair

   This reduced action-surface mismatch risk and made eval more trustworthy.

5. Reward shaping as cleanup

   Full shaping helped, but mainly as a better learning signal. It did not solve B2 alone.

6. Action-level diagnostics

   B2/B4 audits showed the problem was public tactical behavior and state/action drift, not just noisy scalar training curves.

7. Balanced direct-win trajectory BC

   This was the first major quality jump that improved the multi-anchor surface.

8. No-public strong-anchor continuation

   This made the balanced seed better without reintroducing public heuristic misalignment.

9. Confirm-depth-aware candidate selection

   This made the final artifact defensible.

10. Paper-readiness pipeline

   This turned "looks good" into a reproducible thesis artifact.

## What Did Not Help Enough

These were useful to try, but did not solve the core problem by themselves.

1. Training the original medium64 run longer

   Proxies improved, but policy quality was non-monotonic and update 100 could beat later checkpoints.

2. Bigger local model probes

   Wider 96/128 probes approached the local VRAM ceiling and were rejected for long runs.

3. Reward shaping alone

   Full shaping was better than terminal/damage-only, but B2 still remained weak in pure reward probes.

4. Actor sync and low entropy alone

   Sync/entropy variants did not remove collapse or transfer issues.

5. Public heuristic guidance and online imitation

   These could contaminate B1 claims and misalign behavior.

6. More exposure or lane mechanics alone

   Earlier reserved-anchor/PFSP-style exposure did not fix B2 transfer by itself.

7. Guards alone

   Mulligan/main-move/attack/anti-pass guards improved local symptoms but were brittle.

8. Exact teacher targets alone

   Some exact-target or sharp-teacher variants looked promising locally but drifted or failed multi-anchor confirmation.

9. Simple trajectory retention

   Retention alone did not fix B4.

10. B4-only direct BC

   It helped B4 but was less balanced across B2/B3/B4 than the final merged direct-win data.

11. Checkpoint interpolation

   The alpha `0.5` interpolation candidate was worse and was not promoted.

12. Blind recursive continuation

   The final branch plateaued. The last gain was only `+1/1536`.

## Key Files Added Or Changed

This list is not the entire dirty worktree. It is the high-signal B1 learning set.

Configs:

- `configs/thesis/b1_noleague.yaml`
- `configs/thesis/final_eval.yaml`
- `configs/thesis/final_eval_no_replay.yaml`
- `configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml`
- `configs/thesis/main_league_guided_bootstrap_selected_trajbc_b4win_anchor_nopublic.yaml`
- `configs/thesis/ablations/*.yaml`

Scripts:

- `python/scripts/eval.py`
- `python/scripts/train.py`
- `python/scripts/reward_component_probe.py`
- `python/scripts/learning_progress_diagnostic.py`
- `python/scripts/learning_run_compare.py`
- `python/scripts/b2_disagreement_audit.py`
- `python/scripts/replay_inspector.py`
- `python/scripts/trajectory_audit_compare.py`
- `python/scripts/trajectory_policy_drift.py`
- `python/scripts/replay_trajectory_bc_dataset.py`
- `python/scripts/trajectory_bc_warmstart.py`
- `python/scripts/merge_replay_trajectory_bc_datasets.py`
- `python/scripts/select_b1_candidate.py`
- `python/scripts/interpolate_checkpoints.py`
- `python/scripts/guarded_league_bootstrap.py`
- `python/scripts/segmented_b1_guided_bootstrap.py`

Core/training/eval:

- `python/weiss_rl/replay/trajectory_bc.py`
- `python/weiss_rl/training/checkpoint_interpolation.py`
- `python/weiss_rl/experiments/b1_candidate_selection.py`
- `python/weiss_rl/learners/impala_learner.py`
- `python/weiss_rl/learners/vtrace*.py`
- `python/weiss_rl/learners/*structured*`
- `python/weiss_rl/models/*`
- `python/weiss_rl/eval/*`
- `python/weiss_rl/runtime*`

Tests:

- `python/weiss_rl/tests/test_replay_trajectory_bc.py`
- `python/weiss_rl/tests/test_checkpoint_interpolation.py`
- `python/weiss_rl/tests/test_b1_candidate_selection.py`
- `python/weiss_rl/tests/test_config_loader.py`
- `python/weiss_rl/tests/test_reward_component_probe.py`
- `python/weiss_rl/tests/test_learning_progress_diagnostic.py`
- `python/weiss_rl/tests/test_impala_learner.py`
- `python/weiss_rl/tests/test_runtime_reward_shaping.py`
- `python/weiss_rl/tests/test_policy_anchor.py`
- many existing runtime/model/eval tests were extended during the broader rebuild

Docs:

- `docs/rebuild_log.md`
- `docs/thesis_workflow.md`
- `docs/artifact_contract.md`
- this report: `docs/b1_learning_rebuild_report_20260517.md`

## Important Runs And Artifacts

Early medium64 evidence:

- `runs/phase2_b1_medium64_probe_20260512`
  - 2 updates
  - mean throughput around `21001.79` samples/sec
  - max GPU memory around `10474 MB`
- `runs/phase2_b1_medium64_stability20_20260512`
  - 20 updates
  - mean throughput around `23783.95` samples/sec
  - max GPU memory around `10486 MB`
  - 0 AMP overflows
- `runs/b1_medium64_thesis_local_20260512`
  - reached update 181 in scalar logs
  - durable checkpoint/snapshot to update 175
  - throughput around `25.2k` samples/sec
  - not a clean canonical B1 artifact

Reward probes:

- `runs/b1_reward_terminal_only_probe25b_20260513`
- `runs/b1_reward_damage_only_probe25b_20260513`
- `runs/b1_reward_damage_level_probe25b_20260513`
- `runs/b1_reward_full_shaping_probe25b_20260513`
- `runs/b1_reward_full_shaping_probe100_20260513`
- `runs/b1_reward_damage_only_probe100_20260513`
- `runs/b1_reward_full_shaping_sync10_probe100_20260513`
- `runs/b1_reward_full_shaping_entropy01_probe100_20260513`
- `runs/b1_reward_full_shaping_entropy01_sync20_probe100_20260513`

Trajectory and BC:

- `runs/trajectory_bc_direct_vs_b2_replay_audit64_20260516`
- `runs/trajectory_bc_direct_vs_b3_replay_audit64_20260516`
- `runs/trajectory_bc_direct_b2_win_64_20260516`
- `runs/trajectory_bc_direct_b3_win_64_20260516`
- `runs/trajectory_bc_direct_b4_win_64_20260516`
- `runs/trajectory_bc_direct_b2_b3_b4_win_64_20260516`
- `runs/trajectory_bc_warmstart_direct_b2_b3_b4_win_e1_20260516`

Final selected branch:

- `runs/guarded_direct_b2b3b4_anchor_nopublic_u20_20260516_seg01`
- `runs/guarded_recontinue_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`

Rejected/intermediate:

- `runs/checkpoint_interp_olddirect_balanceddirect_a050_20260516`

Selection diagnostics:

- `diagnostics/b1_candidate_selection_balanced_direct_published_20260516.json`
- `diagnostics/b1_candidate_selection_balanced_continuation_published_20260516.json`
- `diagnostics/b1_candidate_selection_recontinued2_selected_published_20260516.json`

## Current Final Matrix

Mean win-rate matrix:

```csv
focal_policy_id,selected_candidate,B0 RandomLegal,B2 HeuristicPublic,B3 HeuristicPublicAggro,B4 HeuristicPublicControl
selected_candidate,0.5,1.0,0.666015625,0.64453125,0.626953125
B0 RandomLegal,0.0,0.48046875,0.01953125,0.017578125,0.017578125
B2 HeuristicPublic,0.333984375,0.98046875,0.5,0.3046875,0.494140625
B3 HeuristicPublicAggro,0.35546875,0.982421875,0.6953125,0.5,0.486328125
B4 HeuristicPublicControl,0.373046875,0.982421875,0.505859375,0.513671875,0.5
```

Wins matrix:

```csv
focal_policy_id,selected_candidate,B0 RandomLegal,B2 HeuristicPublic,B3 HeuristicPublicAggro,B4 HeuristicPublicControl
selected_candidate,256,512,341,330,321
B0 RandomLegal,0,246,10,9,9
B2 HeuristicPublic,171,502,256,156,253
B3 HeuristicPublicAggro,182,503,356,256,249
B4 HeuristicPublicControl,191,503,259,263,256
```

## Interpretation

The final B1 is not "perfect Weiss Schwarz AI." It is a defensible B1 NoLeague thesis baseline:

- it beats random cleanly;
- it beats B2/B3/B4 with confirm256 and final-eval evidence;
- it has reproducible artifacts;
- it passes paper readiness;
- it preserves selected-vs-latest semantics;
- it does not rely on hidden public heuristic actor contamination.

The original failure mode is addressed:

- We no longer assume longer training means a better policy.
- We do not promote latest by default.
- We have diagnostics that reveal plateau/collapse.
- We have a selected checkpoint that is better than later chronological snapshots.

## Remaining Risks

1. B1 is solved enough for its current role, not for all possible future opponents.

   The selected candidate is strong on the fixed B0-B4 surface. It has not yet been proven as the seed inside the final main league run.

2. Confirm depth is capped.

   `configs/seeds/report_eval_seeds.txt` currently contains 256 seeds. A confirm512 run is invalid until that contract is extended.

3. Recursive continuation has plateaued.

   Further B1 improvement should come from a new structural hypothesis, not another identical continuation.

4. Promotion automation still matters.

   The selector now works, but future training should automatically run the right confirmation gates and publish best-confirmed aliases.

5. The main league path may reintroduce drift.

   The next phase must watch B4 and selected-vs-latest behavior carefully.

## Recommended Next Goal

Start the main guided league/bootstrap phase using this B1 selected candidate as the seed.

The next goal should be something like:

```text
Improve and validate the main Weiss Schwarz thesis league run seeded from the paper-ready B1 selected_candidate artifact. Preserve best/latest semantics, use B0-B4 fixed-deck eval, require multi-anchor confirmation before promotion, diagnose B4 drift early, and produce paper-ready final eval/metagame/figure artifacts for the main league model.
```

Suggested first actions:

1. Wire `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01/training/snapshots/selected_candidate/weights.pt` as the B1 seed for the main guided league/bootstrap run.
2. Make promotion automatic:
   - require B2/B3/B4 anchor floors;
   - require requested confirmation depth;
   - never let confirm64 override confirm256;
   - never treat latest as best.
3. Run a short main league smoke/probe with aggressive B4 monitoring.
4. Compare selected, latest, and rollback candidates after each segment.
5. Only after transfer is stable, launch the longer thesis main run.

## Bottom Line

The B1 rebuild succeeded because the work stopped treating the problem as "train longer" or "tune one knob." The useful path was:

1. clean the B1 route;
2. fix structural reward/value/eval/checkpoint issues;
3. make non-monotonicity visible;
4. use reward shaping as a better signal, not as the whole solution;
5. diagnose state/action drift;
6. build balanced direct-win trajectory BC;
7. continue from selected checkpoints under strong B2/B3/B4 anchor floors;
8. publish a selected alias only after confirm256 evidence;
9. run the full B0-B4 paper-ready eval.

That produced a defensible B1 NoLeague artifact and cleared the way to move into the full league model.
