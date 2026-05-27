# Main League Rebuild Report

Date: 2026-05-18

Repo: `C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl`

Scope: longer B1-seeded main-league training, champion/hard-negative ecology
diagnosis, interpolation-based selected checkpoint publication, and
paper-ready fixed-deck B0-B4 plus B1 evaluation.

The chronological work log is `docs/rebuild_log.md`.

## Current Status

The current paper-ready fixed-deck main artifact is:

- run: `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- selected alias: `main_league_selected`
- selected source policy id: `main_interp_repair_a015`
- selected update: `5`
- selected weights hash:
  `1a13b49b73ed71af0914c97fede5b30703eb576a5e85c4c636310c2d76897b26`
- selection summary:
  `diagnostics/main_champion_hardneg_interp_u10_repair_a015_selected.json`
- confirm256 summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/targeted_confirm256_b0_b4/targeted_confirm256_summary.json`

The selected checkpoint is an explicit model-space interpolation:

- first checkpoint:
  `runs/main_champion_hardneg_long_v1_u10_20260517_seg01/training/checkpoints/checkpoint_10.pt`
- second checkpoint:
  `runs/main_champion_hardneg_rehearsal_from_u20_u5_20260517_seg01/training/checkpoints/checkpoint_5.pt`
- second weight: `0.15`
- interpolation summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/diagnostics/checkpoint_interpolation_summary.json`

The locked B1 seed remained:

- run: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- selected policy id: `selected_candidate`
- selected source policy id: `policy_000003`
- selected update: `15`
- weights hash:
  `66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685`

Snapshot semantics:

- `latest` remains chronological latest.
- `selected_candidate` remains the locked best-confirmed B1 alias.
- `main_league_selected` is published from confirm256 evidence.
- No model was promoted because scalar loss improved.

## Fixed Deck Surface

The final evidence uses the thesis fixed-deck policy:

- focal/main/B0/B1/B2: `preset:main_deck_5hy_yotsuba_v1`
- B3 aggro: `preset:aggro_deck_5hy_nino_v1`
- B4 control: `preset:control_deck_jj_s66_v1`

No multideck results are mixed into the selected-main claim.

## Final Evidence

The selected alias was confirmed against every required fixed-deck opponent at
256 paired seeds:

| Opponent | Wins | Games | Win rate | 95% CI | P(p > 0.5) |
|---|---:|---:|---:|---|---:|
| B0 RandomLegal | 512 | 512 | 1.000000 | [1.000000, 1.000000] | 1.000 |
| B1 NoLeague baseline | 322 | 512 | 0.628906 | [0.591930, 0.664397] | 1.000 |
| B2 HeuristicPublic | 399 | 512 | 0.779297 | [0.745009, 0.811999] | 1.000 |
| B3 HeuristicPublicAggro | 365 | 512 | 0.712891 | [0.674358, 0.748142] | 1.000 |
| B4 HeuristicPublicControl | 382 | 512 | 0.746094 | [0.707268, 0.785702] | 1.000 |

Overall fixed-deck confirm256:

- `1980/2560 = 0.773438`
- prior p2 selected artifact:
  `1971/2560 = 0.769922`
- first champion/hard-negative u10 candidate:
  `1974/2560 = 0.771094`

Compared with the previous p2 selected artifact
`runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01`, this selected model
improves B1, B2, B3, and overall while B4 is slightly lower but still strongly
above parity:

| Opponent | Previous p2 selected | Current selected | Delta |
|---|---:|---:|---:|
| B1 NoLeague baseline | 0.621094 | 0.628906 | +0.007812 |
| B2 HeuristicPublic | 0.765625 | 0.779297 | +0.013672 |
| B3 HeuristicPublicAggro | 0.708984 | 0.712891 | +0.003906 |
| B4 HeuristicPublicControl | 0.753906 | 0.746094 | -0.007812 |
| Overall B0-B4+B1 | 0.769922 | 0.773438 | +0.003516 |

Canonical final eval completed with `main_league_selected`, B0, B1, B2, B3,
and B4:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python python/scripts/eval.py --stack-config configs/thesis/final_eval.yaml --run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517 --snapshot-registry-json runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry.json --b1-baseline-run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 --policy-id main_league_selected --policy-id "B0 RandomLegal" --policy-id "B1 NoLeague baseline" --policy-id "B2 HeuristicPublic" --policy-id "B3 HeuristicPublicAggro" --policy-id "B4 HeuristicPublicControl" --paired-seed-limit 256 --stage1-paired-seeds 64 --max-paired-seeds 256 --bootstrap-samples 2000
```

Canonical final-eval selected-main row, with adaptive stopping:

| Opponent | Win rate |
|---|---:|
| B0 RandomLegal | 1.000000 |
| B1 NoLeague baseline | 0.640625 |
| B2 HeuristicPublic | 0.781250 |
| B3 HeuristicPublicAggro | 0.703125 |
| B4 HeuristicPublicControl | 0.718750 |

The headline fixed-deck claim should cite the separate confirm256 table above;
canonical final eval is the artifact-complete matrix/metagame run.

## Champions And Hard Negatives

The first longer league segment from the p2 selected checkpoint used real
learned champion and hard-negative sampling:

- run: `runs/main_champion_hardneg_long_v1_u10_20260517_seg01`
- imported learned seed snapshots: `8`
- final pool metrics:
  - `pfsp_champion_pool_size = 7`
  - `pfsp_hard_negative_pool_size = 1`
  - `pfsp_champion_envs = 240`
  - `pfsp_hard_negative_envs = 135`
  - `pfsp_noleague_baseline_envs = 461`

The selected interpolation was evaluated against the imported champion panel at
128 paired seeds:

| Imported opponent | Wins | Games | Win rate |
|---|---:|---:|---:|
| `seed_c3aac2f9dc_policy_000001` | 169 | 256 | 0.660156 |
| `seed_c3aac2f9dc_policy_000002` | 148 | 256 | 0.578125 |
| `seed_c3aac2f9dc_checkpoint_000025` | 149 | 256 | 0.582031 |
| `seed_c3aac2f9dc_main_bestresponse_u25_devbest` | 149 | 256 | 0.582031 |
| `seed_c3aac2f9dc_main_league_selected` | 140 | 256 | 0.546875 |
| `seed_c3aac2f9dc_policy_000003` | 140 | 256 | 0.546875 |
| `seed_c3aac2f9dc_policy_000004` | 144 | 256 | 0.562500 |
| `seed_c3aac2f9dc_policy_000005` | 143 | 256 | 0.558594 |

Champion-panel aggregate:

- `1182/2048 = 0.577148`

This is positive against every imported learned champion/hard-negative
candidate, but it is not a strict improvement over the raw u10 champion run
(`1187/2048`). The thesis-defensible claim is therefore:

- longer league training exposed the model to real champions and hard negatives;
- failed continuations revealed B3/B4 and B1 drift modes;
- the selected interpolation improves the fixed-deck thesis surface while
  retaining positive learned-opponent scores;
- the champion/hard-negative claim is supportive robustness evidence, not the
  headline selection criterion.

## Rejected Alternatives

Rejected or not selected:

- u20 direct continuation:
  `runs/main_champion_hardneg_long_v1_u10_to_u20_20260517_seg01`
  - confirm64 overall: `490/640 = 0.765625`
  - B3 dropped below the reference-drop guard at `0.664063`
  - stopped as `stopped_guard_failed`
- u20 rehearsal repair:
  `runs/main_champion_hardneg_rehearsal_from_u20_u5_20260517_seg01`
  - B3 recovered to `0.718750`
  - B1 softened to `0.593750`
  - useful as interpolation source, not selected directly
- consolidation after repair:
  `runs/main_champion_hardneg_consolidation_from_repair_u10_20260517_seg01`
  - confirm64 overall: `476/640 = 0.743750`
  - B1 collapsed to `0.539063`
- stable-long branch from u10:
  `runs/main_champion_hardneg_stable_from_u10_to_u20_20260517_seg01`
  - confirm64 overall: `482/640 = 0.753125`
  - B1/B4 drifted
- polish from selected interpolation:
  `runs/main_champion_hardneg_polish_from_interp_a015_u5_20260517_seg01`
  - confirm64 matched the selected interpolation smoke pattern but did not beat it

## Artifact Manifest

Selected-main artifacts:

- final eval summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/summary.json`
- final eval matrices:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/matrices/`
- final eval payoff matrices:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/payoff_matrices/`
- policy set:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/policy_set.json`
- metagame summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/metagame/summary.json`
- replay verification:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/diagnostics/replay_verification.json`
- paper readiness:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/paper_readiness_summary.json`
- paper figures:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/`

Paper readiness passed:

- `passed: true`
- alarms: `[]`
- replay verification: `3/3` sampled episodes verified, `0` failed
- truncation rate: `0.0`
- aggregate seat-bias check: passed

Rendered paper figures:

- `fig_learning_curves.{pdf,png}`
- `fig_matchup_heatmap.{pdf,png}`
- `fig_seat_bias.{pdf,png}`
- `fig_truncation_heatmap.{pdf,png}`

For this interpolated artifact, `fig_learning_curves` is an interpolation
provenance figure, because no standalone training curve exists for a
model-space interpolation.

## Code And Config Changes

New or updated behavior:

- guarded league bootstrap supports `--b1-baseline-run-dir`, so the learned
  seed-snapshot pool and locked B1 baseline source can be different run dirs;
- interpolation runs now satisfy canonical eval artifacts without requiring a
  fake training curve;
- paper readiness accepts checkpoint interpolation provenance as the training
  history artifact for interpolated selected checkpoints;
- eval writes missing run-summary, determinism, and environment scaffolding for
  generated interpolation runs before readiness.

New ablation configs:

- `configs/thesis/ablations/main_league_champion_hardneg_long_probe.yaml`
- `configs/thesis/ablations/main_league_champion_hardneg_rehearsal_probe.yaml`
- `configs/thesis/ablations/main_league_champion_hardneg_consolidation_probe.yaml`
- `configs/thesis/ablations/main_league_champion_hardneg_stable_long_probe.yaml`
- `configs/thesis/ablations/main_league_champion_hardneg_polish_probe.yaml`

## Validation

Focused tests and lint run during this phase:

- `uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_guarded_league_bootstrap.py python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_long_probe`
- `uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_polish_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_stable_long_probe python/weiss_rl/tests/test_config_loader.py::test_load_stack_config_supports_main_league_champion_hardneg_rehearsal_probe`
- `uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_entrypoints.py::test_eval_report_helpers_create_defaults_for_interpolated_runs python/weiss_rl/tests/test_paper_readiness.py::test_build_paper_readiness_summary_accepts_interpolation_provenance_instead_of_training_metrics python/weiss_rl/tests/test_paper_figures.py::test_render_paper_figures_uses_interpolation_provenance_when_training_metrics_are_absent`
- `uv run --extra dev ruff check python/scripts/eval.py python/weiss_rl/eval/paper_readiness.py python/weiss_rl/tests/test_entrypoints.py python/weiss_rl/tests/test_paper_readiness.py python/weiss_rl/plotting/paper_figures.py python/weiss_rl/tests/test_paper_figures.py`

Canonical final eval completed and paper readiness passed.

## Remaining Risks

The selected model is the strongest fixed-deck thesis artifact by confirmed
overall B0-B4+B1 score. It does not strictly improve every row: B4 is
`-0.007812` below the previous p2 selected artifact on confirm256. That should
be reported directly.

The learned champion panel is positive for every imported champion/hard-negative
candidate but not strictly better than the raw u10 champion run. Future work, if
needed, should focus on a true multi-objective gate that confirms both fixed
baselines and a champion panel before extension, rather than relying on scalar
loss or single-anchor improvement.

## Continuation Addendum: Multi-Objective Gate

After this report's selected artifact was produced, a first-class
multi-objective gate was added:

- script: `python/scripts/main_league_multiobjective_gate.py`
- tests: `python/weiss_rl/tests/test_main_league_multiobjective_gate.py`
- reference scores:
  `diagnostics/main_champion_hardneg_multiobjective_reference_scores_20260518.json`

The gate makes the current frontier explicit:

| Candidate | Fixed B0-B4+B1 mean | Learned panel mean | Learned delta | Gate |
|---|---:|---:|---:|---|
| raw u10 champion/hard-negative run | 0.771094 | 0.579590 | 0.000000 | pass |
| selected interpolation a015 | 0.773438 | 0.577148 | -0.002441 | fail learned |
| online continuation v2 | 0.762500 | 0.544922 | -0.034668 | fail learned |
| online continuation v3 | 0.760938 | 0.555664 | -0.023926 | fail learned |

The selected interpolation remains the paper-ready fixed-deck selected artifact.
The raw u10 run remains the learned-opponent frontier. The two new online
continuations were not promoted or published because confirm64 evidence showed
learned-panel regression before any confirm256 claim was attempted.

## Confirm256 Learned-Panel Addendum

The earlier learned-panel comparison was confirm128 and was deliberately treated
as provisional. A follow-up confirm256 pass changed the conclusion:

| Candidate | Fixed B0-B4+B1 mean | Learned panel mean | Learned delta vs raw u10 | Gate |
|---|---:|---:|---:|---|
| raw u10 champion/hard-negative run | 0.771094 | 0.564453 | 0.000000 | reference |
| selected interpolation a015 | 0.773438 | 0.565674 | +0.001221 | pass |
| interpolation a010 | 0.771094 | 0.565186 | +0.000732 | pass |

The selected interpolation a015 is therefore still the best current thesis
artifact on the combined fixed-deck plus learned champion/hard-negative
surface. Its learned-panel edge is small, so it should be reported as a guarded
positive result rather than a decisive improvement.

Two five-update online continuations from a010 were also tested with the learned
champion rows included in the guard:

- `runs/main_champion_hardneg_from_a010_multiobj_u5_20260518_seg01`
- `runs/main_champion_hardneg_from_a010_sched0_multiobj_u5_20260518_seg01`

Both were rejected before publication. Resetting the first guidance schedule
offset to `0` did not change the result, so the failure is not explained by
carried schedule time alone. The strongest observed failure was learned-panel
drift: several imported champion/hard-negative rows fell more than the allowed
`0.03` reference drop, while fixed anchors stayed mostly inside the guard.

New evidence artifacts:

- `diagnostics/main_interp_repair_a015_multiobjective_gate_vs_p2_u10_confirm256learned_20260518.json`
- `diagnostics/main_champion_hardneg_interp_u10_repair_a010_multiobjective_gate_confirm256learned_20260518.json`
- `diagnostics/main_champion_hardneg_multiobjective_reference_scores_confirm256learned_20260518.json`
- `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/targeted_confirm256_imported_champions/targeted_confirm256_summary.json`
- `runs/main_champion_hardneg_long_v1_u10_20260517_seg01/eval/targeted_confirm256_imported_champions/targeted_confirm256_summary.json`
- `runs/main_champion_hardneg_interp_u10_repair_a010_20260518/eval/targeted_confirm256_imported_champions/targeted_confirm256_summary.json`

## Longer Hard-Negative Follow-Up

After the confirm256 learned-panel addendum, a longer hard-negative-focused
learned-push segment was run from the same selected a015 seed:

- run:
  `runs/main_champion_hardneg_alloutcome_learnedpush_ids_u8_20260518_seg01`
- config:
  `configs/thesis/ablations/main_league_champion_hardneg_selected_alloutcome_learnedpush_b4b2guard_probe.yaml`
- segment length: `8` updates
- confirm surface: fixed B0-B4+B1 plus the eight imported learned champion rows
- confirm candidates: `policy_000006`, `policy_000007`, `policy_000008`

The run proves the league was actually playing against hard negatives:

- `pfsp_hard_negative_envs` total: `1927`
- `pfsp_champion_envs` total: `922`
- `training/logs/opponent_pool.jsonl` records the actual hard-negative policy
  IDs at refresh time.
- collector hard-negative group improved from `0.485564` first observed to
  `0.509278` last observed.

However, the paired confirm did not improve the thesis model:

| Candidate | Confirm64 | Delta vs selected a015 | Fixed delta | Learned delta | Status |
|---|---:|---:|---:|---:|---|
| `policy_000006` | `1097/1664` | `0` | `+1` | `-1` | rejected |
| `policy_000007` | `1096/1664` | `-1` | `+2` | `-3` | rejected |
| `policy_000008` | `1095/1664` | `-2` | `+1` | `-3` | rejected |

Update 7 passed the online promotion gate, but confirm64 rejected it relative to
the external thesis surface. This is important for the thesis workflow: online
promotion is useful for ecology, but selected/best must remain the
best-confirmed checkpoint on paired external evidence.

The selected paper-ready model is unchanged:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.

New evidence artifacts:

- `runs/main_champion_hardneg_alloutcome_learnedpush_ids_u8_20260518_seg01/training/logs/opponent_pool.jsonl`
- `diagnostics/league_progress_learnedpush_ids_u8_seg01_20260518.json`
- `diagnostics/paired_outcome_compare_selected_a015_vs_learnedpush_ids_u8_policy000006_confirm64shared_20260518.json`
- `diagnostics/paired_outcome_compare_selected_a015_vs_learnedpush_ids_u8_policy000007_confirm64shared_20260518.json`
- `diagnostics/paired_outcome_compare_selected_a015_vs_learnedpush_ids_u8_policy000008_confirm64shared_20260518.json`
- `diagnostics/trajectory_policy_drift_a015_vs_learnedpush_ids_u8_p6_on_selected_a015_imported_champions_all32_20260518.json`

Conclusion: the next improvement needs a structural credit/repair step over
paired swing examples, not just more updates with diffuse hard-negative
sampling.

## Paired Swing Repair Diagnostic

A paired swing diagnostic was added to explain where hard-negative training was
helping or hurting:

- `python/scripts/paired_swing_report.py`
- `diagnostics/paired_swing_report_learnedpush_ids_u8_p6_p7_p8_confirm64shared_20260518.json`

For the rejected learned-push u8 candidates p6-p8, the aggregate paired swing
pattern was:

| Surface | Delta vs selected a015 |
|---|---:|
| all compared | `-3` |
| fixed B0-B4+B1 | `+4` |
| learned/hard-negative | `-7` |

This made the failure mode concrete: longer training was improving some fixed
anchor behavior while losing against the imported learned hard negatives.

A small explicit repair replay dataset was then built from selected-a015 wins on
the hard-negative regression seeds:

- `runs/trajectory_bc_selected_a015_hardneg_regression_swing_win4x6_20260518/trajectory_bc_selected_a015_hardneg_regression_swing_win4x6.npz`
- `2185` train rows, `27` selected focal-win bundles.

That dataset was merged four times into the broader all-outcome champion replay
dataset:

- `runs/trajectory_bc_selected_a015_all32_plus_swing4x_20260518/trajectory_bc_selected_a015_all32_plus_swing4x.npz`
- `50372` train rows, `620` bundles.

The corrected pool run was:

- `runs/main_champion_hardneg_alloutcome_swingrepair_pool_recent3_u6_20260518_seg01`
- config:
  `configs/thesis/ablations/main_league_champion_hardneg_selected_alloutcome_swingrepair_b4b2guard_probe.yaml`

It did play real learned opponents:

- `pfsp_hard_negative_envs`: `1060`
- `pfsp_champion_envs`: `912`
- collector hard-negative group improved from `0.459459` first observed to
  `0.565934` last observed.

The best complete confirm64 candidate was promising:

| Candidate | Confirm64 | Delta vs selected a015 | Fixed delta | Learned delta |
|---|---:|---:|---:|---:|
| `policy_000004` | `1103/1664` | `+6` | `+3` | `+3` |
| `policy_000005` | `1102/1664` | `+5` | `+2` | `+3` |

But confirm128 rejected the candidate:

| Candidate | Confirm128 | Delta vs selected a015 | Fixed delta | Learned delta | Status |
|---|---:|---:|---:|---:|---|
| `policy_000004` | `2176/3328` | `+2` | `+4` | `-2` | rejected |

The multi-objective gate failed only on learned-panel reference delta:
`-0.0009765625`. This is close, but still not publishable, and it reinforces the
rule that confirm64 evidence cannot override confirm128 evidence.

The selected paper-ready model is still unchanged:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.

## Disjoint Hard-Negative Repair Probe

The next repair moved the extra hard-negative replay data off the confirm
surface and onto an explicit disjoint seed file:
`configs/seeds/hardneg_repair_train_seeds_20260518.txt`.

New artifacts:

- `configs/thesis/ablations/main_league_champion_hardneg_selected_alloutcome_disjointrepair_b4b2guard_probe.yaml`
- `runs/trajectory_bc_selected_a015_hardneg_disjointrepair_win16x6_20260518/trajectory_bc_selected_a015_hardneg_disjointrepair_win16x6.npz`
- `runs/trajectory_bc_selected_a015_all32_plus_disjointrepair4x_20260518/trajectory_bc_selected_a015_all32_plus_disjointrepair4x.npz`
- `runs/main_champion_hardneg_alloutcome_disjointrepair_pool_recent2_u5_20260518_seg01`
- `diagnostics/paired_swing_report_disjointrepair_pool_p4_p5_confirm64shared_20260518.json`
- `diagnostics/league_progress_disjointrepair_pool_seg01_20260518.json`

The run did play real learned opponents:

| Collector surface | Exposure |
|---|---:|
| hard negative | `755` |
| champion | `910` |
| sampled league opponents | `4539` |

The hard-negative collector win rate improved during training from `0.459459`
first observed to `0.513514` last observed, but paired external confirmation
again exposed a fixed-vs-learned tradeoff:

| Candidate | Confirm64 | Delta vs selected a015 | Fixed delta | Learned delta | Status |
|---|---:|---:|---:|---:|---|
| `policy_000004` | `1097/1664` | `0` | `+2` | `-2` | rejected |
| `policy_000005` | `1100/1664` | `+3` | `+4` | `-1` | rejected |

Aggregated over p4 and p5, the swing report was `+6` on fixed B0-B4+B1 and
`-3` on learned hard negatives. This is cleaner evidence than the first swing
repair because the repair seeds are disjoint, but it is still not publishable.

The selected paper-ready model remains unchanged:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.

## May 19 B1 Loss-State Top-Action Repair Addendum

The next structural repair moved from replaying winning trajectories to
targeting paired-confirm loss states. The new paired flip manifest:

`diagnostics/paired_flip_targets_a375_overlap_policy_000001_confirm128_lossstates_20260519.json`

extracts exact opponent + pair + swap + seed targets from selected-a015 wins
that the overlap candidate failed to preserve. It found `12` regression targets
and wrote `9` complete candidate episode subsets for replay audits.

The B1 subset was audited first because the overlap run failed the B1 guard by
one confirm128 paired win. The audit:

`runs/disagreement_audit_a375_overlap_policy000001_b1_lossstates_20260519`

reran `4` games across the two B1 regression paired seeds and compared `706`
decision steps. The candidate and B1 agree on top-action family for every
compared step, but exact top-action agreement is only `0.858356940509915`.
The largest disagreements are mostly same-family choices such as
`clock_from_hand` card indices and `trigger_order` indices, so the repair target
is action-level, not broad family-level.

Replay trajectory BC now supports opt-in teacher-action overrides. The replayed
candidate action still drives the simulator trajectory, but selected focal rows
can train on a different teacher action. For this pass, B1 top actions from
high-disagreement inspection rows became the teacher labels:

- override manifest:
  `diagnostics/b1_lossstate_policyb_topaction_overrides_a375_overlap_policy000001_20260519.jsonl`
- B1 corrective dataset:
  `runs/trajectory_bc_a375_overlap_b1_lossstate_policyb_topaction_20260519/trajectory_bc_b1_lossstate_policyb_topaction.npz`
- merged winner-repair plus B1 corrective dataset:
  `runs/trajectory_bc_winnerrepair_plus_b1_lossstate_policyb_topaction_20260519/trajectory_bc_winnerrepair_plus_b1_lossstate_policyb_topaction.npz`
- config:
  `configs/thesis/ablations/main_league_champion_hardneg_selected_alloutcome_stratifiedwinnerrepair_overlap_b1losstopaction_b4b2guard_probe.yaml`

The merged dataset has `1024` bundles and `79932` train rows. The B1 corrective
source is labeled `b1_lossstate_policyb_topaction` and is focus-sampled at
`0.25` of replay-BC episodes in the new config.

A tiny two-update smoke run verified that the path trains end to end:

`runs/main_champion_hardneg_from_overlap_policy000001_b1losstopaction_smoke_u2_20260519_seg01`

Replay-BC was active in that run:

| Metric | Value |
|---|---:|
| replay batch episodes | `32` |
| focused B1-lossstate episodes | `8` |
| replay dataset train rows | `79932` |
| replay teacher-action accuracy | `0.9495967741935484` |

The guarded smoke was rejected because its first guarded confirm intentionally
omitted B0/B2/B3, which made the multiobjective gate report missing fixed
candidate rows. A follow-up full tiny confirm16 on the smoke checkpoint did
cover B0-B4/B1 plus one hard negative:

| Opponent | Confirm16 score |
|---|---:|
| B0 RandomLegal | `32/32` |
| B1 NoLeague baseline | `20/32` |
| B2 HeuristicPublic | `26/32` |
| B3 HeuristicPublicAggro | `25/32` |
| B4 HeuristicPublicControl | `24/32` |
| `seed_b8c698d26a_seed_c3aac2f9dc_policy_000002` | `21/32` |

Same-first16 paired comparison against selected a015 is `+1` all, `+1` fixed,
and `0` learned, with no selected-a015-win / candidate-nonwin flips on those
six rows. This is only smoke evidence, not selection evidence.

The selected paper-ready model remains unchanged:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.

Next step: extend the same override pipeline to the imported hard-negative flip
rows, then run a guarded segment whose confirm set includes all fixed B0-B4/B1
rows and the learned hard-negative/champion guard rows in one pass.

## May 19 Learned-Guard and Overlap-Lane Addendum

A learned-panel anti-regression gate is now wired into guarded league segments.
It evaluates fixed B0-B4/B1 rows and imported learned champion/hard-negative
rows before any publish or unpublished continuation. The gate correctly rejected
the known bad a375 online winner-repair p3 confirm128 candidate because it was
fixed-positive but learned-negative.

Two additional a375-seeded probes were run from
`runs/main_champion_hardneg_interp_a015_extensionrepair_seg01_p5_a375_20260518`:

| Probe | Key change | Confirm128 result vs selected a015 | Status |
|---|---|---:|---|
| stratified winner-repair u8 | 50 percent replay batches reserved for extension winner-repair labels | `-17` all, `+1` fixed, `-18` learned | rejected |
| overlap stratified winner-repair u6 | same replay, but hard negatives can overlap imported champions | `+1` all, `0` fixed, `+1` learned | rejected |

The overlap run is important diagnostically. Earlier live pool refreshes could
move every imported champion into `hard_negative_ids`, leaving the live champion
bucket at `0`. The new opt-in
`league.sampling.hard_negative_overlaps_champions` kept champion pool size at
`8` while hard-negative pool size also grew to `8`, so future runs can honestly
show champion exposure and hard-negative pressure at the same time.

The best overlap candidate was `policy_000001`, not latest `policy_000003`.
It failed only the strict B1 fixed guard:

| Group | Candidate | Reference | Delta |
|---|---:|---:|---:|
| fixed B0-B4+B1 | `992/1280` | `992/1280` | `0` |
| learned imported panel | `1183/2048` | `1182/2048` | `+1` |
| all 13 rows | `2175/3328` | `2174/3328` | `+1` |
| B1 row | `159/256` | `160/256` | `-1` |

The pair-index split still warns against extending this recipe:

| Split | All | Fixed | Learned |
|---|---:|---:|---:|
| first 64 paired seeds | `+4` | `+2` | `+2` |
| extension 64 paired seeds | `-3` | `-2` | `-1` |

The current structural conclusion is sharper now: the league ecology needed the
overlap fix, but replay-BC itself is not teaching the missing decision boundary.
Replay is mechanically active and now reports correct focus/nonfocus counts, but
teacher-action accuracy is already near perfect on replay rows. The next repair
should use loss-state or disagreement data from paired-confirm flips, not a
larger copy of winner trajectories.

The selected paper-ready model remains unchanged:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.

## Learned-Panel Anti-Regression Gate Addendum

The online league failure mode is now enforced by the guarded controller, not
only diagnosed after the fact. `guarded_league_bootstrap` can take learned
champion/hard-negative guard opponents plus a reference full-panel targeted
confirm summary, then stop before publish or unpublished continuation if the
learned aggregate regresses.

The gate was replayed against the known bad a375 winner-repair p3 confirm128
candidate:

| Candidate | Fixed aggregate | Learned aggregate | Gate result |
|---|---:|---:|---|
| `main_champion_hardneg_from_interp_a375_winnerrepair... policy_000003` | `0.7765625` | `0.57568359375` | rejected |

The rejection artifact is:
`diagnostics/main_league_multiobjective_gate_a375_winnerrepair_p3_confirm128_20260519.json`.

It reports two important failures:

- learned imported panel reference delta: `-0.00146484375`;
- B1 NoLeague reference delta: `-0.00390625`.

This matters because the paired comparison had made the same candidate look
tempting at a high level (`+2` fixed, `-3` learned at confirm128). The new gate
turns that thesis-relevant learned-panel regression into a hard stop, so longer
training cannot quietly continue from a candidate that improves fixed totals
while giving back champion/hard-negative performance.

Validation for the gate change:

- `PYTHONHASHSEED=0 uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_main_league_multiobjective_gate.py python/weiss_rl/tests/test_guarded_league_bootstrap.py`
  -> `21 passed`.
- `uv run --extra dev ruff check ...`
  -> passed on the changed gate/controller/test files.

The selected paper-ready model remains unchanged:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.

## Winner-Repair and Interpolation Addendum

The next repair used two signals from the failed extension-repair run:

- selected-a015 winning hard-negative trajectories, to preserve the locked
  model's learned-opponent wins;
- extension-repair p5 winning hard-negative trajectories, to keep the few real
  champion/hard-negative gains that appeared before confirm128 rejection.

This produced:

- `runs/trajectory_bc_extensionrepair_p5_hardneg_disjoint_win16x6_20260518/trajectory_bc_extensionrepair_p5_hardneg_disjoint_win16x6.npz`
- `runs/trajectory_bc_selected_a015_disjoint4x_plus_extensionp5win2x_20260518/trajectory_bc_selected_a015_disjoint4x_plus_extensionp5win2x.npz`
- `configs/thesis/ablations/main_league_champion_hardneg_selected_alloutcome_winnerrepair_b4b2guard_probe.yaml`

The selected-a015 seeded winner-repair segment was rejected:

| Candidate | Confirm64 paired delta | Confirm128 paired delta | Status |
|---|---:|---:|---|
| winner-repair p4 | `+2` fixed, `+2` learned | `0` fixed, `-2` learned | rejected |

It did play the intended hard negatives: the progress summary counted `997`
hard-negative games, but collector hard-negative WR moved
`0.678571 -> 0.529801`, matching the confirm128 rejection.

A low-cost model-space interpolation sweep from selected a015 toward extension
repair p5 was more stable:

| Candidate | Confirm64 paired delta | Confirm128 paired delta | Confirm256 paired delta | Status |
|---|---:|---:|---:|---|
| a125 | `0` fixed, `0` learned | not escalated | not escalated | diagnostic |
| a250 | `+1` fixed, `0` learned | not escalated | not escalated | diagnostic |
| a375 | `+1` fixed, `+3` learned | `+1` fixed, `+3` learned | `+1` fixed, `0` learned | stable but too small |
| a500 | `+1` fixed, `+3` learned | `0` fixed, `+3` learned | not escalated | diagnostic |

The a375 interpolation is the best stable intermediate artifact from this
round:
`runs/main_champion_hardneg_interp_a015_extensionrepair_seg01_p5_a375_20260518`.
It is not selected as the final main model because confirm256 only improves the
fixed surface by one paired win and ties the learned panel.

Finally, a short online continuation from a375 was tried to test whether the
stable interpolation could evolve further in league play. It again showed the
same structural failure:

| Candidate | Confirm64 paired delta | Confirm128 paired delta | Status |
|---|---:|---:|---|
| a375-online p2 | `+2` fixed, `+2` learned | `+2` fixed, `-3` learned | rejected |
| a375-online p3 | `+3` fixed, `+2` learned | `+2` fixed, `-3` learned | rejected |
| a375-online p4 latest | `+3` fixed, `-2` learned | not escalated | rejected |

This segment also played the intended opponents: `948` hard-negative games were
logged, but collector hard-negative WR moved `0.593750 -> 0.518041`.

Current conclusion: the strongest stable progress is model-space interpolation,
not additional IMPALA updates. Online league updates repeatedly buy fixed-row
gains by giving back learned champion/hard-negative performance. The next main
model attempt should change the learner objective or rollback gate, not merely
train longer with the same replay and sampling tools.

Current conclusion: longer league training is happening and the model is
getting live exposure to champions and hard negatives, but replay imitation of
selected-a015 wins is not sufficient. The next structural step should add
opponent-specific hard-negative pressure or a contrastive paired-swing objective
so improvement against fixed baselines does not come by losing older learned
regressors.

## Focused Old Hard Negatives

The next structural change added config-driven hard-negative focus sampling:

- `league.sampling.hard_negative_focus_policy_ids`
- `league.sampling.hard_negative_focus_weight_multiplier`

The focus IDs are suffix-matched after seed-snapshot import, so the config can
name stable source IDs while the runtime samples imported IDs. The focus
multiplier only changes sampling inside the hard-negative bucket; fixed B0-B4/B1
mix fractions are unchanged. Pool logs now record the focus IDs and multiplier.

Two probes were run:

| Probe | Confirm64 paired delta | Confirm128 paired delta | Status |
|---|---:|---:|---|
| focus old HN u5 `policy_000004` | `+5` all, `+2` fixed, `+3` learned | `-1` all, `+1` fixed, `-2` learned | rejected |
| strong focus full-guard u4 `policy_000004` | `+4` all, `+2` fixed, `+2` learned | `-1` all, `+1` fixed, `-2` learned | rejected |

This is better diagnosis than the replay-only repair: focused hard-negative
pressure produced the first full-panel confirm64 gains against both fixed and
learned opponents. But the gains did not survive confirm128, so no selected
alias was published.

The selected paper-ready model is still:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.

Current conclusion: opponent-specific pressure is necessary but not sufficient.
The next repair should target the confirm64-to-confirm128 split directly, likely
with a contrastive paired-swing objective or a disjoint seed repair set built
from candidate regressions rather than only selected-a015 imitation.

## B2-Retention and Exposure Audit Addendum

A stronger public-retention variant was tested after the focused old-hard-negative
probe:

- config:
  `configs/thesis/ablations/main_league_champion_hardneg_selected_alloutcome_focusoldhn_b2retention_b4b2guard_probe.yaml`
- run:
  `runs/main_champion_hardneg_alloutcome_focusoldhn_b2retention_u4x2_20260518_seg01`
- seed:
  selected a015 checkpoint from
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`

The run did play real league opponents:

| Exposure counter | Total |
|---|---:|
| champion envs | `498` |
| hard-negative envs | `466` |
| fixed B1/B2/B3/B4-style envs | `2549` via B1, B2, variants, and mirror buckets |
| sampled non-mirror league envs | `3675` |

The best live-training candidate from this probe was `policy_000003`:

| Surface | Confirm128 paired delta vs selected a015 |
|---|---:|
| all fixed plus learned rows | `+2` wins |
| fixed B0-B4+B1 | `+1` win |
| learned champion/hard-negative panel | `+1` win |

This is real progress relative to the earlier failed continuations, but it is
not enough to publish. The split diagnostic shows the edge came from the first
64 paired seeds and did not generalize to the confirm128 extension:

| Pair-index bucket | All | Fixed | Learned |
|---|---:|---:|---:|
| first 64 seeds | `+4` | `+2` | `+2` |
| extension 64 seeds | `-2` | `-1` | `-1` |

The next checkpoint, `policy_000004`, looked better at confirm64 but regressed
by confirm128:

| Candidate | Confirm64 delta | Confirm128 delta | Status |
|---|---:|---:|---|
| `policy_000003` | `+4` all, `+2` fixed, `+2` learned | `+2` all, `+1` fixed, `+1` learned | diagnostic only |
| `policy_000004` | `+5` all, `+3` fixed, `+2` learned | `-2` all, `+1` fixed, `-3` learned | rejected |

The collector trend also cautions against claiming stable league evolution:
imported-learned collector win rate moved from `0.631579` first observed to
`0.570485` last observed, and fixed-baseline collector win rate moved from
`0.687783` to `0.644737`. Those are unpaired training metrics, not thesis-grade
evidence, but they agree with the confirm128 drift warning.

Instrumentation was added so future longer segments can report not only bucket
totals but exact per-policy champion and hard-negative exposure. New scalar
summary fields include `policy_exposure_totals` and `policy_exposure_max` in
league progress summaries.

The selected paper-ready model remains unchanged:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.

Current conclusion: the league can now produce small paired gains while playing
champions and hard negatives, but the current recipe is not "way better" yet.
The next defensible step is a learned-panel repair that uses captured imported
champion/hard-negative episodes or paired-swing regressions as row-level
retention, then trains longer with per-policy exposure logging and the same
fixed plus learned paired confirmation ladder.

## Extension-Repair Longer Run Addendum

An extension-repair replay dataset was built to target the confirm128 extension
half where earlier candidates lost their confirm64 edge:

- seed file:
  `configs/seeds/confirm128_extension_repair_seeds_20260518.txt`
- merged replay dataset:
  `runs/trajectory_bc_selected_a015_all32_plus_confirm128ext16x3_20260518/trajectory_bc_selected_a015_all32_plus_confirm128ext16x3.npz`
- config:
  `configs/thesis/ablations/main_league_champion_hardneg_selected_alloutcome_focusoldhn_extensionrepair_b4b2guard_probe.yaml`

Two six-update guarded segments were run. Segment 1 correctly selected
`policy_000005` from update 5 rather than latest update 6; segment 2 continued
from that selected checkpoint and correctly refused to publish.

| Candidate | Confirm64 paired delta | Confirm128 paired delta | Status |
|---|---:|---:|---|
| seg01 `policy_000005` | `+4` fixed, `+3` learned | `+3` fixed, `-2` learned | rejected |
| seg02 `policy_000004` | `+2` fixed, `-4` learned | not escalated | rejected |
| seg02 `policy_000006` latest | `+1` fixed, `-3` learned | not escalated | rejected |

The confirm128 split for seg01 `policy_000005` explains the rejection:

| Pair-index bucket | Fixed B0-B4+B1 | Learned champion/hard-negative |
|---|---:|---:|
| first 64 seeds | `+4` | `+3` |
| extension 64 seeds | `-1` | `-5` |
| total confirm128 | `+3` | `-2` |

The run did train against real champions and hard negatives:

| Segment | Champion envs | Hard-negative envs | Sampled league envs | Imported learned WR first -> last | Hard-negative WR first -> last |
|---|---:|---:|---:|---:|---:|
| seg01 | `834` | `921` | `5957` | `0.700000 -> 0.548913` | `0.666667 -> 0.526316` |
| seg02 | `528` | `984` | `5931` | `0.571429 -> 0.539568` | `0.700000 -> 0.585106` |

These collector metrics are diagnostic and unpaired, but they agree with the
paired-confirm result: longer extension-repair training did not produce stable
evolution against champions/hard negatives while preserving baseline progress.

Two artifact robustness fixes were added during this phase:

- paired outcome comparison now resolves unique imported seed-suffix row IDs;
- league progress summary classifies seed-wrapped hard negatives correctly.
- paired swing reports now carry those seed-wrapped pool tags through to
  champion and hard-negative buckets.

The regenerated paired swing report
`diagnostics/paired_swing_report_extensionrepair_seg01_p5_confirm128_20260518.json`
shows the rejection is specifically a learned hard-negative retention failure:

| Group | Confirm128 paired delta |
|---|---:|
| all rows | `+1` |
| fixed B0-B4+B1 | `+3` |
| learned imported panel | `-2` |
| hard-negative tagged rows | `-2` |
| champion tagged rows | `+1` |

The replay seed plan now identifies concrete hard-negative regression episodes
for the next repair pass, including repeated losses around
`main_league_selected`, `policy_000003`, `checkpoint_000025`,
`main_bestresponse_u25_devbest`, and early champion policies.

The selected paper-ready model remains unchanged:
`runs/main_champion_hardneg_interp_u10_repair_a015_20260517`.
