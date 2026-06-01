# Thesis Experiment Findings And Config Notes - 2026-05-07

This file is the compact master note for the emergency thesis result package. It records what was run, which configs matter, which checkpoint is currently selected, and what the preliminary findings support.

## Executive Summary

The current thesis-ready main result is the legacy-runtime League GRU model `policy_000021` from:

`runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506`

The selected model is defensible for Section 7 if described carefully:

- It solves the fixed public anchors B0 and B2.
- It retains strong B1 no-league performance.
- It beats corrected B3/B4 heuristic variants.
- It remains above parity against all tested legacy promoted league snapshots.
- The late legacy rows p15/p16 are narrow positive margins, not decisive wins.
- The result is legacy-simulator comparable, not a clean current-simulator result.

## Current Code / Branch State

Remote Vast repo:

`/root/wsrl-exp034-legacy/weiss_schwarz_rl`

Current remote branch:

`thesis-confirm64-eval-20260507`

Important commits:

- `085acf3 docs: add local thesis diagnostic figures`
- `d0c68e8 docs: finalize thesis confirm64 eval package`
- `2e872cd fix: repair b3 b4 heuristic eval loop`

The branch has not been pushed.

## Runtime And Hardware

Main run runtime:

- Python: `3.11.15`
- Torch: `2.7.0+cu128`
- Simulator: `weiss-sim 0.7.0`
- Platform: Linux on Vast
- CPU count: `128`
- Learner device: `cuda:0`
- Actor device layout: `cuda:0`

Why legacy runtime was used:

- The old exp034 checkpoint/runtime used simulator `0.7.0`.
- The current simulator changed feature metadata, including card feature dimension behavior.
- That changed the model input graph, so exact checkpoint continuation on the current simulator was not cleanly compatible.
- For a defensible thesis result under time pressure, we used the legacy exp034 runtime to preserve comparability with the strongest previous local result.

## Main Training Lineage

Main run:

`runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506`

Launch command recorded in `environment.json`:

```bash
python/scripts/train.py \
  --stack-config configs/presets/structured_acceptance_standard_auto_gpu_exp031_champion_reserved_b1_lane.yaml \
  --run-label main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506 \
  --resume-run-dir runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506 \
  --resume-from runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/training/checkpoints/checkpoint_360.pt \
  --num-envs 8 \
  --unroll-length 4 \
  --max-updates 800 \
  --runtime-mode train_ordered \
  --b1-baseline-run-dir runs/exp-002-current-spec-b1-noleague-baseline \
  --seed-snapshot-run-dir runs/exp034_legacy_oldleague_env8_u4_cont_u320_to340_20260506 \
  --override system.collection_backend=auto
```

Important update counts:

- Main run intended max updates: `800`.
- Recorded latest checkpoint: update `660`, `policy_000033`.
- Checkpoint tracker best by dev-eval mean: update `350`, `policy_000017`, dev-eval mean `0.9583`.
- Selected thesis policy: `policy_000021`, corresponding to update `425` in the dev-eval summaries.
- Later candidate checked: `policy_000033`, but it was not selected because it weakened B1, B3/B4, p11, p14, and p15 relative to p21.

## Main Config

Main config file:

`configs/presets/structured_acceptance_standard_auto_gpu_exp031_champion_reserved_b1_lane.yaml`

Purpose:

`exp-031 champion pressure with explicit reserved B1 lane`

Important high-level settings:

- Algorithm: `impala_vtrace_structured_v1`
- Recurrent core: `gru`
- GRU hidden size: `32`
- Encoder kind: `structured_v2`
- Encoder MLP width: `32`
- Encoder MLP layers: `2`
- Typed feature width: `8`
- Layer norm: enabled
- Learning rate: `0.0002`
- Entropy coefficient: `0.03`, anneal target `0.01`
- Mixed precision: enabled
- Structured warmstart: disabled
- Teacher auxiliary mode: `always`
- Reward objective: `terminal_pm1`
- Damage shaping: enabled, `damage_reward = 0.05`
- Truncation reward: `-0.1`
- Discount gamma: `0.99`

Actual launch overrides:

- `--num-envs 8`
- `--unroll-length 4`
- `--max-updates 800`

Note: the canonical config still contains inherited rollout/system fields such as `rollout.unroll_length = 64`, `actor_process_count = 32`, `envs_per_actor = 8`, and `total_envs = 96`. For exact reproduction of the emergency run, use the recorded launch command above, especially the CLI overrides `--num-envs 8` and `--unroll-length 4`.

## League Sampling Config

Main league settings:

- League enabled: yes
- Recent size: `24`
- Champion size: `4`
- Champion max age updates: `0`
- Warmup first updates: `0`
- Initial window episodes: `512`
- Ramp target updates: `1`
- Ramp target window episodes: `4096`
- Opponent sampling: `PFSP`
- PFSP epsilon uniform: `0.20`
- PFSP power: `1.5`
- PFSP stats source: `online_outcomes`
- Champion mix fraction: `0.55`
- Hard negative mix fraction: `0.15`
- Hard negative min samples: `4`
- Hard negative max win rate: `0.55`
- B1 no-league mix fraction: `0.15`
- B1 no-league mix end updates: `-1` (kept on)
- B1 no-league reserved envs per actor: `1`
- Heuristic public mix fraction: `0.0`
- Heuristic public reserved envs per actor: `0`

Interpretation:

The main model uses constant B1 pressure plus champion/hard-negative league pressure. This is the key design difference from the no-B1-lane and weak-B1 variants.

## Promotion / Dev Eval Config

Promotion gate:

- Required anchors: `B0 RandomLegal`, `B1 NoLeague baseline`
- Optional if available: `B2 HeuristicPublic`
- Paired seeds: `8`
- Seat swap: enabled
- Uncertainty method: `bayesian_bootstrap_seedlevel_v1`
- Threshold: `P(p_anchor > 0.55) > 0.95 using AnchorSet_v1`
- Guardrail max truncation rate: `0.05`

Periodic dev eval:

- Interval: every `25` updates
- Paired seeds: `8`
- Main dev eval updates observed: `350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650`

Main dev-eval endpoint at update `650`:

- Aggregate score: `0.9375`
- B0: `1.0`
- B1: `0.8125`
- B2: `1.0`

## Evaluation Configs

Fast GPU eval config:

`configs/presets/eval_gpu_exp031_fast_20260506.yaml`

This extends the main exp031 config and sets:

- `evaluation.eval_device = cuda`
- `final_matrix_stage1_paired_seeds = 16`
- `final_matrix_stage2_adaptive_max_paired_seeds = 16`

Targeted confirm scripts then overrode paired-seed counts directly:

- Headline confirm64 rows: `64` paired seeds, `128` games per row.
- Close p15/p16 stress check: `128` paired seeds, `256` games per row.

## Selected Main Results

Selected policy:

`policy_000021`

Headline table:

| Opponent | Wins | Games | Win rate | Truncations | Engine errors |
|---|---:|---:|---:|---:|---:|
| B0 RandomLegal | 128 | 128 | 100.00% | 0 | 0 |
| B1 NoLeague baseline | 100 | 128 | 78.12% | 0 | 0 |
| B2 HeuristicPublic | 128 | 128 | 100.00% | 0 | 0 |
| B3 HeuristicPublicAggro | 82 | 128 | 64.06% | 0 | 0 |
| B4 HeuristicPublicControl | 83 | 128 | 64.84% | 0 | 0 |
| Legacy p11 | 76 | 128 | 59.38% | 0 | 0 |
| Legacy p12 | 69 | 128 | 53.91% | 0 | 0 |
| Legacy p14 | 69 | 128 | 53.91% | 0 | 0 |
| Legacy p15 | 65 | 128 | 50.78% | 0 | 0 |
| Legacy p16 | 65 | 128 | 50.78% | 0 | 0 |

Aggregate headline table:

- Overall: `865/1280 = 67.58%`
- Fixed anchors B0/B2: `256/256 = 100.00%`
- B1: `100/128 = 78.12%`
- B3/B4: `165/256 = 64.45%`
- Legacy neural subset p11-p16: `344/640 = 53.75%`
- B1 plus legacy neural subset: `444/768 = 57.81%`

Close-row stress check:

| Opponent | Wins | Games | Win rate | Interpretation |
|---|---:|---:|---:|---|
| Legacy p15 | 129 | 256 | 50.39% | narrow above-parity result; CI overlaps parity |
| Legacy p16 | 129 | 256 | 50.39% | narrow above-parity result; CI overlaps parity |

## Seat Diagnostics

Across the full confirm64 headline table:

- First seat: `419/640 = 65.47%`
- Second seat: `446/640 = 69.69%`
- Difference: second seat advantage of `+4.22` percentage points.

Interpretation:

There is a modest second-seat advantage in these artifacts, especially on later legacy neural rows, but the reported evaluations are paired and seat-swapped. Therefore the seat asymmetry is diagnostic rather than a confound in the headline win rates.

## B3/B4 Fix

Initial B3/B4 rows looked catastrophic (`0/32` or `0/128`), but inspection showed they were invalid evaluation artifacts:

- `wins = 0`
- `losses = 0`
- `truncations = games`
- `timeout_unknown = games`
- Nearly all actions were `main_move`

Cause:

The aggressive/control heuristic profiles could repeatedly select neutral or bad `main_move` actions because their move priority outranked pass. This created deterministic loops until the action cap.

Fix:

`python/weiss_rl/eval/heuristic_public.py`

Behavior after fix:

Aggressive/control profiles may prefer beneficial repositioning, but neutral or bad `main_move` no longer outranks pass.

Post-fix evidence:

- B3: `82/128 = 64.06%`, `0` truncations
- B4: `83/128 = 64.84%`, `0` truncations
- B3/B4 combined: `165/256 = 64.45%`, `0` truncations

Thesis wording:

Do not report the old B3/B4 zeros as model losses. They were invalid heuristic-loop evaluation artifacts.

## Later Candidate Check

Later snapshot:

`policy_000033`

Why it was checked:

It looked promising in a small sweep, and it was the latest recorded policy from the run.

Confirm32 p33 results:

| Opponent | Wins | Games | Win rate |
|---|---:|---:|---:|
| B1 NoLeague baseline | 48 | 64 | 75.00% |
| B3 HeuristicPublicAggro | 36 | 64 | 56.25% |
| B4 HeuristicPublicControl | 36 | 64 | 56.25% |
| Legacy p11 | 37 | 64 | 57.81% |
| Legacy p12 | 36 | 64 | 56.25% |
| Legacy p14 | 32 | 64 | 50.00% |
| Legacy p15 | 32 | 64 | 50.00% |
| Legacy p16 | 36 | 64 | 56.25% |

Overall:

`293/512 = 57.23%`

Decision:

Keep `policy_000021`. p33 improved p12 and p16, but weakened B1, B3/B4, p11, p14, and p15. The thesis needs broad robustness, so p21 is the safer primary model.

## Baselines And Ablations

Runs/artifacts:

- No B1 lane ablation: `runs/ablation_exp028_no_b1_lane_override_from_exp023_to420_20260506`
- Weak B1 guardrail ablation: `runs/ablation_exp029_weak_b1_auto_from_exp023_to420_20260506`
- No-GRU baseline: `runs/baseline_nogru_impala_fixed_heuristic_u220_20260506`
- PPO-lite baseline: `runs/baseline_ppo_lite_fixed_heuristic_u220_20260506`

Main ablation configs:

- `configs/presets/structured_acceptance_standard_auto_gpu_exp028_champion_fast_eval.yaml`
- `configs/presets/structured_acceptance_standard_auto_gpu_exp029_champion_b1_guardrail.yaml`

Exp028 no-B1-lane config:

- B1 reserved envs per actor: `0`
- B1 mix fraction: `0.0`
- Champion mix fraction: `0.80`
- Hard negative mix fraction: `0.15`

Exp029 weak-B1 config:

- B1 reserved envs per actor: `0`
- B1 mix fraction: `0.10`
- Champion mix fraction: `0.70`
- Hard negative mix fraction: `0.15`

Dev-eval endpoint summaries:

| Run | Last observed update | B0 | B1 | B2 | Aggregate |
|---|---:|---:|---:|---:|---:|
| Main p21 lineage | 650 | 100.0% | 81.25% | 100.0% | 93.75% |
| Exp028 no B1 lane | 400 | 100.0% | 81.25% | 100.0% | 93.75% |
| Exp029 weak B1 mix | 400 | 100.0% | 81.25% | 93.75% | 91.67% |

Important caveat:

No-GRU and PPO-lite are fixed-opponent baselines, not full league-robustness baselines. Their matrices include B0/B2-style fixed opponents but do not include the same B1/legacy league opponent coverage as the main model.

## Figures Generated

Figure directory:

`thesis_figures_final`

Primary Section 7 figures:

- `fig_main_targeted_robustness.pdf`
- `main_p21_results_table.tex`
- `fig_result_decomposition.pdf`
- `fig_legacy_margin_ladder.pdf`
- `fig_close_legacy_stress.pdf`
- `fig_b3b4_fixed_validation.pdf`
- `fig_p21_seat_advantage.pdf`

Baselines/ablation figures:

- `fig_anchor_retention.pdf`
- `fig_anchor_ablation_endpoints.pdf`
- `fig_baseline_fixed_grid.pdf`
- `fig_candidate_p21_vs_p33.pdf`

Use cautiously or omit:

- `fig_fast_matrix_sanity.pdf`: low-game sanity matrix only.
- `fig_training_loss_diagnostic.pdf`: optimization diagnostic only; evaluation win rates are more meaningful.

Generated helper files:

- `thesis_figures_final/FIGURE_CAPTIONS.md`
- `thesis_figures_final/main_p21_results_table.tex`
- `thesis_figures_final/section7_figure_snippets.tex`

## Artifact Map

Local copied artifacts:

- `vast_artifacts/main/p21_b1_legacy_confirm64_summary.json`
- `vast_artifacts/main/p21_b3b4_loopfix_confirm64_summary.json`
- `vast_artifacts/main/confirm64_rows/p21_vs_b0_summary.json`
- `vast_artifacts/main/confirm64_rows/p21_vs_b2_summary.json`
- `vast_artifacts/main/p21_p15_p16_confirm128_summary.json`
- `vast_artifacts/main/p33_b1_b3b4_legacy_confirm32_summary.json`
- `vast_artifacts/main/seat_diagnostics/p21_headline_confirm64_seat_diagnostics.json`
- `vast_artifacts/main/dev_eval_summaries.json`
- `vast_artifacts/exp028/dev_eval_summaries.json`
- `vast_artifacts/exp029/dev_eval_summaries.json`
- `vast_artifacts/nogru/final_summary.json`
- `vast_artifacts/ppo/final_summary.json`

Remote eval artifact roots:

- `runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/p21_b1_legacy_confirm64_loopfix/targeted_confirm64_summary.json`
- `runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/p21_b3b4_loopfix_confirm64/targeted_confirm64_summary.json`
- `runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/eval/p21_p15_p16_confirm128_loopfix/targeted_confirm128_summary.json`

## Recommended Thesis Claims

Safe claim:

> The selected League GRU policy solves the fixed public anchors, remains robust to stronger B3/B4 public heuristic variants after correcting an invalid heuristic-loop artifact, and retains modest but consistently above-parity performance against legacy promoted league snapshots.

Safe quantitative sentence:

> In targeted confirmation, the model achieved 100% against B0 and B2, 78.1% against B1, 64.1% and 64.8% against B3 and B4, and 50.8%-59.4% against legacy promoted policies, with every row evaluated over 128 games and no truncations or engine errors.

Close-row caveat:

> The two closest legacy rows, p15 and p16, were additionally checked over 256 games each and remained slightly above parity (`129/256 = 50.4%`), but the margins are narrow and should be interpreted cautiously.

## Claims To Avoid

- Do not claim this is a clean current-simulator result.
- Do not claim the model strongly dominates every legacy league opponent.
- Do not report the old B3/B4 zero rows as losses.
- Do not claim No-GRU/PPO-lite have the same opponent coverage as the main model.
- Do not use the tiny fast matrix as headline quantitative evidence.
- Do not overinterpret actor-critic loss curves; use win-rate evals as the main evidence.

## Reproduction Commands

Regenerate figures locally:

```bash
python3 scripts/make_thesis_figures.py
```

Main confirm64 eval pattern:

```bash
.venv-exp034/bin/python python/scripts/targeted_confirm_eval.py \
  --stack-config configs/presets/eval_gpu_exp031_fast_20260506.yaml \
  --run-dir runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506 \
  --snapshot-registry-json runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506/training/snapshots/registry.json \
  --b1-baseline-run-dir runs/exp-002-current-spec-b1-noleague-baseline \
  --focal-policy-id policy_000021 \
  --paired-seeds 64 \
  --workers 6
```

Exact main training command is recorded above under "Main Training Lineage".

## Current Bottom Line

The package is thesis-worthy if reported honestly. The strongest result is not that p21 crushes every opponent. The strongest result is that the selected League GRU model preserves anchor competence, survives B1 pressure, beats corrected stronger public heuristics, and stays above parity against all tested promoted legacy snapshots in a uniform confirm64 table, with additional confirm128 stress evidence on the two closest rows.
