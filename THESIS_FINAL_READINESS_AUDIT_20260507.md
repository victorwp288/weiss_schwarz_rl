# Thesis Final Readiness Audit - 2026-05-07

## Verdict

The main result package is thesis-usable if reported with the comparability caveats below. The main model result is now substantially stronger than the earlier 32-game table: every headline row is at least confirm32, and B0/B2/B3/B4 are confirm64.

The safest thesis framing is:

- Primary model: `policy_000021`, main legacy League GRU model.
- Primary evidence: targeted confirmation table, not the tiny fast final matrix.
- Baselines: useful but not all equally comparable. No-GRU and PPO-lite are fixed-opponent baselines, not full league-robustness ablations.
- B3/B4: valid only after the heuristic move-loop fix; old B3/B4 zero rows are invalid truncation artifacts and must not be reported as losses.

## Main Headline Table

| Opponent | Wins | Games | Win rate | Evidence level | Truncations | Engine errors |
|---|---:|---:|---:|---|---:|---:|
| B0 RandomLegal | 128 | 128 | 100.00% | confirm64 | 0 | 0 |
| B1 NoLeague baseline | 50 | 64 | 78.12% | confirm32 | 0 | 0 |
| B2 HeuristicPublic | 128 | 128 | 100.00% | confirm64 | 0 | 0 |
| B3 HeuristicPublicAggro | 82 | 128 | 64.06% | confirm64 after heuristic-loop fix | 0 | 0 |
| B4 HeuristicPublicControl | 83 | 128 | 64.84% | confirm64 after heuristic-loop fix | 0 | 0 |
| Legacy p11 | 40 | 64 | 62.50% | confirm32 | 0 | 0 |
| Legacy p12 | 34 | 64 | 53.12% | confirm32 | 0 | 0 |
| Legacy p14 | 36 | 64 | 56.25% | confirm32 | 0 | 0 |
| Legacy p15 | 34 | 64 | 53.12% | confirm32 | 0 | 0 |
| Legacy p16 | 33 | 64 | 51.56% | confirm32 | 0 | 0 |

Combined over the displayed table: `648/896 = 72.32%`. This aggregate is useful as a broad summary, but the thesis should emphasize the per-opponent rows because the table mixes confirm32 and confirm64 evidence.

Subsets:

- Fixed public anchors B0/B2: `256/256 = 100.00%`.
- B3/B4 after fix: `165/256 = 64.45%`.
- B1 plus legacy neural opponents: `227/384 = 59.11%`.
- Legacy-only neural subset: `177/320 = 55.31%`.

## Artifact Sources

Main run:

- `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506`

Local copied artifacts:

- `vast_artifacts/main/confirm64_rows/p21_vs_b0_summary.json`
- `vast_artifacts/main/confirm64_rows/p21_vs_b2_summary.json`
- `vast_artifacts/main/p21_b3b4_loopfix_confirm64_summary.json`
- `vast_artifacts/main/p21_b1_legacy_confirm32_summary.json`

Main figures:

- `thesis_figures_final/fig_main_targeted_robustness.pdf`
- `thesis_figures_final/fig_b3b4_fixed_validation.pdf`
- `thesis_figures_final/fig_b3b4_seat_balance.pdf`
- `thesis_figures_final/fig_anchor_retention.pdf`
- `thesis_figures_final/fig_baseline_fixed_grid.pdf`

LaTeX helpers:

- `thesis_figures_final/main_p21_results_table.tex`
- `thesis_figures_final/section7_figure_snippets.tex`

## Comparability Assessment

### Main vs B1-pressure ablations

Comparable enough for ablation discussion:

- Same simulator/runtime: `weiss-sim 0.7.0`, Python `3.11.15`, Torch `2.7.0+cu128`.
- Same max decisions: `2000`.
- Same high-level training scale: `num-envs 8`, `unroll-length 4`.
- Same model family: structured encoder, GRU recurrent core, GRU hidden size 32, typed feature width 8.

Use the B1-pressure ablations primarily for the anchor-retention story and B1-pressure discussion.

### Main vs No-GRU / PPO-lite

Not directly comparable as full final league evaluations:

- No-GRU/PPO-lite use different model settings and fixed-opponent matrices.
- Their available matrices include B0 and B2 but do not include B1 or legacy league opponents.
- They are still useful as weaker baselines showing that simpler/fixed-opponent setups do not match the main model’s fixed-anchor performance.

Thesis wording should say “fixed-opponent baselines,” not “full league robustness baselines.”

### Fast final matrices

The final matrix sanity artifacts are low-game sanity checks, not headline evidence. Example: rows are often 8 games. Keep `fig_fast_matrix_sanity` as an artifact/sanity figure only, or omit it from the main results section.

## B3/B4 Validity

The old B3/B4 `0%` rows were invalid:

- `wins=0`, `losses=0`, `truncations=games`.
- Termination reason was `timeout_unknown`.
- Episodes hit the action cap with almost all actions as `main_move`.

Fix applied:

- File: `python/weiss_rl/eval/heuristic_public.py`
- Behavior: aggressive/control profiles may prefer beneficial repositioning, but neutral or bad `main_move` actions no longer outrank pass.

Post-fix B3/B4 rows:

- B3: `82/128`, `0` truncations.
- B4: `83/128`, `0` truncations.
- Seat diagnostics: p21 remains above 50% from both first and second seat.

## Current Remote Commits

Important Vast commits:

- `2e872cd fix: repair b3 b4 heuristic eval loop`
- `0f26d4d docs: add thesis result figures`
- `62ff27d docs: strengthen p21 confirmation results`
- `689343e docs: add section 7 latex snippets`

No train/eval jobs were running at the time of this audit.

## What To Claim

Recommended claim:

> The selected League GRU policy solves the fixed public anchors, remains robust to stronger B3/B4 public heuristic variants after correcting an invalid heuristic-loop evaluation artifact, and retains modest but consistently above-parity performance against legacy promoted league snapshots.

Recommended quantitative wording:

> In targeted confirmation, the model achieved 100% against B0 and B2 over 128 games each, 78.1% against B1 over 64 games, 64.1% and 64.8% against B3 and B4 over 128 games each, and 51.6%-62.5% against legacy promoted policies over 64 games each.

## What Not To Claim

- Do not claim this is a clean current-simulator result; it is a legacy exp034-runtime result for comparability.
- Do not report the old B3/B4 zeros as losses.
- Do not claim every row is confirm64; B1 and legacy rows are confirm32.
- Do not claim No-GRU/PPO-lite have the same opponent coverage as the main model.
- Do not use the tiny fast final matrix as headline quantitative evidence.

## Remaining Optional Improvements

If more time is available, the only major quantitative upgrade would be a full overnight confirm64 for B1 and legacy neural opponents. It is not necessary for a defensible result package because confirm32 is already much stronger than the original 32-game table, but confirm64 would make the table more uniform.

