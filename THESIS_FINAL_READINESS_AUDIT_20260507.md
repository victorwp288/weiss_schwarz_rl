# Thesis Final Readiness Audit - 2026-05-07

## Verdict

The main result package is thesis-usable if reported with the comparability caveats below. The main model result is now substantially stronger than the earlier 32-game table: every headline row is confirm64, using 64 paired seeds / 128 seat-swapped games per opponent.

The safest thesis framing is:

- Primary model: `policy_000021`, main legacy League GRU model.
- Primary evidence: targeted confirmation table, not the tiny fast final matrix.
- Baselines: useful but not all equally comparable. No-GRU and PPO-lite are fixed-opponent baselines, not full league-robustness ablations.
- B3/B4: valid only after the heuristic move-loop fix; old B3/B4 zero rows are invalid truncation artifacts and must not be reported as losses.
- Close legacy rows: p15 and p16 were stress-checked at confirm128 and remained narrowly positive (`129/256` each), but their confidence intervals overlap parity.

After a later-checkpoint sweep, `policy_000021` remains the recommended thesis model. Later `policy_000033` looked promising in a small 8-paired-seed stress sweep, but did not hold up in confirm32: it underperformed p21 on B1, B3/B4, p11, p14, and p15, while only improving p12 and p16.

## Main Headline Table

| Opponent | Wins | Games | Win rate | Evidence level | Truncations | Engine errors |
|---|---:|---:|---:|---|---:|---:|
| B0 RandomLegal | 128 | 128 | 100.00% | confirm64 | 0 | 0 |
| B1 NoLeague baseline | 100 | 128 | 78.12% | confirm64 | 0 | 0 |
| B2 HeuristicPublic | 128 | 128 | 100.00% | confirm64 | 0 | 0 |
| B3 HeuristicPublicAggro | 82 | 128 | 64.06% | confirm64 after heuristic-loop fix | 0 | 0 |
| B4 HeuristicPublicControl | 83 | 128 | 64.84% | confirm64 after heuristic-loop fix | 0 | 0 |
| Legacy p11 | 76 | 128 | 59.38% | confirm64 | 0 | 0 |
| Legacy p12 | 69 | 128 | 53.91% | confirm64 | 0 | 0 |
| Legacy p14 | 69 | 128 | 53.91% | confirm64 | 0 | 0 |
| Legacy p15 | 65 | 128 | 50.78% | confirm64 | 0 | 0 |
| Legacy p16 | 65 | 128 | 50.78% | confirm64 | 0 | 0 |

Combined over the displayed table: `865/1280 = 67.58%`. This aggregate is useful as a broad summary, but the thesis should emphasize the per-opponent rows because the opponent set mixes fixed anchors, heuristic variants, and legacy neural snapshots.

Subsets:

- Fixed public anchors B0/B2: `256/256 = 100.00%`.
- B3/B4 after fix: `165/256 = 64.45%`.
- B1 plus legacy neural opponents: `444/768 = 57.81%`.
- Legacy-only neural subset: `344/640 = 53.75%`.

## Artifact Sources

Main run:

- `/root/wsrl-exp034-legacy/weiss_schwarz_rl/runs/main_thesis_exp034_legacy_oldleague_env8_u4_u340_to800_20260506`

Local copied artifacts:

- `vast_artifacts/main/confirm64_rows/p21_vs_b0_summary.json`
- `vast_artifacts/main/confirm64_rows/p21_vs_b2_summary.json`
- `vast_artifacts/main/p21_b3b4_loopfix_confirm64_summary.json`
- `vast_artifacts/main/p21_b1_legacy_confirm64_summary.json`
- Close-row stress check: `vast_artifacts/main/p21_p15_p16_confirm128_summary.json`
- Candidate check: `vast_artifacts/main/p33_b1_b3b4_legacy_confirm32_summary.json`

Main figures:

- `thesis_figures_final/fig_main_targeted_robustness.pdf`
- `thesis_figures_final/fig_b3b4_fixed_validation.pdf`
- `thesis_figures_final/fig_b3b4_seat_balance.pdf`
- `thesis_figures_final/fig_p21_seat_advantage.pdf`
- `thesis_figures_final/fig_close_legacy_stress.pdf`
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

## Candidate Selection Check

A small sweep over later snapshots (`policy_000022`, `policy_000025`, `policy_000029`, `policy_000033`) suggested that `policy_000033` might be competitive. It was then evaluated more carefully on B1, B3/B4, and the five legacy neural opponents.

`policy_000033` confirm32 results:

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

Overall p33 on that candidate set: `293/512 = 57.23%`.

Decision: keep `policy_000021`. p33 improves p12 and p16, but it weakens B1, B3/B4, p11, p14, and p15. Since the thesis needs broad robustness rather than one or two better legacy rows, p21 is still the better primary model.

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

## Seat Advantage

The headline evaluations are seat-swapped, so first/second-seat effects can be measured directly from diagnostics.

For `policy_000021` across the headline table:

- First seat: `419/640 = 65.47%`
- Second seat: `446/640 = 69.69%`
- Difference: second seat is `+4.22` percentage points.

Interpretation: there is a modest second-seat advantage in these artifacts. It is not driving the headline result because all reported evaluations are paired/seat-swapped, but it is worth mentioning as a diagnostic. The advantage is largest against late legacy neural opponents p14-p16.

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

> In targeted confirmation, the model achieved 100% against B0 and B2, 78.1% against B1, 64.1% and 64.8% against B3 and B4, and 50.8%-59.4% against legacy promoted policies, with every row evaluated over 128 games and no truncations or engine errors.

Optional close-row caveat:

> The two closest legacy rows, p15 and p16, were additionally checked at 256 games each and remained slightly above parity (`129/256 = 50.4%`), but the margin is narrow and should be interpreted cautiously.

## What Not To Claim

- Do not claim this is a clean current-simulator result; it is a legacy exp034-runtime result for comparability.
- Do not report the old B3/B4 zeros as losses.
- Do not overclaim the close legacy rows; p15 and p16 are positive but narrow at `65/128`.
- Do not claim No-GRU/PPO-lite have the same opponent coverage as the main model.
- Do not use the tiny fast final matrix as headline quantitative evidence.

## Remaining Optional Improvements

If more time is available, the only major quantitative upgrade would be a wider confirm128 or confirm256 over all legacy neural rows. That is optional; the current confirm64 table plus p15/p16 confirm128 stress check is already uniform enough for a defensible thesis result.
