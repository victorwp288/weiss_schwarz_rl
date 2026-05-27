# Thesis Figure Audit - 2026-05-21

## Scope

Inspected:

- thesis source directory:
  `C:/Users/Bruger/Desktop/this one/Kandidatspeciale`
- PDF render:
  `C:/Users/Bruger/Downloads/2026-05-14 Kandidatspeciale_no_images (2).pdf`
- source figures in:
  `Kandidatspeciale/Figures/new results`
  and `Kandidatspeciale/Figures/results_figures`
- newly locked K4 search artifacts from:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`

The supplied PDF is a `no_images` export, so the compiled result pages show
figure captions and layout positions but mostly blank image slots. I therefore
inspected the real source PNGs directly and used the PDF render only for page
flow, captions, and annotation context.

## Critical Findings

1. The current result graphs are outdated.

   The thesis result section still describes `policy_000021` and 64 paired-seed
   rows. The locked final evidence is now `main_league_selected` with K4
   same-world search at confirm256. The current figures no longer match the
   strongest model or final claim.

2. The old result figures are mostly readable, but the story is fragmented.

   `fig_main_targeted_robustness.png` is the best old plot because it shows all
   rows together. The smaller decomposition, B3/B4, promoted-margin, close-row,
   and seat-sensitivity figures each answer narrow questions, but together they
   make the reader assemble the result story manually.

3. Some old plots invite over-reading.

   `fig_close_promoted_stress.png` uses a zoomed y-axis around 50%. That is
   defensible for a close-row diagnostic, but it must be labelled as a zoomed
   parity stress check. Otherwise tiny differences look visually larger than
   they are.

4. `fig_b3b4_evaluation.png` mixes "completed games" and "win rate" on the same
   visual axis.

   That is confusing because completed games at `128/128` appears like a
   performance bar. Completion/truncation is better reported in text or a small
   diagnostic table, not as a competing bar beside win rate.

5. The gameplay figure layout needs work.

   The supervisor annotation on PDF page 20 is right: the gameplay subfigures
   should be aligned in a grid. The current sequence is visually heavy, spans
   multiple pages, and uses inconsistent rotation/cropping.

6. The result figures need to support the supervisor's larger request:
   concrete methodology.

   The comments are not just asking for prettier results. They ask for exact
   pipeline details: reward function, baselines, league training, opponent
   sampling, deck setup, evaluation protocol, masking, V-trace, and scripts.
   At least two new non-result figures/tables should be added:

   - an experimental pipeline diagram;
   - a baseline/deck/evaluation table;
   - optionally a league-training pseudocode box or flow diagram.

## New Thesis Figure Pack

Generated a cleaned, thesis-facing K4 search figure pack in:

```text
C:/Users/Bruger/Desktop/this one/Kandidatspeciale/Figures/main_search_20260521
```

This is the preferred pack to use in the thesis. It uses human-facing labels
such as `B1 No-League`, `B2 Public heuristic`, `League policy 1`, and
`Imported main selected`, rather than raw run IDs.

The earlier provenance-oriented pack remains available in:

```text
C:/Users/Bruger/Desktop/this one/Kandidatspeciale/Figures/god_search_20260521
```

Core generated figures:

```text
paper/main_search_strength_ladder.png
paper/main_search_confirm256_row_win_rates.png
paper/main_search_confirm256_delta_wins.png
paper/main_search_confirm256_group_rates.png
paper/main_search_validation_progression.png
paper/main_search_decision_changes.png
paper/main_search_seat_balance.png
paper/main_search_first_second_balance.png
```

Each has a PDF version in the same folder. Data sidecars are in:

```text
data/
```

### Restyle Pass

After visual inspection, the pack was restyled to remove overlapping text and
reduce default plotting clutter:

- moved crowded legends outside the plotting area;
- removed unnecessary single-series legends;
- switched win-rate axes to percentages;
- added percentage labels to group-rate bars;
- simplified the decision-change diagnostic to a single "additional wins" bar
  series;
- widened the seat-balance and strength-ladder layouts so value labels no
  longer collide with opponent names.

Visual contact sheet:

```text
diagnostics/thesis_figure_audit_20260521/main_search_restyled_contact_sheet.png
```

## Recommended Figure Set for the Revised Results

Use this as the main Results sequence:

1. `main_search_strength_ladder`

   Purpose: show the thesis arc from B1 NoLeague to selected no-search to K4
   search. This gives the reader a quick "what improved?" view.

2. `main_search_confirm256_group_rates`

   Purpose: one clean grouped result: fixed baselines, learned/hard negatives,
   and all rows.

3. `main_search_confirm256_row_win_rates`

   Purpose: full row-level evidence. This replaces old
   `fig_main_targeted_robustness.png`.

4. `main_search_confirm256_delta_wins`

   Purpose: paired-seed improvement versus selected no-search. This is stronger
   than just showing raw K4 win rates because it directly answers whether K4
   improved the selected model.

5. `main_search_validation_progression`

   Purpose: show that K4 was escalated through confirm64, confirm128, and
   confirm256 rather than selected from a tiny smoke test.

Optional mechanism/diagnostic figures:

- `main_search_decision_changes`
- `main_search_seat_balance`
- `main_search_first_second_balance`

Use these if there is room or in an appendix. The changed-decision plot is a
nice bridge between "it won more" and "the search mechanism actually changed
decisions." The seat-balance plot is only a seat-slot diagnostic; use
`main_search_first_second_balance` when discussing whether the focal player was
better going first or second.

## Figures to Keep, Replace, or Move

- Replace:
  - `Figures/new results/fig_main_targeted_robustness.png`
  - `Figures/new results/fig_result_decomposition.png`
  - `Figures/new results/fig_b3b4_evaluation.png`
  - `Figures/new results/fig_promoted_margin_ladder.png`
  - `Figures/new results/fig_close_promoted_stress.png`

- Move to appendix or historical no-search section:
  - old promoted-snapshot robustness plots;
  - old algorithm/anchor ablation figures from `results_figures`.

- Keep only if text still discusses the no-search plateau:
  - the old close-promoted stress and seat-sensitivity diagnostics.

## Suggested New Methodology Visuals

These are not generated yet, but they would directly answer supervisor feedback:

1. Experimental pipeline diagram:
   simulator -> actor rollout -> learner/V-trace -> snapshot registry ->
   targeted confirm -> figures/readiness.

2. League opponent ecology diagram:
   fixed anchors, B1 baseline, public heuristics, champions, hard negatives,
   selected/best checkpoint.

3. Baseline table:
   policy, hidden-information access, deck, behavior, script/config, purpose.

4. Evaluation protocol schematic:
   paired seeds, seat swap, fixed decks, bootstrap uncertainty, confirm64 ->
   confirm128 -> confirm256.

These are at least as important as the final result plots because the
supervisor's strongest feedback is that the pipeline is currently too implicit.

## Defensibility Note

The new K4 figures must be captioned as same-world decision-time search. Do not
present them as the raw trained policy. Suggested caption phrase:

> K4 search denotes the locked main league policy wrapped in one-decision
> same-world prefix-rollout search. It is reported separately from the
> no-search policy because the wrapper evaluates candidate actions by replaying
> the sampled hidden world.

Seat and turn-order diagnostics need one extra caveat. In the eval records,
`seat0` and `seat1` are policy assignment slots. The simulator sets the actual
starting player from `episode_seed & 1`, so paired seed swaps give the focal
policy one game as first player and one game as second player for each seed.
Do not describe `seat1 - seat0` as a first-vs-second advantage.
