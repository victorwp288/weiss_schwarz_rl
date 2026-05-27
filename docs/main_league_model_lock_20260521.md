# Main League Model Lock

Date: 2026-05-21

Repo: `C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl`

This file is the short future-me lock note. The longer curated report is
`docs/main_league_rebuild_report_20260521.md`; the raw chronological record is
`docs/rebuild_log.md`.

## Locked Thesis Main Model

Use this model as the thesis main model unless a future candidate passes the
full no-regression gate through confirm256.

- run: `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- selected source policy id: `main_interp_repair_a015`
- published alias: `main_league_selected`
- update: `5`
- weights hash:
  `1a13b49b73ed71af0914c97fede5b30703eb576a5e85c4c636310c2d76897b26`
- registry:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry.json`
- selection summary:
  `diagnostics/main_champion_hardneg_interp_u10_repair_a015_selected.json`

The snapshot registry already pins both `main_interp_repair_a015` and
`main_league_selected`, and both point to weights with the same hash above.

Do not replace this model from confirm64 or confirm128 evidence. Do not replace
it because scalar loss, replay accuracy, or aggregate-only score improved.

## Fixed-Deck Result

Primary thesis surface:

- focal/main/B0/B1/B2: `preset:main_deck_5hy_yotsuba_v1`
- B3 aggro: `preset:aggro_deck_5hy_nino_v1`
- B4 control: `preset:control_deck_jj_s66_v1`

Confirm256 fixed B0-B4+B1 evidence:

- summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/targeted_confirm256_b0_b4/targeted_confirm256_summary.json`
- aggregate: `1980/2560 = 0.773438`

| Opponent | Wins | Games | Win rate |
|---|---:|---:|---:|
| B0 RandomLegal | 512 | 512 | 1.000000 |
| B1 NoLeague baseline | 322 | 512 | 0.628906 |
| B2 HeuristicPublic | 399 | 512 | 0.779297 |
| B3 HeuristicPublicAggro | 365 | 512 | 0.712891 |
| B4 HeuristicPublicControl | 382 | 512 | 0.746094 |

This is strong enough for the thesis main-model claim: it beats every required
fixed-deck baseline at 256 paired seeds.

## Champion And Hard-Negative Result

Imported champion/hard-negative confirm256 evidence:

- summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/targeted_confirm256_imported_champions/targeted_confirm256_summary.json`
- aggregate: `2317/4096 = 0.565674`
- all eight imported learned opponents are above `0.5`

This is supportive robustness evidence. It is not a claim that the selected
model is the best possible learned-opponent model; it is a claim that the
selected fixed-deck model also remains positive against the learned league
panel.

## Paper Artifacts

The selected run has the artifact surface needed for the thesis:

- final eval:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/summary.json`
- final eval matrices:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/matrices/`
- payoff matrices:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/payoff_matrices/`
- metagame:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/metagame/summary.json`
- replay verification:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/diagnostics/replay_verification.json`
- paper readiness:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/paper_readiness_summary.json`
- paper figures:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/`

Verified status:

- paper readiness: `passed: true`
- alarms: `[]`
- replay verification: `3/3` sampled episodes verified, `0` failed
- figures exist for learning curves, matchup heatmap, seat bias, and
  truncation heatmap in both PDF and PNG forms

## What We Did

Started from the locked B1 NoLeague artifact:

- run: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- selected policy id: `selected_candidate`
- source policy id: `policy_000003`
- update: `15`
- weights hash:
  `66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685`

Then built the main league path around:

- B1 seeding;
- fixed-deck B0-B4+B1 evaluation;
- champion and hard-negative opponent import;
- guarded segmented training;
- checkpoint confirmation before continuation;
- final eval, metagame, replay verification, figures, and paper readiness;
- faster-loop diagnostics to prevent wasting hours on bad candidates.

The selected model is an interpolation artifact:

- first checkpoint:
  `runs/main_champion_hardneg_long_v1_u10_20260517_seg01/training/checkpoints/checkpoint_10.pt`
- second checkpoint:
  `runs/main_champion_hardneg_rehearsal_from_u20_u5_20260517_seg01/training/checkpoints/checkpoint_5.pt`
- second weight: `0.15`
- interpolation summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/diagnostics/checkpoint_interpolation_summary.json`

Why interpolation was accepted:

- direct longer training exposed and improved learned-opponent ecology, but
  drifted fixed rows;
- rehearsal recovered some fixed behavior but softened other rows;
- interpolation gave the best confirmed fixed-deck balance while retaining
  positive champion/hard-negative scores.

## What Worked

- The locked B1 seed was usable and stable enough for main league training.
- Champion and hard-negative import worked; the selected model is positive
  against all imported learned rows at confirm256.
- Fixed-deck thesis eval worked and produced a clear B0-B4+B1 win.
- `latest` versus selected semantics stayed clean:
  chronological latest was not treated as best-confirmed.
- Segmenting and confirming candidates prevented a bad long run from silently
  becoming selected.
- The faster-loop gate was useful:
  mechanistic gates, sentinel panels, full confirm64, confirm128, and
  confirm256 were kept separate.
- The frontier audit now correctly collapses stale aliases and lower-stage
  artifacts into candidate families:
  `diagnostics/main_league_frontier_audit_20260521.json`.

## What Did Not Work

Longer or more aggressive training did not automatically make a publishable
successor. The recurring failure was row-level interference:

- learned/champion/hard-negative aggregate improved;
- fixed aggregate often stayed close or even nonnegative;
- but individual rows such as B2 or policy0004 regressed by one or two games.

Specific rejected families included:

- direct u20 continuation: B3/B4/B1 drift issues;
- rehearsal/consolidation branches: recovered one row while weakening another;
- grouped replay and trajectory-BC repair: moved learned rows but taxed fixed
  rows;
- conservative fixed-preservation branches: protected fixed rows but suppressed
  learned progress;
- interpolation variants: useful for final selected model, but later
  interpolations did not produce a strict successor;
- pair205-only repair: too narrow and not thesis-defensible as a general
  solution;
- global action-id bias: unsafe because it changes action preference beyond the
  intended local conflict;
- broad rich-bilinear/residual probes: moved local margins but failed drift,
  row, or edge anti-churn gates;
- live-pressure u1 branches: sometimes improved fixed aggregate, but learned
  movement was flat or failed strict gates;
- strongest a050/b2-old-hard-negative frontier: improved aggregate learned
  performance but failed B2/policy0004 row-level no-regression.

## Why We Stopped Replacing The Model

The best non-published successor showed real movement:

- confirm256 versus selected:
  - overall delta: `+6`
  - fixed delta: `0`
  - learned delta: `+6`

But it failed strict row gates:

- B2 HeuristicPublic: `-2`
- policy0004: `-1`

That means it was not publishable under the thesis contract. It was better on
one aggregate surface, but not better in the required row-wise sense.

The 2026-05-21 frontier audit is the final ledger for this lock:

- candidate families: `80`
- candidates by next stage: `{"stop": 80}`
- publishable successor exists: `false`
- selected remains locked: `true`

## Current Interpretation

The selected model is good enough and thesis-defensible:

1. It is paper-ready.
2. It clearly beats B0, B1, B2, B3, and B4 on the fixed-deck surface.
3. It is positive against imported champion and hard-negative rows.
4. Successor attempts exposed a real structural plateau, not just a lack of
   training time.
5. The unresolved blocker is row-level objective interference, especially
   around B2 and policy0004.

This is not "we failed to learn." It is: we learned a strong general fixed-deck
main model, then found that the next aggregate gains are not publishable without
a more conditional or localized mechanism.

## If Future Me Wants To Continue

Do not continue by launching another blind long run.

The next credible research branch should:

1. Start from the strongest verified non-published aggregate frontier only as a
   diagnostic source, not as selected.
2. Target the broader conflict family:
   - B2 pair16
   - B2 pair205
   - B2 pair229
   - policy0004 pair205
   - policy0004 pair222
   - learned counterparts tied to pair16/pair205
3. Avoid global "make action 104 beat 124" objectives.
4. Use only model-legal inputs:
   public/current state, recurrent/history representation, legal candidate
   features, action-family metadata, and legitimate opponent context.
5. Forbid diagnostic-only inputs:
   pair index, episode seed, swap index, replay id, and source labels.
6. Pass offline gates before sentinel:
   context coverage, mechanistic margins, row guard, edge guard, drift guard,
   no protected non-near-tie target-top losses.
7. Escalate only in this order:
   offline gate -> sentinel -> full confirm64 -> confirm128 -> confirm256.

Until a future candidate passes that ladder, this model stays locked.

