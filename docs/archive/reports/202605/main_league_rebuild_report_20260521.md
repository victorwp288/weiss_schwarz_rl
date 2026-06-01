# Main League Rebuild Report

Date: 2026-05-21

Repo: `C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl`

Scope: B1-seeded main league training, champion/hard-negative diagnostics,
localized repair attempts, faster-loop gating, and final thesis selection
status for the fixed-deck main model.

The chronological work log is `docs/rebuild_log.md`. The machine-readable
frontier audit is `diagnostics/main_league_frontier_audit_20260521.json`.

## Current Status

The current thesis-selected main model remains locked:

- run: `runs/main_champion_hardneg_interp_u10_repair_a015_20260517`
- selected alias: `main_league_selected`
- source policy id: `main_interp_repair_a015`
- selected update: `5`
- weights hash:
  `1a13b49b73ed71af0914c97fede5b30703eb576a5e85c4c636310c2d76897b26`
- selection summary:
  `diagnostics/main_champion_hardneg_interp_u10_repair_a015_selected.json`
- fixed B0-B4+B1 confirm256:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/targeted_confirm256_b0_b4/targeted_confirm256_summary.json`
- imported champion/hard-negative confirm256:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/targeted_confirm256_imported_champions/targeted_confirm256_summary.json`

No successor has been published. Confirm64 and confirm128 artifacts remain
triage evidence only. The 2026-05-21 frontier audit found:

- candidate families audited: `80`
- candidates by next stage: `{"stop": 80}`
- publishable successor exists: `false`
- selected remains locked: `true`

## Fixed Deck Surface

The primary thesis comparison uses the fixed deck policy:

- focal/main/B0/B1/B2: `preset:main_deck_5hy_yotsuba_v1`
- B3 aggro: `preset:aggro_deck_5hy_nino_v1`
- B4 control: `preset:control_deck_jj_s66_v1`

Multideck results are not mixed into the selected-main claim.

## Fixed Baseline Evidence

The locked selected model was evaluated against B0-B4 plus B1 at 256 paired
seeds, with two games per seed:

| Opponent | Wins | Games | Win rate | 95% CI | P(p > 0.5) |
|---|---:|---:|---:|---|---:|
| B0 RandomLegal | 512 | 512 | 1.000000 | [1.000000, 1.000000] | 1.000 |
| B1 NoLeague baseline | 322 | 512 | 0.628906 | [0.594455, 0.664650] | 1.000 |
| B2 HeuristicPublic | 399 | 512 | 0.779297 | [0.746255, 0.813301] | 1.000 |
| B3 HeuristicPublicAggro | 365 | 512 | 0.712891 | [0.670363, 0.753840] | 1.000 |
| B4 HeuristicPublicControl | 382 | 512 | 0.746094 | [0.708529, 0.785453] | 1.000 |

Aggregate fixed surface:

- wins/games: `1980/2560`
- win rate: `0.773438`

This is the headline fixed-deck thesis result. It clearly beats every fixed
baseline row, including the locked B1 NoLeague model and the B2/B3/B4 public
heuristics.

## Champion And Hard-Negative Evidence

The selected model was also evaluated at 256 paired seeds against imported
league champions, hard negatives, checkpoint opponents, and best-response
opponents:

| Imported opponent | Wins | Games | Win rate | 95% CI | P(p > 0.5) |
|---|---:|---:|---:|---|---:|
| `seed_c3aac2f9dc_policy_000001` | 328 | 512 | 0.640625 | [0.603177, 0.680050] | 1.000 |
| `seed_c3aac2f9dc_policy_000002` | 299 | 512 | 0.583984 | [0.548641, 0.619471] | 1.000 |
| `seed_c3aac2f9dc_checkpoint_000025` | 289 | 512 | 0.564453 | [0.530064, 0.600208] | 1.000 |
| `seed_c3aac2f9dc_main_bestresponse_u25_devbest` | 289 | 512 | 0.564453 | [0.529609, 0.599773] | 1.000 |
| `seed_c3aac2f9dc_main_league_selected` | 274 | 512 | 0.535156 | [0.501103, 0.570314] | 0.9785 |
| `seed_c3aac2f9dc_policy_000003` | 274 | 512 | 0.535156 | [0.498960, 0.567846] | 0.9710 |
| `seed_c3aac2f9dc_policy_000004` | 291 | 512 | 0.568359 | [0.530726, 0.604962] | 1.000 |
| `seed_c3aac2f9dc_policy_000005` | 273 | 512 | 0.533203 | [0.497311, 0.569305] | 0.9685 |

Aggregate learned-opponent surface:

- wins/games: `2317/4096`
- win rate: `0.565674`

The defensible claim is that the selected model is positive against every
imported learned opponent at confirm256. The stronger claim that a later
successor improves this learned surface without any fixed-row cost is not
supported by the current evidence.

## Successor Frontier

The strongest non-published frontier showed real aggregate movement but failed
row-level gates:

- candidate family: `a050_b2oldhn_u2_policy_000002`
- confirm128 result versus selected:
  - all delta: `+6`
  - fixed delta: `+2`
  - learned delta: `+4`
  - stop reason: B2 row regression
  - B2 delta: `-1` game
- related confirm256 successor evidence:
  - all delta: `+6`
  - fixed delta: `0`
  - learned delta: `+6`
  - stop reason: B2 and policy0004 row regressions
  - B2 delta: `-2` games
  - policy0004 delta: `-1` game

This is the core structural result: mixture/objective changes could improve
aggregate learned-opponent performance, but the strict thesis surface repeatedly
paid for it through small B2 and policy0004 regressions.

## Faster-Loop Gates

The 2026-05-21 faster loop now treats local diagnostics as mandatory before
game evaluation:

1. mechanistic gate and context coverage;
2. row/edge/drift anti-churn checks;
3. sentinel panel;
4. full confirm64;
5. confirm128;
6. confirm256;
7. publication only after confirm256.

The audit tooling prevents stale lower-stage evidence from overriding later
stops. In particular:

- sentinel pass does not survive if full confirm64 later stops the family;
- confirm64/confirm128 evidence cannot publish a model;
- `run_confirm256` means "run confirm256", not "publish";
- compare-only evidence with no learned improvement is stopped;
- short diagnostic aliases are folded into the same candidate family.

The resulting audit reports all current candidate families as stopped:

- `diagnostics/main_league_frontier_audit_20260521.json`
- `docs/main_league_frontier_audit_20260521.md`

## Artifact Readiness

The selected run has the expected paper artifacts:

- final eval summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/summary.json`
- final eval matrices:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/matrices/`
- payoff matrices:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/final_eval/payoff_matrices/`
- metagame summary:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/metagame/summary.json`
- replay verification:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/eval/diagnostics/replay_verification.json`
- paper readiness:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/paper_readiness_summary.json`
- paper figures:
  `runs/main_champion_hardneg_interp_u10_repair_a015_20260517/figures/paper/`

Paper readiness status:

- `passed: true`
- `alarms: []`
- replay verification: `3/3` sampled episodes verified, `0` failed

Rendered paper figures:

- `fig_learning_curves.{pdf,png}`
- `fig_matchup_heatmap.{pdf,png}`
- `fig_seat_bias.{pdf,png}`
- `fig_truncation_heatmap.{pdf,png}`

## Reproducibility Commands

Fixed B0-B4+B1 confirm256:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python python/scripts/targeted_confirm_eval.py --stack-config configs/thesis/ablations/main_league_champion_hardneg_rehearsal_probe.yaml --run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517 --snapshot-registry-json runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry.json --b1-baseline-run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517 --focal-policy-id main_interp_repair_a015 --paired-seeds 256 --workers 1 --bootstrap-samples 2000 --output-subdir b1_candidate_confirm256_main_interp_repair_a015 --opponent "B0 RandomLegal" --opponent "B1 NoLeague baseline" --opponent "B2 HeuristicPublic" --opponent "B3 HeuristicPublicAggro" --opponent "B4 HeuristicPublicControl"
```

Canonical final eval:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python python/scripts/eval.py --stack-config configs/thesis/final_eval.yaml --run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517 --snapshot-registry-json runs/main_champion_hardneg_interp_u10_repair_a015_20260517/training/snapshots/registry.json --b1-baseline-run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 --policy-id main_league_selected --policy-id "B0 RandomLegal" --policy-id "B1 NoLeague baseline" --policy-id "B2 HeuristicPublic" --policy-id "B3 HeuristicPublicAggro" --policy-id "B4 HeuristicPublicControl" --paired-seed-limit 256 --stage1-paired-seeds 64 --max-paired-seeds 256 --bootstrap-samples 2000
```

Frontier audit:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev python python/scripts/main_league_frontier_audit.py --diagnostics-dir diagnostics --date-token 20260521 --output-json diagnostics/main_league_frontier_audit_20260521.json --output-md docs/main_league_frontier_audit_20260521.md
```

## Thesis Interpretation

The current thesis-defensible position is:

1. `main_interp_repair_a015` is a paper-ready fixed-deck main model.
2. It beats B0, locked B1, B2, B3, and B4 decisively on 256 paired seeds.
3. It is positive against all imported champion and hard-negative opponents on
   the confirm256 learned panel.
4. Later successor attempts demonstrate a real learned-opponent improvement
   frontier, but none satisfy the strict row-level no-regression contract.
5. Therefore `main_league_selected` should remain the thesis main model unless
   a new candidate passes the full faster-loop ladder through confirm256.

## Remaining Risks

- The result does not prove no future architecture can improve the selected
  model. It proves that the current audited frontier has no publishable
  successor.
- The most informative failure mode remains B2/policy0004 row-level
  incompatibility. Future work should use localized, model-input-eligible
  mechanisms only after passing offline row/edge/drift gates.
- Do not publish any successor from confirm64 or confirm128 evidence.
