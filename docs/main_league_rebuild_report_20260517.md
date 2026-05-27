# Main League Rebuild Report

Date: 2026-05-17

Repo: `C:\Users\Bruger\Desktop\this one\weiss_schwarz_rl`

Scope: B1-seeded main model diagnosis, structural league repair, selected-main
checkpoint publication, and paper-ready fixed-deck B0-B4 plus B1 evaluation.

The chronological work log is `docs/rebuild_log.md`.

## Current Status

The current paper-ready fixed-deck main artifact is:

- run: `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01`
- selected alias: `main_league_selected`
- selected source policy id: `policy_000001`
- selected update: `5`
- selected weights hash:
  `efd9d085b3482ff0470f2dcfe2e0a3a0b26e8264cf297e970beb27cdd32bdc34`
- selection summary:
  `diagnostics/main_b1only_p2_trust_region_u5_v3_20260517_seg01_candidate_published_confirm256.json`
- confirm256 summary:
  `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/eval/targeted_confirm256_b0_b4/targeted_confirm256_summary.json`

The locked B1 seed used for this phase was:

- run: `runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01`
- selected policy id: `selected_candidate`
- selected source policy id: `policy_000003`
- selected update: `15`
- weights hash:
  `66767c1e70c70d1706c058bfd38a7b20cb902c9740d96b6fb1ba664a2b65a685`
- B1 report: `docs/b1_learning_rebuild_report_20260517.md`

Snapshot semantics:

- `latest` remains chronological latest.
- `selected_candidate` remains the locked best-confirmed B1 alias.
- `main_league_selected` is now republished from 256 paired-seed evidence, not
  confirm64.
- No checkpoint was promoted because scalar loss improved.

## Fixed Deck Surface

The final evidence uses the thesis fixed-deck policy:

- focal/main/B0/B1/B2: `preset:main_deck_5hy_yotsuba_v1`
- B3 aggro: `preset:aggro_deck_5hy_nino_v1`
- B4 control: `preset:control_deck_jj_s66_v1`

No multideck results are mixed into the selected-main claim.

## Final Evidence

The selected source checkpoint was confirmed separately against every required
fixed-deck opponent at 256 paired seeds:

| Opponent | Wins | Games | Win rate | 95% CI | P(p > 0.5) |
|---|---:|---:|---:|---|---:|
| B0 RandomLegal | 512 | 512 | 1.000000 | [1.000000, 1.000000] | 1.0 |
| B1 NoLeague baseline | 318 | 512 | 0.621094 | [0.585161, 0.657309] | 1.0 |
| B2 HeuristicPublic | 392 | 512 | 0.765625 | [0.729840, 0.798540] | 1.0 |
| B3 HeuristicPublicAggro | 363 | 512 | 0.708984 | [0.667191, 0.746810] | 1.0 |
| B4 HeuristicPublicControl | 386 | 512 | 0.753906 | [0.716671, 0.787953] | 1.0 |

Canonical final eval was then run with the full B0-B4 plus B1 policy set and
`main_league_selected` as the focal policy:

```powershell
$env:PYTHONHASHSEED='0'
uv run --extra dev --extra sim python python/scripts/eval.py --stack-config configs/thesis/final_eval.yaml --run-dir runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01 --snapshot-registry-json runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/training/snapshots/registry.json --b1-baseline-run-dir runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01 --policy-id main_league_selected --policy-id "B0 RandomLegal" --policy-id "B1 NoLeague baseline" --policy-id "B2 HeuristicPublic" --policy-id "B3 HeuristicPublicAggro" --policy-id "B4 HeuristicPublicControl" --paired-seed-limit 256 --stage1-paired-seeds 64 --max-paired-seeds 256 --bootstrap-samples 2000
```

Canonical final-eval selected-main row:

| Opponent | Wins | Games | Win rate |
|---|---:|---:|---:|
| B0 RandomLegal | 128 | 128 | 1.000000 |
| B1 NoLeague baseline | 83 | 128 | 0.648438 |
| B2 HeuristicPublic | 99 | 128 | 0.773438 |
| B3 HeuristicPublicAggro | 91 | 128 | 0.710938 |
| B4 HeuristicPublicControl | 91 | 128 | 0.710938 |

The canonical final eval used adaptive stopping; selected-main rows stopped
decisively at 64 paired seeds. The thesis selection claim above is backed by
the separate confirm256 targeted rows.

Compared with the previously rejected selected main artifact
`runs/main_b1only_bestresponse_from_u50_to_u100_20260517`, the confirm256 rows
improved every required non-random opponent:

| Opponent | Rejected selected | Current selected | Delta |
|---|---:|---:|---:|
| B1 NoLeague baseline | 0.580078 | 0.621094 | +0.041016 |
| B2 HeuristicPublic | 0.714844 | 0.765625 | +0.050781 |
| B3 HeuristicPublicAggro | 0.677734 | 0.708984 | +0.031250 |
| B4 HeuristicPublicControl | 0.681641 | 0.753906 | +0.072266 |

Against the previous p2 internal frontier, the current model is a narrow but
confirmed balance improvement: `1971/2560` wins versus `1969/2560`, with B1 and
B4 improving and B2/B3 slightly lower.

## Artifact Manifest

Selected-main artifacts:

- final eval summary:
  `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/eval/final_eval/summary.json`
- final eval matrices:
  `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/eval/final_eval/matrices/`
- policy set:
  `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/eval/final_eval/policy_set.json`
- metagame summary:
  `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/eval/metagame/summary.json`
- replay verification:
  `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/eval/diagnostics/replay_verification.json`
- paper readiness:
  `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/paper_readiness_summary.json`
- paper figures:
  `runs/main_b1only_p2_trust_region_u5_v3_20260517_seg01/figures/paper/`

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

## Structural Repairs

### B1 Seed Loading And Import Contract

The main lane resolves the locked B1 `selected_candidate` alias instead of
assuming chronological latest. Imported B1 payloads preserve
`imported_from_weights_sha256`, and tensor equality against the locked B1
weights was verified earlier in the rebuild.

### Confirmation Before Publication

The guarded bootstrap controller now has `publish_min_confirm_paired_seeds`.
Passing confirm64 can stop as `accepted_unpublished`, but it cannot publish a
selected alias below the configured evidence floor. The current selected alias
was republished from confirm256.

### Checkpoint Publication

`publish_checkpoint_snapshot.py` can materialize unregistered numbered
checkpoints as snapshot candidates without moving `latest`. Its path resolution
now accepts both run-relative and repo-relative checkpoint paths.

### Opponent Ecology Probes

The accepted model comes from a B1-only p2 trust-region continuation:

`configs/thesis/ablations/main_b1only_p2_trust_region_probe.yaml`

Additional opponent-ecology probes were rejected and left as reproducible
ablation configs:

- `configs/thesis/ablations/main_b1only_p2_trust_region_no_warmup_probe.yaml`
- `configs/thesis/ablations/main_b1only_p2_trust_region_argmax_opp_probe.yaml`
- `configs/thesis/ablations/main_b1only_p2_free_argmax_opp_probe.yaml`

## Rejected Alternatives

- Old best-response dev-best `checkpoint_25`:
  B1 `0.539062`, B2 `0.726562`, B3 `0.742188`, B4 `0.687500`.
- Continuing the accepted trust-region branch another 5 updates:
  B1 `0.632812`, B2 `0.773438`, B3 `0.703125`, B4 `0.726562`.
- No-warmup B1-only trust region:
  B1 `0.617188`, B2 `0.781250`, B3 `0.710938`, B4 `0.734375`.
- Argmax-opponent B1-only trust region:
  B1 `0.640625`, B2 `0.796875`, B3 `0.695312`, B4 `0.718750`.
- Free deterministic-B1 best response with no p2 KL anchor:
  B1 `0.617188`, B2 `0.781250`, B3 `0.718750`, B4 `0.710938`.
- Per-update checkpointing did not find a better hidden update:
  B1 screens for updates 1-5 were `81/128`, `80/128`, `79/128`,
  `81/128`, and `80/128`.

## Validation

- Focused regression slice: `21 passed`.
- Checkpoint-publish tests: `3 passed`.
- New config-loader tests for trust-region ablations: passed.
- Ruff on touched Python files: passed.
- Canonical final eval: completed.
- Paper readiness: passed.

## Remaining Risks

The current selected main model clearly beats B0-B4 and B1 on the fixed-deck
thesis surface and is materially stronger than the rejected selected artifact.
The p2 internal frontier comparison is narrow: B1 and B4 improve, while B2/B3
are slightly lower. Report the claim as a fixed-deck thesis model result, not as
a strict domination of every intermediate research checkpoint.

Future margin work should use deeper same-family/card-choice replay diagnosis
around B1 losses rather than blindly extending B1-only training.
