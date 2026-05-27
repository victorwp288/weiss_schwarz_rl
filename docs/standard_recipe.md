# Standard Recipe

This is the current ship-ready training and evaluation surface for the thesis run.

## Canonical Thesis Configs

- `configs/thesis/b1_noleague.yaml`
  - B1 NoLeague baseline on the fixed main deck.
- `configs/thesis/main_league.yaml`
  - main IMPALA/V-trace league thesis lane.
- `configs/thesis/main_league_auto_gpu.yaml`
  - server-oriented main lane with process collectors.
- `configs/thesis/final_eval.yaml`
  - final evaluation companion with B0-B4 anchors.
- `configs/thesis/multideck_exploratory.yaml`
  - exploratory deck-diversity/generalization variant.

The `configs/presets/structured_acceptance_standard*` files are the
compatibility preset layer underneath these names.

## Deck scoping

The primary Phase 1 lane is same-deck:

- focal model policies: `preset:main_deck_5hy_yotsuba_v1`
- `B0 RandomLegal`: `preset:main_deck_5hy_yotsuba_v1`
- `B1 NoLeague baseline`: `preset:main_deck_5hy_yotsuba_v1`
- `B2 HeuristicPublic`: `preset:main_deck_5hy_yotsuba_v1`

The themed public heuristics are separate robustness anchors:

- `B3 HeuristicPublicAggro`: `preset:aggro_deck_5hy_nino_v1`
- `B4 HeuristicPublicControl`: `preset:control_deck_jj_s66_v1`

Final-eval schedules write `seat0_deck` and `seat1_deck` into episode records and matchup summaries, so themed rows remain visible in artifacts instead of being folded into an anonymous aggregate.

## Baseline prerequisite

The main league command requires a completed dedicated B1 run so the canonical
`B1 NoLeague baseline` anchor can be imported into the training league.

Bootstrap that baseline once with the matching model/environment surface:

```bash
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_anchor_seed1 --profile thesis-local
```

## Fastest commands

Launch the canonical training recipe:

```bash
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label thesis_seed1 --b1-run runs/b1_anchor_seed1 --profile thesis-local
```

That command trains with `configs/thesis/main_league.yaml`.

Launch the recommended Linux server variant on a multi-GPU node:

```bash
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label thesis_server_seed1 --b1-run runs/b1_anchor_seed1 --profile thesis-server
```

`standard-auto-gpu` uses `learner_device: cuda:auto`, `actor_device: cuda:auto`, and `collection_backend: process`. On a 2+ GPU node that means the learner takes one GPU and the process collectors round-robin actor inference across the remaining visible GPUs.

Run the richer final thesis evaluation on an existing run:

```bash
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<run_dir> --b1-run runs/b1_anchor_seed1
```

Play against the finalized focal model from a completed run:

```bash
uv run python python/scripts/play_vs_model.py \
  --run-dir runs/<run_dir>
```

Try the multideck generalization variant:

```bash
uv run python python/scripts/train.py --stack-config configs/thesis/multideck_exploratory.yaml --run-label thesis_multideck_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1
```

Keep multideck artifacts labeled as exploratory/generalization results.

## Thesis ablations

These are the most defensible next ablations around the current best recipe.

1. `ablate-no-tactical-bias`
   - removes the tactical public-bias shaping while keeping the rest of the curriculum close.
   - answers whether the tactical bias is genuinely responsible for the stronger B1/B2 learning curve.
2. `ablate-teacher-fade`
   - keeps the current guided recipe early, then fades the public teacher loss, heuristic actor fraction, and tactical learner-side public bias later in training.
   - answers whether persistent heuristic guidance is helping asymptotically or holding the policy too close to the teacher.
3. `ablate-no-b1-cutoff`
   - keeps persistent B1 exposure instead of annealing it away after update 10.
   - answers whether the early-B1 curriculum matters more than persistent baseline pressure.
4. `standard-multideck`
   - introduces actor-level deck diversity.
   - answers whether the standard recipe is over-specializing to a single deck and whether diversity helps robustness enough to justify any throughput loss.

Suggested direct invocations:

```bash
uv run python python/scripts/train.py --stack-config configs/thesis/ablations/norecurrence_impala.yaml --run-label ablate_norecurrence_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1
uv run python python/scripts/train.py --stack-config configs/thesis/ablations/ppo_lite.yaml --run-label ablate_ppo_lite_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1
uv run python python/scripts/train.py --stack-config configs/thesis/multideck_exploratory.yaml --run-label ablate_multideck_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1
```

## What Still Matters

- Run at least `2-3` seeded confirmations on the university server.
- Use `configs/thesis/final_eval.yaml` through `python -m weiss_rl.cli eval-final` for the final reported evaluation bundle.
- Keep one multideck/generalization run in the final study, even if the main reported frontier stays single-deck, because it makes the thesis story much stronger on robustness.
