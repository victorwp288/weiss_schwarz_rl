# Standard Recipe

This is the current ship-ready training and evaluation surface for the thesis run.

## Canonical presets

- `configs/presets/structured_acceptance_standard.yaml`
  - current default training recipe
  - best local learning recipe so far
- `configs/presets/structured_acceptance_standard_auto_gpu.yaml`
  - same recipe, but automatically shards process-collector actor inference across visible CUDA devices while reserving one learner GPU
- `configs/presets/structured_acceptance_standard_thesis_eval.yaml`
  - same training recipe, plus the richer thesis anchor set for final evaluation
- `configs/presets/structured_acceptance_standard_multideck.yaml`
  - same recipe, but with actor-cycled bundled Quintuplets deck diversity

## Baseline prerequisite

`standard`, `standard-auto-gpu`, `standard-multideck`, and the named ablations all require a completed dedicated `baseline_noleague` run so the canonical `B1 NoLeague baseline` anchor can be imported into the training league.

Bootstrap that baseline once with the matching model/environment surface:

```bash
uv run python python/scripts/train.py \
  --stack-config configs/presets/baselines/structured_acceptance_tiny32_fast_noleague.yaml \
  --run-label b1_anchor_seed1 \
  --device cuda \
  --num-envs 4096 \
  --unroll-length 64 \
  --runtime-mode train_async_fast \
  --max-updates 200
```

## Fastest commands

List the named presets exposed by the wrapper:

```bash
uv run python python/scripts/thesis_run.py --list-presets
```

Launch the canonical training recipe:

```bash
uv run python python/scripts/thesis_run.py \
  --preset standard \
  --run-label thesis_seed1 \
  --b1-baseline-run-dir runs/b1_anchor_seed1 \
  --device cuda \
  --num-envs 4096 \
  --unroll-length 64 \
  --runtime-mode train_async_fast \
  --max-updates 200
```

That wrapper call trains with `standard` and, by default, evaluates with `standard-thesis-eval`.

Launch the recommended Linux server variant on a multi-GPU node:

```bash
uv run python python/scripts/thesis_run.py \
  --preset standard-auto-gpu \
  --run-label thesis_server_seed1 \
  --b1-baseline-run-dir runs/b1_anchor_seed1 \
  --num-envs 4096 \
  --unroll-length 64 \
  --runtime-mode train_async_fast \
  --max-updates 200
```

`standard-auto-gpu` uses `learner_device: cuda:auto`, `actor_device: cuda:auto`, and `collection_backend: process`. On a 2+ GPU node that means the learner takes one GPU and the process collectors round-robin actor inference across the remaining visible GPUs.

Run the richer final thesis evaluation on an existing run:

```bash
uv run python python/scripts/eval.py \
  --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml \
  --run-dir runs/<run_dir>
```

Play against the finalized focal model from a completed run:

```bash
uv run python python/scripts/play_vs_model.py \
  --run-dir runs/<run_dir>
```

Try the multideck generalization variant:

```bash
uv run python python/scripts/thesis_run.py \
  --preset standard-multideck \
  --run-label thesis_multideck_seed1 \
  --b1-baseline-run-dir runs/b1_anchor_seed1 \
  --device cuda \
  --num-envs 4096 \
  --unroll-length 64 \
  --runtime-mode train_async_fast \
  --max-updates 200
```

`standard-multideck` defaults its companion eval stack to `structured_acceptance_standard_multideck.yaml`, not `standard-thesis-eval`, so the wrapper keeps the multideck surface aligned unless you override it explicitly.

## Thesis ablations

These are the most defensible next ablations around the current best recipe.

1. `ablate-no-tactical-bias`
   - removes the tactical public-bias shaping while keeping the rest of the curriculum close.
   - answers whether the tactical bias is genuinely responsible for the stronger B1/B2 learning curve.
2. `ablate-no-b1-cutoff`
   - keeps persistent B1 exposure instead of annealing it away after update 10.
   - answers whether the early-B1 curriculum matters more than persistent baseline pressure.
3. `standard-multideck`
   - introduces actor-level deck diversity.
   - answers whether the standard recipe is over-specializing to a single deck and whether diversity helps robustness enough to justify any throughput loss.

Suggested wrapper invocations:

```bash
uv run python python/scripts/thesis_run.py --preset ablate-no-tactical-bias --run-label ablate_no_tactical_bias_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1 --device cuda --num-envs 4096 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset ablate-no-b1-cutoff --run-label ablate_no_b1_cutoff_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1 --device cuda --num-envs 4096 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset standard-multideck --run-label ablate_multideck_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1 --device cuda --num-envs 4096 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
```

## What Still Matters

- Run at least `2-3` seeded confirmations on the university server.
- Use `structured_acceptance_standard_thesis_eval.yaml` for the final reported evaluation bundle.
- Keep one multideck/generalization run in the final study, even if the main reported frontier stays single-deck, because it makes the thesis story much stronger on robustness.
