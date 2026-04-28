# Thesis Model Recipe

This is the frozen training and evaluation surface for the thesis-model run.

## Canonical presets

- `configs/main_impala_league_server.yaml`
  - frozen tiny248 thesis training recipe
  - current recommended server run surface
- `configs/baselines/noleague_impala.yaml`
  - matching `baseline_noleague` prerequisite for the frozen thesis model
- `configs/main_eval.yaml`
  - frozen thesis-model eval surface, including the richer final anchor set
- `configs/ablations/multideck.yaml`
  - separate multideck branch, not the frozen single-deck thesis model

## Baseline prerequisite

`thesis-model-auto-gpu` and the named strong ablations all require a completed dedicated `baseline_noleague` run so the canonical `B1 NoLeague baseline` anchor can be imported into the training league.

Bootstrap that baseline once with the matching model/environment surface:

```bash
uv run python python/scripts/train.py \
  --stack-config configs/baselines/noleague_impala.yaml \
  --run-label b1_anchor_thesis_model_seed1 \
  --num-envs 2048 \
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
  --preset thesis-model-auto-gpu \
  --run-label thesis_model_seed1 \
  --b1-baseline-run-dir runs/b1_anchor_thesis_model_seed1 \
  --num-envs 2048 \
  --unroll-length 64 \
  --runtime-mode train_async_fast \
  --max-updates 400 \
  --skip-compare
```

That wrapper call trains with `thesis-model-auto-gpu` and, by default, evaluates with `thesis-model-eval-auto-gpu`.

`thesis-model-auto-gpu` uses `learner_device: cuda:auto`, `actor_device: cuda:auto`, and `collection_backend: process`. On a 2+ GPU node that means the learner takes one GPU and the process collectors round-robin actor inference across the remaining visible GPUs.

Run the richer final thesis evaluation on an existing run:

```bash
uv run python python/scripts/eval.py \
  --stack-config configs/main_eval.yaml \
  --run-dir runs/<run_dir>
```

Parallel eval on a multi-GPU box:

```bash
uv run python python/scripts/eval.py \
  --stack-config configs/main_eval.yaml \
  --run-dir runs/<run_dir> \
  --parallel-workers 2 \
  --parallel-worker-device cuda:0 \
  --parallel-worker-device cuda:1
```

Play against the finalized focal model from a completed run:

```bash
uv run python python/scripts/play_vs_model.py \
  --run-dir runs/<run_dir>
```

Try the multideck generalization variant:

```bash
uv run python python/scripts/thesis_run.py \
  --preset thesis-model-multideck \
  --run-label thesis_multideck_seed1 \
  --b1-baseline-run-dir runs/b1_anchor_seed1 \
  --device cuda \
  --num-envs 4096 \
  --unroll-length 64 \
  --runtime-mode train_async_fast \
  --max-updates 200
```

`thesis-model-multideck` reuses `structured_acceptance_thesis_model_multideck_auto_gpu.yaml` for eval too, so the wrapper keeps the multideck surface aligned unless you override it explicitly.

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
4. `ablate-reward-shaping`
   - turns on the local-style terminal-plus-shaping reward and truncation penalty on top of the frozen thesis model.
   - answers whether the stronger thesis result depends on staying on the conservative terminal-only reward surface.
5. `thesis-model-multideck`
   - introduces actor-level deck diversity.
   - answers whether the thesis-model recipe is over-specializing to a single deck and whether diversity helps robustness enough to justify any throughput loss.

Suggested wrapper invocations:

```bash
uv run python python/scripts/thesis_run.py --preset ablate-no-tactical-bias --run-label ablate_no_tactical_bias_seed1 --b1-baseline-run-dir runs/<matching_b1_run> --device cuda --num-envs 2048 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset ablate-teacher-fade --run-label ablate_teacher_fade_seed1 --b1-baseline-run-dir runs/<matching_b1_run> --num-envs 2048 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset ablate-no-b1-cutoff --run-label ablate_no_b1_cutoff_seed1 --b1-baseline-run-dir runs/<matching_b1_run> --device cuda --num-envs 2048 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset ablate-reward-shaping --run-label ablate_reward_shaping_seed1 --b1-baseline-run-dir runs/<matching_b1_run> --device cuda --num-envs 2048 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset thesis-model-multideck --run-label ablate_multideck_seed1 --b1-baseline-run-dir runs/<matching_multideck_b1_run> --device cuda --num-envs 2048 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
```

## What Still Matters

- Run at least `2-3` seeded confirmations on the university server.
- Use `structured_acceptance_thesis_model_eval_auto_gpu.yaml` for the final reported evaluation bundle of the frozen thesis model.
- Keep one multideck/generalization run in the final study, even if the main reported frontier stays single-deck, because it makes the thesis story much stronger on robustness.

