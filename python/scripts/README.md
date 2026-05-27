# Entry-point scripts

Run these from the repo root.

Preferred workflow after setup:

```bash
uv sync --extra dev
uv run python ...
```

If you are doing an ad-hoc local invocation without installing the package, use `PYTHONPATH=python python ...` explicitly. For normal development, prefer `uv run` or an editable install.

## Canonical recipe shortcuts

The current ship-ready stack surface is:

- `configs/presets/structured_acceptance_standard.yaml`
- `configs/presets/structured_acceptance_standard_auto_gpu.yaml`
- `configs/presets/structured_acceptance_standard_thesis_eval.yaml`
- `configs/presets/structured_acceptance_standard_multideck.yaml`

The wrapper exposes them as named presets, so the shortest standard commands are:

```bash
uv run python python/scripts/thesis_run.py --list-presets
uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_tiny32_fast_noleague.yaml --run-label b1_anchor_seed1 --device cuda --num-envs 4096 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset standard --run-label thesis_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1 --device cuda --num-envs 4096 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset standard-auto-gpu --run-label thesis_server_seed1 --b1-baseline-run-dir runs/b1_anchor_seed1 --num-envs 4096 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml --run-dir runs/<run_dir>
uv run python python/scripts/play_vs_model.py --run-dir runs/<run_dir>
```

`thesis_run.py --preset standard` now trains with `structured_acceptance_standard.yaml`, requires `--run-label`, imports the canonical `B1 NoLeague baseline` from `--b1-baseline-run-dir`, and by default evaluates with `structured_acceptance_standard_thesis_eval.yaml`.

For the current recommended ablations and what they answer, see `docs/standard_recipe.md`.

## Release verification

Use the cross-platform verification entrypoint for the release-facing local check:

```bash
uv run python python/scripts/verify_repo.py
```

If GNU Make is installed, `make verify` delegates to the same command chain. On Unix-like shells, `bash scripts/run_local_ci_parity.sh` remains the matching wrapper.

## Current script status

### `train.py`

Canonical single-node training entrypoint with a separate scaffold-only `stack_smoke` path.

```bash
make train-min
# or, when you want a low-level simulator-backed smoke on the canonical stack
make train-inline-smoke
# or directly
PYTHONPATH=../weiss-schwarz-simulator/python uv run python python/scripts/train.py --stack-config configs/presets/structured_acceptance_standard.yaml --run-label m3_08_smoke --device cpu
```

What it does today:

- loads the stack config
- probes `weiss_sim.export_spec_bundle()` and verifies the runtime contract, unless you use `--public-demo`
- computes run IDs and writes `manifest.json`, `spec_bundle.json`, `config_canonical.json`, and scaffold directories under `runs/`
- for locked simulator-backed stacks, it runs the canonical single-node queue runtime and writes training artifacts under `runs/<run>/training/`
- with `--public-demo`, stages a built-in public-safe toy catalog and deterministic toy policy bundle under `runs/<run>/public_demo/`
- always writes resumable checkpoint artifacts under `training/checkpoints/`, including `latest.pt`, `best.pt`, `observed_best.pt`, and `checkpoint_tracker.json`
- writes TensorBoard event files under `runs/<run>/tensorboard/` with run metadata, learner/runtime scalars, checkpoint alias updates, and periodic dev-eval summaries

What it does **not** do today:

- no multi-actor rollout workers or learner queues
- no long-running production training loop
- no claim that the inline smoke path equals the full master-plan actor/learner system

`--run-label` controls the human-friendly run directory name only. The immutable computed run identity still comes from the runtime spec/config/git/nonce inputs and is reported separately in the startup banner and manifest. `--run-id` remains accepted as a deprecated compatibility alias for the label override.

`manifest.json` only records a resolved deterministic `policy_set_selection` when you also pass the matching `--snapshot-registry-json` and `--dev-eval-summaries-json` inputs. Otherwise the manifest records an unresolved `policy_set_selection_details` block instead of pretending the final set is already known.

If `weiss_sim` is not installed in the active interpreter, the script still tries to collect provenance through a working local simulator interpreter/check-out when available. Configure `WEISS_SIM_PYTHONPATH` and optionally `WEISS_SIM_PYTHON` if your simulator lives elsewhere.

Important: the simulator-backed training path requires the active interpreter itself to import a simulator runtime with stepping APIs. `make train-inline-smoke` handles the usual sibling-checkout case by prepending `../weiss-schwarz-simulator/python` to `PYTHONPATH`. The only supported manifest-only path is the explicit scaffold stack (`configs/stack_smoke.yaml`).

Resume support:

```bash
uv run python python/scripts/train.py \
  --stack-config configs/presets/structured_acceptance_standard.yaml \
  --resume-run-dir runs/my_locked_run \
  --resume-from latest \
  --max-updates 200
```

That continues the existing run in-place. You can also pass a direct checkpoint path to `--resume-from` to start a fresh resumed run under a new `--run-label`.

TensorBoard live view:

```bash
tensorboard --logdir runs/<run>/tensorboard
```

The same entrypoint also supports the shipped baseline stacks:
- `configs/presets/baselines/noleague_impala.yaml`
- `configs/presets/baselines/norecurrence_impala.yaml`
- `configs/presets/baselines/ppo_lite.yaml`

Public-safe toy/demo staging:

```bash
uv run python python/scripts/train.py \
  --stack-config configs/presets/structured_acceptance_standard.yaml \
  --public-demo \
  --run-label toy_public_demo
```

That mode is explicitly synthetic. It writes demo-only catalog/policy artifacts and does not claim simulator training happened.

### `eval.py`

Evaluation reporting and contract-check entrypoint.

Contract-only smoke check:

```bash
uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml
```

Summary export from an existing episodes file:

```bash
uv run python python/scripts/eval.py \
  --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml \
  --episodes-jsonl runs/some_eval/episodes.jsonl \
  --summary-json runs/some_eval/summary.json \
  --summary-csv runs/some_eval/summary.csv \
  --diagnostics-json runs/some_eval/diagnostics.json
```

What it does today:

- validates config-hash and optional simulator contract inputs
- reports the evaluation config/contract surface
- summarizes an existing seat-swapped `episodes.jsonl` file into JSON/CSV and optional diagnostics
- with `--public-demo`, generates a deterministic demo-only `final_eval/` artifact tree from the staged toy catalog/policies
- appends final-eval, metagame, and readiness summaries to the run's TensorBoard stream under `runs/<run>/tensorboard/`

What it does **not** do today:

- outside `--public-demo`, no policy rollout execution
- outside `--public-demo`, no simulator-driven episode generation
- no hidden claim that a reported `run_label` is a computed run identity

`--run-label` is just a human label for the startup banner/log output. Unlike `train.py`, this script does not compute a run directory identity or persist the label into summary exports.

Public-safe toy/demo eval:

```bash
uv run python python/scripts/eval.py \
  --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml \
  --public-demo \
  --run-dir runs/toy_public_demo
```

That path writes demo-only `final_eval/` artifacts and labels them clearly in metadata.

### `thesis_run.py`

Thin orchestration wrapper for the canonical thesis flow: `train.py -> eval.py -> compare_runs.py`.

```bash
uv run python python/scripts/thesis_run.py \
  --preset standard \
  --run-label thesis_seed1 \
  --b1-baseline-run-dir runs/b1_anchor_seed1 \
  --device cuda:0 \
  --max-updates 200
```

Dry-run planning:

```bash
uv run python python/scripts/thesis_run.py \
  --preset standard \
  --run-label thesis_seed1 \
  --dry-run
```

The wrapper writes one summary JSON so the exact command chain is inspectable before and after execution. If you pass a custom `--stack-config`, the wrapper now reuses that same stack for eval unless you also provide an explicit `--eval-preset` or `--eval-stack-config`.

### `paper_readiness_check.py`

Paper-readiness audit over a full run directory, with compatibility support for direct `final_eval/` guardrail checks.

```bash
uv run python python/scripts/paper_readiness_check.py \
  --run-dir runs/some_run
```

Compatibility mode:

```bash
uv run python python/scripts/paper_readiness_check.py \
  --final-eval-dir runs/some_run/eval/final_eval
```

What it does today:

- with `--run-dir`, audits the broader paper-grade artifact contract under the run directory and writes one `paper_readiness_summary.json` at the run root
- validates key required artifact families, manifest completeness and consistency, and `final_eval` artifact references
- keeps the existing `final_eval` guardrails as a sub-check: aggregate truncation rate, seat-bias, and focal-policy win rate versus `B0 RandomLegal`
- exits non-zero when required artifacts are missing or any readiness guardrail fails

Default focal-policy behavior:

- if you do not pass `--focal-policy-id`, the script auto-resolves only when exactly one eligible non-baseline policy exists
- if final-eval metadata explicitly names the focal policy, that metadata is used
- otherwise the script fails clearly and requires `--focal-policy-id`

### `replay_inspector.py`

Compare two policies on the recorded states from a deterministic replay bundle.

```bash
uv run python python/scripts/replay_inspector.py \
  --bundle runs/some_run/replays/regression/replay_deadbeef.zip \
  --stack-config configs/presets/structured_acceptance_standard.yaml \
  --run-dir runs/some_run \
  --policy-a policy_000123 \
  --policy-b policy_000456 \
  --report-json runs/some_run/replays/replay_inspection.json
```

What it does today:

- reconstructs the replay environment from the bundle's persisted rerun contract
- resolves policy specs as either direct weights paths or snapshot-registry policy IDs
- evaluates both policies on each recorded replay state while advancing the env with the recorded action sequence
- prints a readable top-k diff summary and can persist a structured JSON report

What it does **not** do today:

- it does not branch the replay by executing each policy's sampled actions
- it does not search multiple replay bundles for you, pass one bundle per invocation

### `metagame.py`

Metagame sensitivity reporting over an existing `final_eval/` artifact tree.

```bash
uv run python python/scripts/metagame.py \
  --study-config configs/study/metagame_sensitivity.yaml \
  --final-eval-dir runs/some_run/eval/final_eval
```

What it does today:

- replays the actual final-eval matchup episodes through the metagame reporting pipeline for each configured sensitivity case (`S0`, `S1`, `S2`)
- writes `sensitivity/` artifacts with per-case payoff, Nash, and AlphaRank outputs
- exports delta tables versus `S0` for matchup `p_ij`, Nash mixture mass, and AlphaRank stationary mass

Default output location:

- `runs/some_run/eval/final_eval/sensitivity/`

### `artifact_scan.py`

Artifact hygiene gate over tracked data-like files plus generated artifact trees.

```bash
uv run python python/scripts/artifact_scan.py --artifact-root runs/toy_public_demo_ci
# or regenerate the built-in demo tree and scan it in one step
make artifact-hygiene
```

What it does today:

- scans tracked repo files via `git ls-files`
- scans text-like files (`.json`, `.jsonl`, `.csv`, `.txt`, `.yaml`, `.yml`) under selected artifact roots
- inspects suspicious file paths, narrow forbidden binary asset types, and replay-bundle zip members
- exits non-zero on likely bundled logo/art assets, card-text payload surfaces, or explicit franchise markers in artifact/data files
- intentionally skips markdown/docs prose surfaces for trademark matching to avoid documentation false positives

### `make_figures.py`

Paper figure renderer for completed run artifacts.

```bash
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir>
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir> --fig-id seat_bias
```

Current behavior:

- renders all paper figures by default, or a single figure when `--fig-id` is set
- stable figure IDs: `matchup_heatmap`, `truncation_heatmap`, `seat_bias`, `learning_curves`
- checks that the selected figure's required input artifacts exist before rendering
- reads `eval/final_eval/payoff_matrices/p_mean.csv`
- reads `eval/diagnostics/truncation_heatmap_data.csv`
- reads `eval/diagnostics/seat_bias.json`
- reads `training/logs/training_metrics.jsonl`
- writes `fig_*.pdf` and `fig_*.png` under `runs/<run_dir>/figures/paper/`
- with `--public-demo`, renders a clearly-labeled demo-only placeholder bundle from `final_eval/summary.json`

### `launch_experiments.py`

Single-node launcher for multi-seed and multi-stack runs. It round-robins jobs across a provided device list and falls back cleanly to one GPU or CPU.

```bash
uv run python python/scripts/launch_experiments.py \
  --group-label baseline_suite \
  --stack-config configs/presets/baselines/noleague_impala.yaml \
  --stack-config configs/presets/baselines/ppo_lite.yaml \
  --seed 1 \
  --seed 2 \
  --device cuda:0 \
  --device cuda:1
```

### `compare_runs.py`

Cross-run comparison renderer for baseline and scaling artifacts. It writes seed-aggregated benchmark plots plus JSON/CSV/Markdown summaries from existing run directories.

```bash
uv run python python/scripts/compare_runs.py \
  --run-dir runs/impala_main_seed1 \
  --run-dir runs/ppo_baseline_seed1
```

Or directly from a launcher/sweep group summary:

```bash
uv run python python/scripts/compare_runs.py \
  --launch-group-summary runs/launch_groups/impala_tune_a/summary.json
```

### `sweep_experiments.py`

Compact reproducible sweep launcher for the thesis baselines. It uses deterministic config overrides so each sweep run still has a proper canonical config hash and manifest.

```bash
uv run python python/scripts/sweep_experiments.py \
  --preset impala_compact \
  --group-label impala_tune_a \
  --seed 1 \
  --seed 2 \
  --device cuda:0
```

Shipped presets:
- `impala_compact`
- `ppo_compact`

Current non-claim:

- does not generate evaluation artifacts itself; it only renders figures from an existing run directory

Public-safe toy/demo figures:

```bash
uv run python python/scripts/make_figures.py \
  --public-demo \
  --final-eval-dir runs/toy_public_demo/eval/final_eval \
  --out-dir runs/toy_public_demo/figures
```
