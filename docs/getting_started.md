# Getting Started (Dev)

Goal: get a fresh clone to a successful verification run without needing a 1:1 walkthrough.

## Read first

- [Docs hub](README.md)
- [Runtime modes](runtime_modes.md)
- [Artifact contract](artifact_contract.md)
- [Troubleshooting](troubleshooting.md)

## Current entrypoint reality

The repo now supports three useful lanes:

- `configs/stack_smoke.yaml` with `make train-min` is the explicit scaffold-only path. It proves config loading, simulator provenance capture, and run-manifest writing, but it does not claim thesis-grade training.
- `python/scripts/thesis_run.py --preset thesis-model-auto-gpu --run-label <run> --b1-baseline-run-dir runs/<baseline_run>` is the frozen simulator-backed thesis path once the dedicated B1 anchor exists.
- `configs/presets/typed_thesis_locked.yaml` and `configs/presets/typed_local.yaml` remain available as lower-level legacy/compatibility stack surfaces.
- `train.py --public-demo`, `eval.py --public-demo`, and `make_figures.py --public-demo` provide a synthetic public-safe toy/demo pipeline.

The full runtime modes are described in `runtime_modes.md`; this page stays focused on the quickest honest onboarding path.

Repo paths in this guide are relative to the repo root.

## Prereqs

- Python >= 3.10, < 3.13
- `uv` installed and available in your terminal (`pip install uv`)
- Access to `weiss_sim` with `export_spec_bundle()` available.
  - `uv sync --extra dev --extra sim` installs `weiss-sim 0.8.1`, which matches the current repo expectations.
  - If `weiss_sim` is already installed in your active Python environment, the probe uses that first.
  - Otherwise it checks `WEISS_SIM_PYTHONPATH` if set.
  - Otherwise it looks for a sibling checkout at `../weiss-schwarz-simulator/python` relative to this repo.
  - If the simulator needs a different interpreter, also set `WEISS_SIM_PYTHON=/path/to/python3.12`.

## Install

From the repo root:

```bash
uv sync --extra dev
```

That managed install prefers the CUDA 12.4 PyTorch index on supported CPython Linux/Windows targets and otherwise falls back to the default PyPI `torch` build.

If you want the released simulator dependency instead of relying on a sibling checkout, use:

```bash
uv sync --extra dev --extra sim
```

If `uv sync` fails with `No pyproject.toml found`, you are not in the repo root. In that case, include your current directory (`pwd`) plus the full error output when asking for help.

Preferred command style after install:

```bash
uv run python ...
```

If you are doing an ad-hoc local invocation without installing the package, set `PYTHONPATH=python` explicitly. That is mostly useful for targeted tests and local debugging, not the default onboarding path.

## Scaffold smoke test (<2 minutes CPU)

This does **not** execute training. It verifies:

- the train entrypoint runs
- the strict stack config loader accepts the dedicated smoke stack
- the run scaffold records the real simulator spec bundle and provenance

Run:

```bash
uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label smoke_local
```

Or use the convenience target (same smoke run, but without an explicit label override):

```bash
make train-min
```

Expected console lines for the explicit `--run-label smoke_local` command:

- `Loaded stack config`
- `computed_run_id64:` and `computed_run_id256:` entries in the startup contract
- `run_label:              smoke_local`
- `run_dir_name:           smoke_local`
- `Wrote manifest: /absolute/path/to/repo/runs/smoke_local/manifest.json`
- `Manifest scaffold only: no learner training or rollout collection was executed.`

If you use `make train-min` instead, expect the same contract/manifest scaffold flow but with:

- `run_label:              (default)`
- `run_dir_name:           run_<computed_run_id64>`
- `Wrote manifest: /absolute/path/to/repo/runs/run_<computed_run_id64>/manifest.json`

`--run-label` only controls the human-friendly run directory name. The computed run identity is recorded separately in the banner and manifest. `--run-id` is still accepted as a deprecated compatibility alias for the label override.

## Canonical simulator-backed run

Use this for the real thesis-oriented train/eval path. It exercises the single-node queue runtime through the `DecisionBoundaryEnv` contract and validates against `weiss-sim 0.8.1`.

Requirements:

- use `python/scripts/thesis_run.py --preset thesis-model-auto-gpu` for the frozen multi-GPU thesis path
- install `weiss-sim 0.8.1` with `uv sync --extra dev --extra sim`, or otherwise ensure the active interpreter can import the same validated `weiss_sim`
- keep the run on a single machine for the canonical path
- prepare a dedicated `baseline_noleague` run first and pass it through `--b1-baseline-run-dir`

Run:

```bash
uv run python python/scripts/train.py --stack-config configs/presets/baselines/structured_acceptance_thesis_model_auto_gpu_noleague.yaml --run-label b1_anchor_thesis_model_seed1 --num-envs 2048 --unroll-length 64 --runtime-mode train_async_fast --max-updates 200
uv run python python/scripts/thesis_run.py --preset thesis-model-auto-gpu --run-label thesis_model_seed1 --b1-baseline-run-dir runs/b1_anchor_thesis_model_seed1 --num-envs 2048 --unroll-length 64 --runtime-mode train_async_fast --max-updates 400 --skip-compare
uv run python python/scripts/make_figures.py --run-dir runs/thesis_model_seed1
```

That wrapper call trains with the frozen tiny248 thesis model preset, imports the canonical B1 anchor from the dedicated baseline run, and by default evaluates with `structured_acceptance_thesis_model_eval_auto_gpu.yaml`.

Expected training artifacts when the runtime is available:

- `runs/thesis_model_seed1/training/logs/scalars.jsonl`
- `runs/thesis_model_seed1/training/logs/training_metrics.jsonl`
- `runs/thesis_model_seed1/training/checkpoints/checkpoint_1.pt`
- `runs/thesis_model_seed1/training/checkpoints/checkpoint_metadata_1.json`

Inspection after either path:

```bash
cat runs/thesis_model_seed1/manifest.json
```

You should see fields like `run_id256`, `spec_hash256`, `config_hash256`, `simulator`, and `spec_bundle`.

## Optional follow-up checks

These are the follow-up checks for the other top-level entrypoints:

```bash
uv run python python/scripts/eval.py --stack-config configs/presets/structured_acceptance_thesis_model_eval_auto_gpu.yaml
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir>
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir> --fig-id seat_bias
uv run python python/scripts/verify_repo.py
```

Expected outcomes:

- `eval.py` reports a contract check and seed-set summary when no `--episodes-jsonl` is supplied.
- `make_figures.py` renders all paper figures by default, or a selected figure via `--fig-id` (`matchup_heatmap`, `truncation_heatmap`, `seat_bias`, `learning_curves`).
- `make_figures.py` checks the selected figure inputs before rendering and writes `fig_*.pdf` and `fig_*.png` under `runs/<run_dir>/figures/paper/`.

### Public-safe toy/demo e2e path

This path is intentionally synthetic. It exists so CI and public readers can exercise the same top-level scripts without shipping proprietary assets.

```bash
uv run python python/scripts/train.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_auto_gpu.yaml \
  --public-demo \
  --run-label toy_public_demo

uv run python python/scripts/eval.py \
  --stack-config configs/presets/structured_acceptance_thesis_model_eval_auto_gpu.yaml \
  --public-demo \
  --run-dir runs/toy_public_demo

uv run python python/scripts/make_figures.py \
  --public-demo \
  --final-eval-dir runs/toy_public_demo/eval/final_eval \
  --out-dir runs/toy_public_demo/figures
```

Expected demo-only artifacts:

- `runs/toy_public_demo/public_demo/catalog.json`
- `runs/toy_public_demo/public_demo/policy_manifest.json`
- `runs/toy_public_demo/eval/final_eval/summary.json`
- `runs/toy_public_demo/figures/toy_demo_manifest.json`

Those artifacts are public-safe toy outputs only. Do not treat them as thesis results.

## Examples

If you also want a simulator-only loop smoke example and `weiss_sim` is importable:

```bash
uv run python examples/run_loop_example.py --steps 30
```

That example exercises simulator stepping only. It is not a learner training loop.

The standalone training log example can be run with:

```bash
uv run python examples/training_logs_example.py
```

That example exercises the learner-side JSONL logger directly, not `python/scripts/train.py`.

## Resume tomorrow checklist

When picking the repo back up later, start here:

1. `make train-min`
2. `python/scripts/README.md`
3. `docs/training_logs.md`
4. `examples/run_loop_example.py` only if you specifically need simulator API context

## Common errors

`ModuleNotFoundError: No module named 'weiss_rl'`
- Cause: running outside the managed environment or without installing the repo package.
- Fix: run the command via `uv run ...` and ensure `uv sync --extra dev` succeeded.

`Python 3.13` or newer
- Cause: this repo is intentionally narrowed to the supported thesis/test matrix.
- Fix: use Python 3.10, 3.11, or 3.12.

`Unable to collect simulator provenance via weiss_sim.export_spec_bundle()`
- Cause: the train scaffold could not import a compatible `weiss_sim` package or find a working simulator checkout/interpreter.
- Fix: either install `weiss_sim` into the active environment with `uv sync --extra dev --extra sim`, or set `WEISS_SIM_PYTHONPATH` (and, if needed, `WEISS_SIM_PYTHON`) to a working simulator environment.

`--stack-config` not found or YAML load error
- Cause: wrong working directory or incorrect config path.
- Fix: run from repo root and verify the config file exists.
