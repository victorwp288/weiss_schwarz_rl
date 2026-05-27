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
- `python -m weiss_rl.cli train-b1 --profile thesis-local` is the canonical simulator-backed B1 path once you are ready for non-smoke training.
- `python -m weiss_rl.cli train-main --profile thesis-local --b1-run runs/<baseline_run>` is the canonical local main-league path once the dedicated B1 anchor exists.
- `python -m weiss_rl.cli train-main --profile thesis-server --b1-run runs/<baseline_run>` is the canonical server variant with process collectors.
- `configs/presets/typed_thesis_locked.yaml` and `configs/presets/typed_local.yaml` remain available as lower-level legacy/compatibility stack surfaces.
- `train.py --public-demo`, `eval.py --public-demo`, and `make_figures.py --public-demo` provide a synthetic public-safe toy/demo pipeline.

The full runtime modes are described in `runtime_modes.md`; this page stays focused on the quickest honest onboarding path.

Repo paths in this guide are relative to the repo root.

## Prereqs

- Python >= 3.10, < 3.13
- `uv` installed and available in your terminal (`pip install uv`)
- Access to `weiss_sim` with `export_spec_bundle()` available.
  - `uv sync --extra dev --extra sim` installs `weiss-sim 1.2.0`, which matches the current repo expectations.
  - If `weiss_sim` is already installed in your active Python environment, the probe uses that first.
  - Otherwise it checks `WEISS_SIM_PYTHONPATH` if set.
  - Otherwise it looks for a sibling checkout at `../weiss-schwarz-simulator/python` relative to this repo.
  - If the simulator needs a different interpreter, also set `WEISS_SIM_PYTHON=/path/to/python3.12`.

## Install

From the repo root:

```bash
uv sync --extra dev
```

That managed install now resolves the CUDA 12.4 PyTorch wheels on Windows and Linux by default. macOS and other non-CUDA platforms continue to use the platform-default PyTorch build.

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

Use this for the real thesis-oriented train/eval path. It exercises the single-node queue runtime through the `DecisionBoundaryEnv` contract and validates against `weiss-sim 1.2.0`.

Requirements:

- use `python -m weiss_rl.cli` for the standard thesis surface
- install `weiss-sim>=1.2.0,<2` with `uv sync --extra dev --extra sim`, or otherwise ensure the active interpreter can import the same validated `weiss_sim`
- keep the run on a single machine for the canonical path
- prepare a dedicated B1 NoLeague run first and pass it through `--b1-run`

Run:

```bash
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_anchor_seed1 --profile thesis-local
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label thesis_local --b1-run runs/b1_anchor_seed1 --profile thesis-local
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/thesis_local --b1-run runs/b1_anchor_seed1
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/thesis_local --format png --format pdf
```

The main call trains with `configs/thesis/main_league.yaml`, imports the
canonical B1 anchor from the dedicated baseline run, and evaluates with
`configs/thesis/final_eval.yaml`.

Expected training artifacts when the runtime is available:

- `runs/thesis_local/training/logs/scalars.jsonl`
- `runs/thesis_local/training/logs/training_metrics.jsonl`
- `runs/thesis_local/training/checkpoints/checkpoint_1.pt`
- `runs/thesis_local/training/checkpoints/checkpoint_metadata_1.json`

Inspection after either path:

```bash
cat runs/thesis_local/manifest.json
```

You should see fields like `run_id256`, `spec_hash256`, `config_hash256`, `simulator`, and `spec_bundle`.

## Optional follow-up checks

These are the follow-up checks for the other top-level entrypoints:

```bash
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/<run_dir> --b1-run runs/<b1_run>
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/<run_dir>
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/<run_dir> --fig-id seat_bias
uv run python python/scripts/verify_repo.py
```

Expected outcomes:

- `smoke-eval` runs a tiny B0-B4 deterministic eval.
- `figures` renders all paper figures by default, or a selected figure via `--fig-id` (`matchup_heatmap`, `truncation_heatmap`, `seat_bias`, `learning_curves`).
- `figures` checks the selected figure inputs before rendering and writes `fig_*.pdf` and `fig_*.png` under `runs/<run_dir>/figures/paper/`.

### Public-safe toy/demo e2e path

This path is intentionally synthetic. It exists so CI and public readers can exercise the same top-level scripts without shipping proprietary assets.

```bash
uv run python python/scripts/train.py \
  --stack-config configs/presets/structured_acceptance_standard.yaml \
  --public-demo \
  --run-label toy_public_demo

uv run python python/scripts/eval.py \
  --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml \
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
