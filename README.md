# weiss_schwarz_rl

Thesis RL pipeline scaffold for Weiss Schwarz.

Today this repo is strongest at config, contracts, provenance, and small standalone components. It now also includes a **minimal inline training smoke path** in `python/scripts/train.py` plus a clearly-labeled **public-safe toy/demo pipeline** for CI and artifact checks, but it still does **not** provide the full multi-actor training or proprietary evaluation pipeline from the top-level entrypoints.

## Current capability snapshot

- `make train-min` performs a manifest/provenance smoke run via `configs/stack_smoke.yaml`.
  - It loads the stack config, verifies the runtime simulator spec bundle, computes run IDs, and writes `runs/<run_dir>/manifest.json` plus scaffold directories.
  - It does **not** execute learner updates or rollout collection.
- `python/scripts/train.py` can also run a **minimal inline end-to-end training smoke** when you pass a full training stack such as `configs/rl_stack_locked.yaml` and the active interpreter can import a simulator runtime with stepping APIs. The `make train-inline-smoke` target wires in the standard sibling simulator checkout (`../weiss-schwarz-simulator/python`) when present.
  - That path writes the same provenance scaffold plus `training/logs/scalars.jsonl`, `training/logs/training_metrics.jsonl`, and `training/checkpoints/` artifacts under the run directory.
  - If the active interpreter can only expose `weiss_sim.export_spec_bundle()` but not the stepping runtime, the script falls back to manifest-only mode and prints the reason.
- `python/scripts/eval.py` performs contract checks, can summarize an existing seat-swapped `episodes.jsonl` file into JSON/CSV plus diagnostics, and can run a **public-safe toy/demo final-eval path** when paired with `train.py --public-demo`.
  - Outside `--public-demo`, it still does **not** launch evaluation rollouts or generate episodes from policies.
- `python/scripts/make_figures.py` renders paper figures from completed run artifacts under `runs/.../figures/paper/`, either all at once or by stable `--fig-id`.
  - It can also render a clearly-labeled **toy/demo figure bundle** from public-demo final-eval artifacts.
- `examples/run_loop_example.py` exercises the `weiss_sim` stepping API only. It is a simulator smoke example, not RL training.

## Not implemented end-to-end yet

- the full multi-actor / queue-based IMPALA pipeline from the master plan
- evaluation rollout generation against the proprietary simulator via `python/scripts/eval.py`
- paper-ready figure generation via `python/scripts/make_figures.py`

## Layout

- `python/weiss_rl/`: RL package modules (env wrappers, learner, eval, metagame, replay, plotting).
- `python/scripts/`: runnable entrypoint scripts with their current limits documented in `python/scripts/README.md`.
- `configs/`: locked config stack and committed seed sets.
- `docs/`: focused notes for getting started and standalone subsystems.
- `examples/`: small local examples for simulator usage and learner-side logging.
- `runs/`: generated run artifacts (kept out of git except placeholders).

## Setup

### uv (recommended)

```bash
uv sync --extra dev
```

### Editable install with pip extras

```bash
python -m pip install -e ".[dev]"
```

### Torch / CUDA note

This project depends on `torch` without forcing CPU-only builds.
If you want a CUDA wheel, install the matching PyTorch build for your platform after syncing, for example:

```bash
uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

### Pre-commit

```bash
uv run --extra dev pre-commit install
uv run --extra dev pre-commit run -a
```

## Fastest honest resume path

From the repo root:

```bash
uv sync --extra dev
make train-min
make train-inline-smoke   # uses the standard sibling simulator checkout when present
uv run python python/scripts/eval.py --stack-config configs/rl_stack_locked.yaml
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir>
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir> --fig-id seat_bias
make toy-public-e2e       # built-in public-safe toy/demo artifacts only
```

Then read:

- `docs/getting_started.md`
- `python/scripts/README.md`
- `docs/training_logs.md`

## Public-safe toy/demo path

This is the smallest honest end-to-end path that works without proprietary simulator assets.
It is explicitly demo-only and all generated artifacts are labeled that way.

```bash
uv run python python/scripts/train.py \
  --stack-config configs/rl_stack_locked.yaml \
  --public-demo \
  --run-label toy_public_demo

uv run python python/scripts/eval.py \
  --stack-config configs/rl_stack_locked.yaml \
  --public-demo \
  --run-dir runs/toy_public_demo

uv run python python/scripts/make_figures.py \
  --public-demo \
  --final-eval-dir runs/toy_public_demo/eval/final_eval \
  --out-dir runs/toy_public_demo/figures
```

Outputs include:

- `runs/toy_public_demo/public_demo/catalog.json`
- `runs/toy_public_demo/eval/final_eval/summary.json`
- `runs/toy_public_demo/figures/toy_demo_manifest.json`

## Quick checks

```bash
uv run python -c "import weiss_rl; print(weiss_rl.__all__)"
uv run pytest -q python/weiss_rl/tests
make train-min
make toy-public-e2e
```

## Schema validation line

1. `uv run python -c "from weiss_rl.trajectory.schema import TRAJ_SCHEMA_VERSION, TrajectoryStep; assert TRAJ_SCHEMA_VERSION==1; _=TrajectoryStep; print('ok')"`
2. `uv run python -c "import weiss_rl.trajectory.schema as s; print('schema_version', s.TRAJ_SCHEMA_VERSION); print('schema_path', s.__file__)"`
