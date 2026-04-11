# Getting Started (Dev)

Goal: get a fresh clone to a successful verification run without needing a 1:1 walkthrough.

## Read first

- [Docs hub](README.md)
- [Runtime modes](runtime_modes.md)
- [Artifact contract](artifact_contract.md)
- [Troubleshooting](troubleshooting.md)

## Current entrypoint reality

The repo now supports three useful lanes:

- `make train-min` proves config loading, simulator provenance capture, and run-manifest writing.
- `train.py` can run a tiny inline smoke path when the active interpreter can step the simulator.
- `train.py --public-demo`, `eval.py --public-demo`, and `make_figures.py --public-demo` provide a synthetic public-safe toy/demo pipeline.

The full thesis runtime is described in `runtime_modes.md`; this page stays focused on the quickest honest onboarding path.

Repo paths in this guide are relative to the repo root.

## Prereqs

- Python >= 3.10, < 3.13
- `uv` installed and available in your terminal (`pip install uv`)
- Access to `weiss_sim` with `export_spec_bundle()` available.
  - If `weiss_sim` is already installed in your active Python environment, the probe uses that first.
  - Otherwise it checks `WEISS_SIM_PYTHONPATH` if set.
  - Otherwise it looks for a sibling checkout at `../weiss-schwarz-simulator/python` relative to this repo.
  - If the simulator needs a different interpreter, also set `WEISS_SIM_PYTHON=/path/to/python3.12`.

## Install

From the repo root:

```bash
uv sync --extra dev
```

If you want the published simulator dependency instead of a sibling checkout, use:

```bash
uv sync --extra dev --extra sim
```

If `uv sync` fails with `No pyproject.toml found`, you are not in the repo root. In that case, include your current directory (`pwd`) plus the full error output when asking for help.

Preferred command style after install:

```bash
uv run python ...
```

If you are doing an ad-hoc local invocation without installing the package, set `PYTHONPATH=python` explicitly. That is mostly useful for targeted tests and local debugging, not the default onboarding path.

## Manifest smoke test (<2 minutes CPU)

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

- `Loaded stack config with 0 components`
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

## Expected output files

After the smoke run, you should have either:

- `runs/smoke_local/manifest.json`
- `runs/smoke_local/spec_bundle.json`
- `runs/smoke_local/config_canonical.json`

when you used the explicit `--run-label smoke_local` command above, or the same files under `runs/run_<computed_run_id64>/` when you used `make train-min`.

### Minimal inline training smoke (M3-08 bring-up)

This is the small real training path behind the current M3-08 work.
It still is **not** the full actor/learner architecture from the master plan, but it does run a tiny inline rollout -> learner update -> artifact write path.

Requirements:

- use a full stack config such as `configs/rl_stack_locked.yaml`
- the **active interpreter** must be able to import `weiss_sim` with stepping APIs, not just `export_spec_bundle()`
- `make train-inline-smoke` handles the usual sibling-checkout case by prepending `../weiss-schwarz-simulator/python` to `PYTHONPATH`

Run:

```bash
make train-inline-smoke
# or
uv run python python/scripts/train.py --stack-config configs/rl_stack_locked.yaml --run-label m3_08_smoke --device cpu
```

Expected training artifacts when the runtime is available:

- `runs/m3_08_smoke/training/logs/scalars.jsonl`
- `runs/m3_08_smoke/training/logs/training_metrics.jsonl`
- `runs/m3_08_smoke/training/checkpoints/checkpoint_1.pt`
- `runs/m3_08_smoke/training/checkpoints/checkpoint_metadata_1.json`

If the active interpreter still cannot step the simulator, `train.py` writes the manifest scaffold and then prints a manifest-only fallback reason instead of pretending training ran.

Inspection (replace `smoke_local` with the generated `run_<computed_run_id64>` directory if you used `make train-min`):

```bash
cat runs/smoke_local/manifest.json
```

You should see fields like `run_id256`, `spec_hash256`, `config_hash256`, `simulator`, and `spec_bundle`.

## Optional follow-up checks

These are honest smoke checks for the other top-level entrypoints:

```bash
uv run python python/scripts/eval.py --stack-config configs/rl_stack_locked.yaml
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir>
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir> --fig-id seat_bias
make verify
bash scripts/run_local_ci_parity.sh
```

Expected outcomes:

- `eval.py` reports a contract check and seed-set summary when no `--episodes-jsonl` is supplied.
- `make_figures.py` renders all paper figures by default, or a selected figure via `--fig-id` (`matchup_heatmap`, `truncation_heatmap`, `seat_bias`, `learning_curves`).
- `make_figures.py` checks the selected figure inputs before rendering and writes `fig_*.pdf` and `fig_*.png` under `runs/<run_dir>/figures/paper/`.

### Public-safe toy/demo e2e path

This path is intentionally synthetic. It exists so CI and public readers can exercise the same top-level scripts without shipping proprietary assets.

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
