# Entry-point scripts

Run these from the repo root.

Preferred workflow after setup:

```bash
uv sync --extra dev
uv run python ...
```

If you are doing an ad-hoc local invocation without installing the package, use `PYTHONPATH=python python ...` explicitly. For normal development, prefer `uv run` or an editable install.

## Current script status

### `train.py`

Manifest/provenance smoke entrypoint.

```bash
make train-min
# or
uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label smoke_local
```

What it does today:

- loads the stack config
- probes `weiss_sim.export_spec_bundle()` and verifies the runtime contract
- computes run IDs and writes `manifest.json`, `spec_bundle.json`, `config_canonical.json`, and scaffold directories under `runs/`

What it does **not** do today:

- no actor rollout collection
- no learner update loop
- no checkpointed training run beyond scaffold directory creation

`--run-label` controls the human-friendly run directory name only. The immutable computed run identity still comes from the runtime spec/config/git/nonce inputs and is reported separately in the startup banner and manifest. `--run-id` remains accepted as a deprecated compatibility alias for the label override.

If `weiss_sim` is not installed in the active interpreter, the script falls back to a working local simulator interpreter/check-out when available. Configure `WEISS_SIM_PYTHONPATH` and optionally `WEISS_SIM_PYTHON` if your simulator lives elsewhere.

### `eval.py`

Evaluation reporting and contract-check entrypoint.

Contract-only smoke check:

```bash
uv run python python/scripts/eval.py --stack-config configs/rl_stack_locked.yaml
```

Summary export from an existing episodes file:

```bash
uv run python python/scripts/eval.py \
  --stack-config configs/rl_stack_locked.yaml \
  --episodes-jsonl runs/some_eval/episodes.jsonl \
  --summary-json runs/some_eval/summary.json \
  --summary-csv runs/some_eval/summary.csv \
  --diagnostics-json runs/some_eval/diagnostics.json
```

What it does today:

- validates config-hash and optional simulator contract inputs
- reports the evaluation config/contract surface
- summarizes an existing seat-swapped `episodes.jsonl` file into JSON/CSV and optional diagnostics

What it does **not** do today:

- no policy rollout execution
- no simulator-driven episode generation
- no hidden claim that a reported `run_label` is a computed run identity

`--run-label` is just a human label for the startup banner/log output. Unlike `train.py`, this script does not compute a run directory identity or persist the label into summary exports.

### `make_figures.py`

Placeholder figure writer.

```bash
uv run python python/scripts/make_figures.py --out runs/figures/placeholder.txt
```

Current behavior:

- writes a placeholder artifact

Current non-claim:

- not a paper-ready figure pipeline yet
