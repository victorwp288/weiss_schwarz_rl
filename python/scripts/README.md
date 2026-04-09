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

Manifest/provenance scaffold entrypoint with an optional minimal inline training smoke path.

```bash
make train-min
# or, when you want the real M3-08 inline smoke path
make train-inline-smoke
# or directly
PYTHONPATH=../weiss-schwarz-simulator/python uv run python python/scripts/train.py --stack-config configs/rl_stack_locked.yaml --run-label m3_08_smoke --device cpu
```

What it does today:

- loads the stack config
- probes `weiss_sim.export_spec_bundle()` and verifies the runtime contract, unless you use `--public-demo`
- computes run IDs and writes `manifest.json`, `spec_bundle.json`, `config_canonical.json`, and scaffold directories under `runs/`
- when the active interpreter can import a simulator runtime with stepping APIs and the stack contains the training/model/environment blocks, it runs a tiny inline training smoke and writes training artifacts under `runs/<run>/training/`
- with `--public-demo`, stages a built-in public-safe toy catalog and deterministic toy policy bundle under `runs/<run>/public_demo/`

What it does **not** do today:

- no multi-actor rollout workers or learner queues
- no long-running production training loop
- no claim that the inline smoke path equals the full master-plan actor/learner system

`--run-label` controls the human-friendly run directory name only. The immutable computed run identity still comes from the runtime spec/config/git/nonce inputs and is reported separately in the startup banner and manifest. `--run-id` remains accepted as a deprecated compatibility alias for the label override.

`manifest.json` only records a resolved deterministic `policy_set_selection` when you also pass the matching `--snapshot-registry-json` and `--dev-eval-summaries-json` inputs. Otherwise the manifest records an unresolved `policy_set_selection_details` block instead of pretending the final set is already known.

If `weiss_sim` is not installed in the active interpreter, the script still tries to collect provenance through a working local simulator interpreter/check-out when available. Configure `WEISS_SIM_PYTHONPATH` and optionally `WEISS_SIM_PYTHON` if your simulator lives elsewhere.

Important: the **inline training smoke path** requires the active interpreter itself to import a simulator runtime with stepping APIs. `make train-inline-smoke` handles the usual sibling-checkout case by prepending `../weiss-schwarz-simulator/python` to `PYTHONPATH`. If only the provenance probe works, `train.py` falls back to manifest-only mode and prints why.

Public-safe toy/demo staging:

```bash
uv run python python/scripts/train.py \
  --stack-config configs/rl_stack_locked.yaml \
  --public-demo \
  --run-label toy_public_demo
```

That mode is explicitly synthetic. It writes demo-only catalog/policy artifacts and does not claim simulator training happened.

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
- with `--public-demo`, generates a deterministic demo-only `final_eval/` artifact tree from the staged toy catalog/policies

What it does **not** do today:

- outside `--public-demo`, no policy rollout execution
- outside `--public-demo`, no simulator-driven episode generation
- no hidden claim that a reported `run_label` is a computed run identity

`--run-label` is just a human label for the startup banner/log output. Unlike `train.py`, this script does not compute a run directory identity or persist the label into summary exports.

Public-safe toy/demo eval:

```bash
uv run python python/scripts/eval.py \
  --stack-config configs/rl_stack_locked.yaml \
  --public-demo \
  --run-dir runs/toy_public_demo
```

That path writes demo-only `final_eval/` artifacts and labels them clearly in metadata.

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
  --stack-config configs/rl_stack_locked.yaml \
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
  --stack-config configs/rl_stack_locked.yaml \
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

Current non-claim:

- does not generate evaluation artifacts itself; it only renders figures from an existing run directory

Public-safe toy/demo figures:

```bash
uv run python python/scripts/make_figures.py \
  --public-demo \
  --final-eval-dir runs/toy_public_demo/eval/final_eval \
  --out-dir runs/toy_public_demo/figures
```
