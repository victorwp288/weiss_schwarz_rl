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
- probes `weiss_sim.export_spec_bundle()` and verifies the runtime contract
- computes run IDs and writes `manifest.json`, `spec_bundle.json`, `config_canonical.json`, and scaffold directories under `runs/`
- when the active interpreter can import a simulator runtime with stepping APIs and the stack contains the training/model/environment blocks, it runs a tiny inline training smoke and writes training artifacts under `runs/<run>/training/`

What it does **not** do today:

- no multi-actor rollout workers or learner queues
- no long-running production training loop
- no claim that the inline smoke path equals the full master-plan actor/learner system

`--run-label` controls the human-friendly run directory name only. The immutable computed run identity still comes from the runtime spec/config/git/nonce inputs and is reported separately in the startup banner and manifest. `--run-id` remains accepted as a deprecated compatibility alias for the label override.

`manifest.json` only records a resolved deterministic `policy_set_selection` when you also pass the matching `--snapshot-registry-json` and `--dev-eval-summaries-json` inputs. Otherwise the manifest records an unresolved `policy_set_selection_details` block instead of pretending the final set is already known.

If `weiss_sim` is not installed in the active interpreter, the script still tries to collect provenance through a working local simulator interpreter/check-out when available. Configure `WEISS_SIM_PYTHONPATH` and optionally `WEISS_SIM_PYTHON` if your simulator lives elsewhere.

Important: the **inline training smoke path** requires the active interpreter itself to import a simulator runtime with stepping APIs. `make train-inline-smoke` handles the usual sibling-checkout case by prepending `../weiss-schwarz-simulator/python` to `PYTHONPATH`. If only the provenance probe works, `train.py` falls back to manifest-only mode and prints why.

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

### `paper_readiness_check.py`

Paper-readiness guardrails over an existing `final_eval/` artifact tree.

```bash
uv run python python/scripts/paper_readiness_check.py \
  --final-eval-dir runs/some_run/eval/final_eval
```

What it does today:

- reads the approved `final_eval/summary.json` plus per-matchup `diagnostics.json` files
- writes a single `paper_readiness_summary.json` artifact
- checks aggregate truncation rate, global seat-bias alarm, and focal-policy win rate versus `B0 RandomLegal`
- exits non-zero when any readiness guardrail fails

Default focal-policy behavior:

- if you do not pass `--focal-policy-id`, the script uses the first non-baseline policy in `policy_ids`

### `make_figures.py`

Placeholder figure writer.

```bash
uv run python python/scripts/make_figures.py --out runs/figures/placeholder.txt
```

Current behavior:

- writes a placeholder artifact

Current non-claim:

- not a paper-ready figure pipeline yet
