# weiss_schwarz_rl

Thesis RL pipeline for Weiss Schwarz.

This repo is strongest at config, contracts, provenance, and small standalone components. It now centers on a canonical single-node queue runtime for simulator-backed train/eval, with the published `weiss-sim` package as the validation source. A separate `stack_smoke` scaffold path and a public-safe toy/demo pipeline remain available for CI and artifact checks.

## Start here

Read these first:

- [Docs hub](docs/README.md)
- [Getting started](docs/getting_started.md)
- [Runtime modes](docs/runtime_modes.md)
- [Artifact contract](docs/artifact_contract.md)
- [Troubleshooting](docs/troubleshooting.md)

## Supported install paths

```bash
uv sync --extra dev
```

Optional simulator package extra:

```bash
uv sync --extra dev --extra sim
```

If you are not using `uv`, install the editable package with dev extras:

```bash
python -m pip install -e ".[dev]"
```

## Fast verification

```bash
uv run python python/scripts/verify_repo.py
# optional convenience wrappers
make verify
bash scripts/run_local_ci_parity.sh
```

`python/scripts/verify_repo.py` is the cross-platform verification entrypoint. `make verify` and `scripts/run_local_ci_parity.sh` delegate to the same release-facing checks when those wrappers are available in your shell.

## Working with runs

The repo keeps three paths separate on purpose: scaffold-only smoke, simulator-backed canonical train/eval, and the public-safe demo pipeline.

```bash
make train-min
make train-inline-smoke
make toy-public-e2e
make artifact-hygiene
```

`make train-min` is the scaffold-only path. The canonical thesis-oriented run paths are described in `docs/runtime_modes.md` and `docs/artifact_contract.md`, and they use the `DecisionBoundaryEnv` contract on top of `weiss-sim`. By default, the training hot path now runs on the simulator `fast` profile with packed legal IDs, while evaluation keeps the deterministic pinned protocol and only densifies legality where the analysis layer benefits from it.

The release-facing experiment surface is the `structured_acceptance_standard` family:

- `configs/presets/structured_acceptance_standard.yaml` is the canonical current training recipe.
- `configs/presets/structured_acceptance_standard_auto_gpu.yaml` is the canonical Linux server variant with automatic multi-GPU actor sharding.
- `configs/presets/structured_acceptance_standard_thesis_eval.yaml` is the canonical richer final-eval companion.
- `configs/presets/structured_acceptance_standard_multideck.yaml` is the canonical deck-diversity/generalization variant.
- `configs/presets/baselines/*.yaml` contains the comparison baselines.
- `configs/study/metagame_sensitivity.yaml` holds the study-only metagame/sensitivity settings.

Legacy typed presets remain available for older experiments and lower-level direct entrypoint access:

- `configs/presets/typed_thesis_locked.yaml`
- `configs/presets/typed_local.yaml`

`PPO-lite` is implemented in-repo and is mask-aware by construction, so it shares the same legality semantics, snapshot format, and evaluation pipeline as the main IMPALA path.

For tuning, the repo also ships compact sweep presets for the grouped typed and baseline flows. Those sweeps use deterministic config overrides, so each candidate still lands in its own canonical run with a distinct config hash instead of being a one-off shell mutation.

For operator safety, canonical runs now keep both `training/checkpoints/latest.pt` and `training/checkpoints/best.pt` plus a `checkpoint_tracker.json` manifest, and `train.py` can resume either in-place or from a direct checkpoint path. If you want one command that chains the release-facing thesis flow, use `python/scripts/thesis_run.py --preset standard`.

Canonical runs also write TensorBoard event files under `runs/<run>/tensorboard/`. Those events include run metadata, training/runtime scalars, checkpoint alias updates, periodic dev-eval summaries, and the post-run final-eval/metagame/readiness summaries. Inspect them with:

```bash
tensorboard --logdir runs/<run>/tensorboard
```
