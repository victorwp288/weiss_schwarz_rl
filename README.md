# weiss_schwarz_rl

Behavior-sensitive thesis RL pipeline for Weiss Schwarz.

This repository contains the training, evaluation, league, checkpoint, reporting, and artifact tooling used for Weiss Schwarz reinforcement-learning experiments. It integrates with the `weiss-sim` simulator through an explicit decision-boundary contract and keeps public demo/scaffold paths separate from thesis-grade simulator-backed results.

## Start here

Read these first:

- [Docs hub](docs/README.md)
- [Thesis workflow](docs/thesis_workflow.md)
- [Getting started](docs/getting_started.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Runtime modes](docs/runtime_modes.md)
- [Checkpoints](docs/checkpoints.md)
- [League](docs/league.md)
- [Reproducibility](docs/reproducibility.md)
- [Testing](docs/testing.md)
- [Artifact contract](docs/artifact_contract.md)
- [Artifacts](docs/artifacts.md)
- [Troubleshooting](docs/troubleshooting.md)

## Supported install paths

```bash
uv sync --extra dev
```

On Windows and Linux, the managed `uv` path now resolves `torch` from PyTorch's CUDA 12.4 wheel index by default. On platforms without CUDA wheels, it falls back to the platform-default PyTorch build.

Optional simulator package extra:

```bash
uv sync --extra dev --extra sim
```

If you are not using `uv`, install the editable package with dev extras:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e ".[dev]"
```

## Fast verification

```bash
uv run python python/scripts/verify_repo.py
# optional convenience wrappers
make verify
bash scripts/run_local_ci_parity.sh
```

`python/scripts/verify_repo.py` is the cross-platform verification entrypoint. `make verify` and `scripts/run_local_ci_parity.sh` delegate to the same release-facing checks when those wrappers are available in your shell.

## Repository structure

- `python/weiss_rl/config/`: strict config models, parser, overrides, and hashes.
- `python/weiss_rl/envs/`: simulator-backed environment wrappers.
- `python/weiss_rl/runtime.py`: queue runtime and rollout collection.
- `python/weiss_rl/learners/`: IMPALA/V-trace and PPO-lite learners.
- `python/weiss_rl/model.py`: policy/value model and structured action scoring.
- `python/weiss_rl/eval/`: deterministic final eval, policy resolution, uncertainty, diagnostics, and paper readiness.
- `python/weiss_rl/league/`: snapshot registry, PFSP, promotion gates, and opponent pools.
- `python/weiss_rl/training/`: reusable training helpers extracted from public scripts.
- `python/scripts/`: path-based public CLI entrypoints.
- `configs/`: grouped stack presets and committed seed files.
- `docs/`: architecture, safety contracts, and contributor documentation.
- `runs/`, `run_logs/`, `vast_artifacts/`, `thesis_figures_final/`: thesis artifacts. Treat existing historical outputs as read-only.

## Working with runs

The repo keeps three paths separate on purpose: scaffold-only smoke, simulator-backed canonical train/eval, and the public-safe demo pipeline.

```bash
make train-min
make train-inline-smoke
make toy-public-e2e
make artifact-hygiene
```

`make train-min` is the scaffold-only path. The canonical thesis-oriented run paths are described in `docs/runtime_modes.md` and `docs/artifact_contract.md`, and they use the `DecisionBoundaryEnv` contract on top of `weiss-sim`. By default, the training hot path now runs on the simulator `fast` profile with packed legal IDs, while evaluation keeps the deterministic pinned protocol and only densifies legality where the analysis layer benefits from it.

The release-facing thesis workflow is:

```bash
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/main_smoke --format png
```

The public thesis configs live under `configs/thesis/`. The older
`structured_acceptance_standard` family remains as the compatibility preset
layer beneath those names:

- `configs/presets/structured_acceptance_standard.yaml` is the canonical current training recipe.
- `configs/presets/structured_acceptance_standard_auto_gpu.yaml` is the canonical Linux server variant with automatic multi-GPU actor sharding.
- `configs/presets/structured_acceptance_standard_thesis_eval.yaml` is the canonical richer final-eval companion.
- `configs/presets/structured_acceptance_standard_multideck.yaml` is the exploratory deck-diversity/generalization variant.
- `configs/presets/baselines/*.yaml` contains the comparison baselines.
- `configs/study/metagame_sensitivity.yaml` holds the study-only metagame/sensitivity settings.

The canonical `standard` lane uses `preset:main_deck_5hy_yotsuba_v1` for the focal model, B0, B1, and B2. The themed B3/B4 public heuristics keep their aggro/control decks as explicit robustness rows in final eval.

Legacy typed presets remain available for older experiments and lower-level direct entrypoint access:

- `configs/presets/typed_thesis_locked.yaml`
- `configs/presets/typed_local.yaml`

`PPO-lite` is implemented in-repo and is mask-aware by construction, so it shares the same legality semantics, snapshot format, and evaluation pipeline as the main IMPALA path.

For tuning, the repo also ships compact sweep presets for the grouped typed and baseline flows. Those sweeps use deterministic config overrides, so each candidate still lands in its own canonical run with a distinct config hash instead of being a one-off shell mutation.

For operator safety, canonical runs keep `training/checkpoints/latest.pt`, strict `training/checkpoints/best.pt`, `training/checkpoints/observed_best.pt`, and a `checkpoint_tracker.json` manifest. `latest.pt` is chronological, `best.pt` is promotion-gated, and `observed_best.pt` is the highest periodic dev-eval candidate for follow-up confirmation. The package CLI is the standard thesis entrypoint; `python/scripts/thesis_run.py` remains a compatibility wrapper for older workflows.

Canonical runs also write TensorBoard event files under `runs/<run>/tensorboard/`. Those events include run metadata, training/runtime scalars, checkpoint alias updates, periodic dev-eval summaries, and the post-run final-eval/metagame/readiness summaries. Inspect them with:

```bash
tensorboard --logdir runs/<run>/tensorboard
```

## Human play web UI

Use the React web UI to play a simulator-backed match as the human player
against a selected model or baseline. The UI lists readable deck preset names,
shows only simulator-legal actions, enriches visible card ids with catalog
names, and writes transcript artifacts under the selected run's `human_play/`
directory.

```powershell
cd web/human-play
npm install
npm run build
cd ../..
.\.venv\Scripts\python.exe -m weiss_rl.human_play.web_server --host 127.0.0.1 --port 8765 --static-dir web/human-play/dist
```

Then open `http://127.0.0.1:8765/`. If `/api/health` does not report
`human_decision_view: true`, install or sync against `weiss-sim>=1.2.0` with
the human decision view API before launching the web server.

## Contributing safely

This is a behavior-preserving research codebase. Do not change training semantics, evaluation semantics, config meanings, checkpoint compatibility, legal action ordering, RNG behavior, reward semantics, league behavior, promotion criteria, or metric aggregation unless a bug is proven with a failing test and documented.

Before changing risky code, read [CONTRIBUTING.md](CONTRIBUTING.md), [RL_REBUILD_PLAN.md](RL_REBUILD_PLAN.md), and [docs/rebuild_log.md](docs/rebuild_log.md). Keep public script paths compatible and record validation commands in the rebuild log.
