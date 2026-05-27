# Training

The standard thesis entrypoints are:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label <run_label> --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label <run_label> --b1-run runs/<b1_run> --profile smoke
```

Use `--profile gpu-probe` for the short local CUDA preflight before launching
the 200-update `thesis-local` profile.

The standard thesis training configs now use the medium64 structured model
surface (`gru_hidden_size: 64`, `encoder_mlp_width: 64`,
`typed_feature_width: 16`). The old tiny32 surface is still available through
legacy presets, but it is no longer the B1/main thesis default.

The compatibility training entrypoint is:

```powershell
uv run python python/scripts/train.py --stack-config <config>
```

`python/scripts/thesis_workflow.py` mirrors the package CLI for older
script-oriented callers. `python/scripts/thesis_run.py` is retained only as a
legacy compatibility wrapper for named presets.

Reusable training helpers live under `weiss_rl.training`. Algorithm/model compatibility validation is in `weiss_rl.training.algorithm_contracts`; promotion-anchor resolution and small promotion support helpers live in `weiss_rl.training.promotion`. The public script still owns model construction, learner wiring, and simulator-backed promotion/eval loops.

## Safe Smoke

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
```

These execute one tiny learner update and write real simulator-backed smoke
artifacts. The old `configs/stack_smoke.yaml` manifest-only path is still
available for config/scaffold checks, but it is not a thesis training lane.

## Public Demo

```powershell
uv run python python/scripts/train.py `
  --stack-config configs/presets/structured_acceptance_standard.yaml `
  --public-demo `
  --run-label toy_public_demo
```

This stages synthetic public-safe artifacts only.

## Canonical Runtime

Training uses `QueueRuntime` with one of two runtime modes:

- `train_ordered`: deterministic merge order, best for debugging and reproducibility checks.
- `train_async_fast`: higher throughput, scheduling-dependent update ordering, provenance still recorded.

## Checkpoint Rhythm

Canonical runs write:

- `training/checkpoints/latest.pt`
- `training/checkpoints/best.pt`
- `training/checkpoints/observed_best.pt`
- `training/checkpoints/checkpoint_tracker.json`
- snapshot registry entries under `training/snapshots/`

`latest.pt` stays chronological. `best.pt` stays confidence-gated and is the
only alias used by checkpoint-guard rollback/finalization. `observed_best.pt`
tracks the highest periodic dev-eval score even when the candidate is too noisy
to promote, so follow-up confirmation can resume or evaluate it without manual
checkpoint hunting.

Do not change checkpoint cadence, alias semantics, or promotion rules without a failing characterization test.
