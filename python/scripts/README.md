# Entry-Point Scripts

Run commands from the repository root.

The canonical thesis surface is the package CLI:

```bash
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke
uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/main_smoke --format png
```

The scripts in this directory remain for compatibility, diagnostics, public-demo
smokes, and lower-level experiments. Prefer the package CLI for B1, main league,
final evaluation, smoke evaluation, and figure export unless you are deliberately
debugging a lower-level path.

## Verification

```bash
uv run --extra dev --extra sim python python/scripts/verify_repo.py
```

If GNU Make is installed, these wrappers are available:

```bash
make verify
make train-b1-smoke
make train-main-smoke
make eval-smoke
make figures-smoke
make thesis-smoke
```

`make train-min` is scaffold-only. It verifies config loading, simulator
provenance capture, and manifest writing with `configs/stack_smoke.yaml`; it is
not a learner-training lane.

## Compatibility Scripts

| Script | Current role |
| --- | --- |
| `train.py` | Lower-level training entrypoint used by package CLI workflows, compatibility wrappers, scaffold smoke, public demo, and direct ablation/debug runs. |
| `eval.py` | Lower-level evaluation/reporting entrypoint and public-demo final-eval generator. Canonical thesis eval should use `python -m weiss_rl.cli eval-final` or `smoke-eval`. |
| `make_figures.py` | Lower-level paper-figure renderer. Canonical figure export should use `python -m weiss_rl.cli figures`. |
| `thesis_workflow.py` | Thin compatibility wrapper that delegates to `weiss_rl.cli`. |
| `thesis_run.py` | Legacy preset wrapper retained for older command records. Prefer `weiss_rl.cli` for new B1/main workflow runs. |
| `paper_readiness_check.py` | Paper-readiness audit over a full run directory. Keep demo and thesis-grade readiness fixtures separate. |
| `write_paper_readiness_fixture.py` | Generates the dedicated readiness fixture used by artifact-contract checks. |
| `artifact_scan.py` | Artifact hygiene scanner for generated artifact trees and tracked data-like files. |
| `replay_inspector.py` | Replay-state policy comparison diagnostic. |
| `metagame.py` | Metagame/sensitivity reporting over an existing final-eval artifact tree. |
| `launch_experiments.py`, `compare_runs.py`, `sweep_experiments.py` | Lower-level experiment helpers for historical or exploratory workflows. |

## Direct Training Script

Use `train.py` directly only when you need the lower-level stack-config surface:

```bash
uv run --extra dev --extra sim python python/scripts/train.py \
  --stack-config configs/thesis/ablations/no_gru.yaml \
  --run-label ablate_no_gru_seed1 \
  --b1-baseline-run-dir runs/b1_anchor_seed1
```

Public-safe demo staging is synthetic and not thesis evidence:

```bash
uv run python python/scripts/train.py \
  --stack-config configs/presets/structured_acceptance_standard.yaml \
  --public-demo \
  --run-label toy_public_demo
```

The only supported manifest-only path is:

```bash
uv run python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label scaffold_smoke
```

## Direct Evaluation Script

Contract-only evaluation smoke:

```bash
uv run python python/scripts/eval.py --stack-config configs/thesis/final_eval.yaml
```

Public-demo evaluation:

```bash
uv run python python/scripts/eval.py \
  --stack-config configs/thesis/final_eval.yaml \
  --public-demo \
  --run-dir runs/toy_public_demo
```

For thesis claims, use saved simulator-backed run artifacts and the package CLI
final-eval route instead of public-demo outputs.

## Direct Figure Script

```bash
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir>
uv run python python/scripts/make_figures.py --run-dir runs/<run_dir> --fig-id seat_bias
```

Stable figure IDs are `matchup_heatmap`, `truncation_heatmap`, `seat_bias`, and
`learning_curves`. Outputs are written under `runs/<run_dir>/figures/paper/`.

## Invariants

- Do not treat public-demo or scaffold outputs as thesis results.
- Do not modify historical `runs/`, `run_logs/`, `vast_artifacts/`, checkpoints,
  or final figures from cleanup work.
- Do not hide new behavior behind script compatibility wrappers; new standard
  workflow behavior belongs in `weiss_rl.cli` and `weiss_rl.workflows`.
