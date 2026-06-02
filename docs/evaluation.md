# Evaluation

Evaluation is a reporting contract. Preserve policy ordering, paired seeds,
seat swaps, payoff folding, uncertainty summaries, and output schemas.

## Entry Points

| Task | Command |
| --- | --- |
| Smoke eval | `uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/<run_dir> --b1-run runs/<b1_run>` |
| Thesis final eval | `uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<run_dir> --b1-run runs/<b1_run>` |
| Low-level eval | `uv run python -m weiss_rl.workflows.eval_entrypoint --stack-config configs/thesis/final_eval.yaml --run-dir runs/<run_dir>` |

Smoke eval uses the packed `configs/thesis/main_league.yaml` contract so it can
evaluate standard B1/main smoke checkpoints. Final eval uses the selected
factorized `configs/thesis/final_eval.yaml` contract.

## Public Demo

```powershell
uv run python -m weiss_rl.workflows.eval_entrypoint `
  --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml `
  --public-demo `
  --run-dir runs/toy_public_demo
```

Demo artifacts are synthetic and must not be cited as thesis results.

## Determinism

Final evaluation uses committed paired seed files, stable policy-set resolution,
pinned RNG, deterministic artifact paths, and explicit policy IDs. CPU is the
default eval device unless the config requests otherwise.

## Periodic Dev Eval

Training configs may run periodic dev eval during learning. That path is useful
for trend detection and checkpoint promotion, but it is not a replacement for
the selected final-eval contract.

Pure support helpers live in `weiss_rl.training.dev_eval`; simulator-backed
execution is wired through explicit training compatibility hooks so call order
and artifact writes stay auditable.

## Outputs

Canonical evaluation writes under `runs/<run>/eval/final_eval/`, including
matchup summaries, uncertainty payloads, diagnostics, payoff matrices, and
final policy-set metadata.

Current selected-run evidence is tracked in [artifacts.md](artifacts.md). Keep
this page focused on evaluation entrypoints and reporting semantics.

## B2 Flatline Diagnosis

Use the disagreement audit when B2 is flat, suspicious, or moving differently
from B0/B1:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli b2-audit `
  --run-dir runs/<run_dir> `
  --episodes-jsonl runs/<run_dir>/eval/final_eval/episodes.jsonl `
  --policy-id policy_000200
```

The source `episodes.jsonl` must be a seat-swapped focal-vs-B2 matchup. The
audit reruns those seeds with replay capture, compares learner and B2 action
preferences on identical public states, and writes
`runs/<run>/eval/b2_disagreement/audit/summary.json`.
