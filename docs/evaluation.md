# Evaluation

Evaluation is a thesis reporting contract. Preserve policy ordering, seed pairing, seat swaps, payoff folding, and summary schemas.

## Standard Entry Points

Smoke evaluation:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/<run_dir> --b1-run runs/<b1_run>
```

The smoke wrapper uses the plain packed `configs/thesis/main_league.yaml`
contract so it can evaluate the standard B1/main smoke checkpoints.

Thesis final evaluation:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/<run_dir> --b1-run runs/<b1_run>
```

The final wrapper uses `configs/thesis/final_eval.yaml`, the selected
factorized final-eval contract for thesis reproduction.

Low-level package entrypoint:

```powershell
uv run python -m weiss_rl.workflows.eval_entrypoint `
  --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml `
  --run-dir runs/<run_dir>
```

## Public Demo

```powershell
uv run python -m weiss_rl.workflows.eval_entrypoint `
  --stack-config configs/presets/structured_acceptance_standard_thesis_eval.yaml `
  --public-demo `
  --run-dir runs/toy_public_demo
```

Demo artifacts are synthetic and must not be cited as thesis results.

## Determinism

Final evaluation uses committed paired seed files, stable policy-set resolution, pinned RNG, and deterministic artifact paths. CPU is the default eval device unless the config explicitly requests otherwise.

## Periodic Dev Eval

Training presets may run periodic dev eval during training. Pure support helpers for that path live in `weiss_rl.training.dev_eval`: contract validation, seed-file resolution, deterministic RNG/bootstrap seeds, interval checks, log-path helpers, summary persistence, and stall-monitor updates. The simulator-backed runner and promotion-gate execution are wired through explicit training compatibility hooks so call order and artifact writes stay auditable.

## Outputs

Canonical evaluation writes under `runs/<run>/eval/final_eval/`, including matchup summaries, uncertainty payloads, diagnostics, and final policy-set metadata. Some canonical eval paths update run-level reports, so do not point ad-hoc eval commands at historical result directories casually.

## B2 Flatline Diagnosis

Use the standard disagreement audit when B2 is flat, suspicious, or improving
differently from B0/B1:

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
