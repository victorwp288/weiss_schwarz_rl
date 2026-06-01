# Configuration

The repository uses grouped stack configs. Config files are behavior: defaults, override paths, inheritance, and hashes must remain stable unless a confirmed bug is documented.

## Loading

Use `load_stack_config(path)` from `weiss_rl.config`. The parser rejects unknown keys and resolves `extends` relative to repo-root config paths.

Common config roots:

- `configs/stack_smoke.yaml`: scaffold-only manifest smoke.
- `configs/thesis/b1_noleague.yaml`: canonical fixed-deck B1 NoLeague training lane.
- `configs/thesis/main_league.yaml`: canonical fixed-deck main league training lane.
- `configs/thesis/main_league_auto_gpu.yaml`: server-oriented main league lane with process collectors.
- `configs/thesis/final_eval.yaml`: canonical fixed-deck final evaluation companion.
- `configs/thesis/ablations/`: named thesis ablation surfaces.
- `configs/presets/`: lower-level compatibility and implementation presets.
- `configs/archive/`: historical dated probes kept for provenance, not current workflow entrypoints.
- `configs/seeds/`: committed deterministic seed sets.

## Overrides

The package CLI is the canonical thesis surface and keeps detailed tuning in
configs:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
```

The lower-level compatibility training script still accepts grouped dotted
overrides for debugging and ablation work:

```powershell
uv run python python/scripts/train.py `
  --stack-config configs/thesis/main_league.yaml `
  --override training.optimizer.learning_rate=0.0001
```

Override values are parsed as JSON. This keeps booleans, numbers, lists, and strings unambiguous.

## Hashes

`compute_config_hash256(stack)` hashes the canonical config dictionary. Refactors must preserve the hash for the same loaded config unless a behavior change is intentionally documented as a bug fix.

## Compatibility Notes

- Historical `config_canonical.json` payloads are compatibility inputs.
- Seed files are not casual tuning knobs; they define deterministic evaluation and promotion surfaces.
- CLI flag defaults can override or complement YAML values. Treat those defaults as public behavior.
