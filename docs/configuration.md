# Configuration

The repository uses grouped stack configs. Config files are behavior: defaults, override paths, inheritance, and hashes must remain stable unless a confirmed bug is documented.

## Loading

Use `load_stack_config(path)` from `weiss_rl.config`. The parser rejects unknown keys and resolves `extends` relative to repo-root config paths.

Common config roots:

- `configs/stack_smoke.yaml`: scaffold-only manifest smoke.
- `configs/presets/structured_acceptance_standard.yaml`: canonical current training recipe.
- `configs/presets/structured_acceptance_standard_thesis_eval.yaml`: richer final-eval companion.
- `configs/presets/baselines/`: comparison baseline surfaces.
- `configs/seeds/`: committed deterministic seed sets.

## Overrides

CLI overrides use grouped dotted paths:

```powershell
uv run python python/scripts/train.py `
  --stack-config configs/presets/structured_acceptance_standard.yaml `
  --override training.optimizer.learning_rate=0.0001
```

Override values are parsed as JSON. This keeps booleans, numbers, lists, and strings unambiguous.

## Hashes

`compute_config_hash256(stack)` hashes the canonical config dictionary. Refactors must preserve the hash for the same loaded config unless a behavior change is intentionally documented as a bug fix.

## Compatibility Notes

- Historical `config_canonical.json` payloads are compatibility inputs.
- Seed files are not casual tuning knobs; they define deterministic evaluation and promotion surfaces.
- CLI flag defaults can override or complement YAML values. Treat those defaults as public behavior.
