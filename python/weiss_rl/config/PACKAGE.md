# Config Package Map

Use this package for immutable config schemas, YAML loading, overrides, and
section parsing. The stable public surfaces remain `weiss_rl.config` and
`weiss_rl.config.models`.

## Public Surface

- `__init__.py`: public config facade for config dataclasses, hashing, loading,
  and overrides.
- `models.py`: compatibility re-export facade for immutable grouped config
  models.

## Subpackages

- `schemas/`: immutable dataclass groups for common, environment, evaluation,
  curriculum/league, study, and training config models.
- `sections/`: section-specific parsers for core, model, environment,
  curriculum, league, reproducibility, and training subsections.
- `loading/`: YAML parsing, study loading, seed sets, overrides, hashing, and
  parsing utilities.
