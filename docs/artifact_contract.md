# Artifact Contract

This repo now treats the run tree as a contract, not an implementation accident.

## Canonical run-root files

- `manifest.json`
- `environment.json`
- `run_summary.json`
- `paper_readiness_summary.json`
- `determinism_report.json` when the pinned thesis pipeline is used
- `config_canonical.json`
- `spec_bundle.json`
- `spec_hash256.txt`

## Canonical tree

- `training/`
- `eval/final_eval/`
- `eval/diagnostics/`
- `eval/b2_disagreement/`
- `eval/metagame/`
- `replays/`
- `figures/paper/`
- `figures/data/`

## What belongs where

### `training/`

- learner metrics
- checkpoint artifacts
- training-time provenance and logging

### `eval/final_eval/`

- raw `episodes.jsonl`
- summary tables
- payoff matrices
- matchup manifests
- posterior or uncertainty artifacts when they are part of the selected evaluation path

### `eval/diagnostics/`

- seat-bias summaries
- truncation inputs
- replay verification summaries
- other run-level eval diagnostics

### `eval/metagame/`

- Nash outputs
- AlphaRank outputs, including the selection mode used (`local` or `global`)
- sensitivity deltas
- solver reports

### `replays/`

- raw simulator replay files
- replay indices
- replay verification annotations

### `figures/paper/`

- rendered thesis figures
- figure manifests and metadata

## Formats

- `JSONL` for raw episode streams
- `CSV` for tables and matrices
- `JSON` for manifests, summaries, and provenance
- `NPZ` only when a dense array representation is materially better than JSON

## Legacy compatibility

The repo may keep short-lived compatibility aliases while paths migrate, but paper-grade checks should consume the canonical tree only.

## Quality bar

- demo runs may be synthetic, but they must still be labeled clearly
- smoke profiles are plumbing checks and must not be confused with thesis-grade train/eval runs
- simulator-backed canonical runs are validated against the published `weiss-sim` package and should record the runtime spec bundle verbatim
- paper-grade runs must have resolved policy selection, stable ordering, and explicit provenance
- readiness should fail if it has to reconstruct missing canonical outputs from fallback paths

## Useful checks

```bash
make artifact-hygiene
uv run python python/scripts/verify_repo.py
make verify
bash scripts/run_local_ci_parity.sh
```
