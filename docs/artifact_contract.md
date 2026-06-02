# Artifact Contract

Run trees are part of the thesis contract. A run is not paper-grade just because
it has checkpoints or a smoke eval; it must contain the canonical files and
directories below.

## Run Root

Canonical run-root files:

- `manifest.json`
- `environment.json`
- `run_summary.json`
- `paper_readiness_summary.json`
- `determinism_report.json` when the pinned thesis pipeline is used
- `config_canonical.json`
- `spec_bundle.json`
- `spec_hash256.txt`

Canonical directories:

- `training/`
- `eval/final_eval/`
- `eval/diagnostics/`
- `eval/b2_disagreement/`
- `eval/metagame/`
- `replays/`
- `figures/paper/`
- `figures/data/`

## Directory Ownership

| Path | Contents |
| --- | --- |
| `training/` | Learner metrics, checkpoints, snapshot registry, training provenance, and TensorBoard logs. |
| `eval/final_eval/` | Raw `episodes.jsonl`, summaries, payoff matrices, matchup manifests, and uncertainty artifacts. |
| `eval/diagnostics/` | Seat-bias summaries, truncation inputs, replay verification summaries, and run-level eval diagnostics. |
| `eval/b2_disagreement/` | Learner-vs-B2 causal disagreement audits and replay-backed summaries. |
| `eval/metagame/` | Nash outputs, AlphaRank outputs, sensitivity deltas, solver reports, and selection mode metadata. |
| `replays/` | Simulator replay files, replay indices, and verification annotations. |
| `figures/paper/` | Rendered thesis figures, figure manifests, and figure metadata. |
| `figures/data/` | Figure source tables and compact data exports. |

## Formats

- `JSONL` for raw episode streams.
- `CSV` for tables and matrices.
- `JSON` for manifests, summaries, reports, and provenance.
- `NPZ` only when a dense array representation is materially better than JSON.

## Quality Bar

- Demo runs may be synthetic, but they must be labeled clearly.
- Smoke profiles are plumbing checks and must not be cited as thesis-quality
  model evidence.
- Paper-grade readiness must consume the canonical tree, not reconstruct
  missing outputs from fallback paths.
- Simulator-backed canonical runs must record the published `weiss-sim` spec
  bundle verbatim.
- Policy selection, policy ordering, seeds, and provenance must be resolved
  explicitly before final reporting.

## Compatibility

Short-lived path aliases are acceptable while implementation paths migrate.
Paper-grade checks should consume canonical paths only.

## Checks

```powershell
uv run python -m weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint --dry-run
uv run python -m weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

On systems with `make`, the equivalent targets are `make artifact-contract` and
`make verify`.
