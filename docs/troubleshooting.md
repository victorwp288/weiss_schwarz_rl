# Troubleshooting

Start with the smallest command that isolates the failing surface.

## Simulator Import Or Spec Failure

```powershell
uv sync --extra dev --extra sim
uv run --extra dev --extra sim python -c "import weiss_sim; print(weiss_sim.__version__, weiss_sim.__file__, weiss_sim.SPEC_HASH)"
```

If using a sibling simulator checkout, ensure its built Python package is on
`PYTHONPATH`; a raw source tree is not enough if the Rust extension is missing.
See [simulator_compatibility.md](simulator_compatibility.md).

## Verification Failure

Run the reported command directly from the repository root. For maintained
focused checks, use [testing.md](testing.md).

```powershell
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
```

If a failure follows a refactor, identify whether it touched a behavior
boundary listed in [architecture.md](architecture.md).

## Artifact Or Readiness Failure

Do not use smoke/demo output as a substitute for a paper-grade run tree. Check
the canonical layout in [artifact_contract.md](artifact_contract.md), then run:

```powershell
uv run python -m weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint --dry-run
```

Retained outputs under `runs/`, `diagnostics/`, `vast_artifacts/`, and
`thesis_figures_final/` should be treated as read-only unless deliberately
replacing a thesis artifact.

## Config Or Docs Link Failure

```powershell
uv run python -m pytest -q python/weiss_rl/tests/test_public_config_surface_docs.py
```

This usually means a local Markdown link moved, a documented config path no
longer exists, or a noncanonical thesis ablation was advertised as public.
