# Documentation Maintenance

This page records docs ownership decisions so future cleanup passes do not
rediscover the same boundaries.

## Current Shape

- `README.md` is a quick orientation and tiny smoke mirror.
- `docs/README.md` owns navigation and the ownership map.
- `docs/getting_started.md` owns first install, verifier, and tiny smoke.
- `docs/thesis_workflow.md` owns commands, profiles, model surface, and deck
  policy.
- `docs/evaluation.md` owns evaluation semantics and public-demo warnings.
- `docs/artifact_contract.md` owns paper-grade run-tree requirements.
- `docs/artifacts.md` owns retained evidence, selected runs, and smoke/probe
  metrics.
- `docs/configuration.md` owns descriptive config guidance; `configs/README.md`
  is only the folder-local index.
- `docs/testing.md` owns validation commands.
- Local READMEs under `configs/`, `tests/`, and `runs/` should stay short and
  point back to owner docs.

## Decisions

- Do not merge smoke/demo artifact examples into the paper-grade artifact
  contract. Smoke output is plumbing evidence, not readiness evidence.
- Do not advertise every compatibility config as public thesis surface. Public
  docs should name only the canonical thesis launch configs and the three public
  ablations.
- Do not turn `docs/thesis_workflow.md` into a results log. Evidence belongs in
  `docs/artifacts.md`; workflow links to it.
- Do not move retained figure trace Markdown out of `thesis_figures_final/`; it
  is part of the compact artifact bundle.
- Keep code-refactor progress logs out of the public docs hub unless they
  change docs ownership or public behavior.

## 2026-06-02 Cleanup Pass

Current result:

- Public docs have a single navigation hub and fewer duplicated config,
  artifact, and smoke-evidence lists.
- Broken contributor references to missing refactor docs were removed.
- `AGENTS.md` exists in the checkout and matches the active long-running-task
  instructions.

Validation target:

```powershell
uv run python -m pytest -q python/weiss_rl/tests/test_public_config_surface_docs.py
```

## 2026-06-02 Markdown Refactor Pass

Current result:

- Removed redundant `docs/training.md`; `docs/thesis_workflow.md` owns training
  commands.
- Rebuilt `docs/README.md` around task-based navigation and source ownership.
- Rewrote `docs/architecture.md` as a conceptual package and behavior-boundary
  map instead of a module inventory.
- Tightened `docs/configuration.md`, `docs/testing.md`,
  `docs/artifact_contract.md`, `docs/evaluation.md`, and
  `docs/troubleshooting.md` into clearer reference pages.
- Synchronized the root README smoke commands with the simulator-extra command
  shape used by the docs.

Validation completed:

```powershell
uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_public_config_surface_docs.py
uv run --extra dev python -m weiss_rl.diagnostics.repo_hygiene_check_entrypoint --json
uv run --extra dev python -m weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint --dry-run
```

The docs/config, artifact hygiene, repo hygiene, and artifact-contract workflow
tests passed together: 20 tests.
