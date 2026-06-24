# Eval Package Map

Use this package for simulator-backed policy evaluation, final-eval summaries,
and readiness checks. The stable public surface remains `weiss_rl.eval`; direct
owner imports live under the subfolders below.

## Public Surface

- `__init__.py`: public evaluation facade for records, sampling helpers,
  final-eval entrypoints, diagnostics, payoff folding, and readiness summaries.
- `final_eval.py`: compatibility facade for final-eval imports.
- `paper_readiness.py`: public paper-readiness entrypoint.

## Subpackages

- `simulator/`: evaluation harness, simulator game lifecycle, policy stepping,
  terminal records, paired-seat handling, and completed-game records.
- `sampling/`: pinned action sampling, model action surfaces, model sampling,
  helper math, and PCG RNG utilities.
- `heuristic_public/`: public-state heuristic policy, scoring, observation, and
  batch-selection helpers.
- `search/`: god-search and simulator rollout search diagnostics.
- `snapshots/`: B1/static policy resolution, snapshot registry resolution, and
  snapshot model loading.
- `replay/`: simulator replay and replay-capture helpers.
- `parallel/`: parallel final-eval planning, worker entrypoint, and core
  execution helpers.
- `analysis/`: diagnostics, exports, payoff folding, uncertainty, and stage-two
  summaries.
- `final/`, `policies/`, `readiness/`, `targeted_confirm/`: existing grouped
  final-eval, policy-set, artifact-contract, and targeted-confirm workflows.
