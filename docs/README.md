# Docs Hub

This is the shortest map from onboarding to the thesis-oriented contracts.

The canonical simulator-backed path uses the published `weiss-sim` package, the single-node queue runtime, and the `DecisionBoundaryEnv` boundary-step contract. `stack_smoke.yaml` is only the scaffold path.

## Start here

- [Getting started](getting_started.md)
- [Thesis Model Recipe](standard_recipe.md)
- [Runtime modes](runtime_modes.md)
- [Artifact contract](artifact_contract.md)
- [Troubleshooting](troubleshooting.md)

## Also useful

- [Training logs](training_logs.md)

## Suggested reading order

1. `getting_started.md`
2. `runtime_modes.md`
3. `artifact_contract.md`
4. `troubleshooting.md`

If you are only trying to verify a local checkout, run `uv run python python/scripts/verify_repo.py`. `make verify` remains a convenience wrapper, and the Bash parity wrapper is still available on Unix-like shells as `bash scripts/run_local_ci_parity.sh`.
