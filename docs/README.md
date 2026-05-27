# Docs Hub

This is the shortest map from onboarding to the thesis-oriented contracts and refactor safety notes.

The canonical simulator-backed path uses the published `weiss-sim` package, the single-node queue runtime, and the `DecisionBoundaryEnv` boundary-step contract. `stack_smoke.yaml` is only the scaffold path.

## Start here

- [Getting started](getting_started.md)
- [Thesis workflow](thesis_workflow.md)
- [Architecture](architecture.md)
- [Configuration](configuration.md)
- [Training](training.md)
- [Evaluation](evaluation.md)
- [Standard recipe](standard_recipe.md)
- [Runtime modes](runtime_modes.md)
- [Simulator compatibility](simulator_compatibility.md)
- [League](league.md)
- [Checkpoints](checkpoints.md)
- [Reproducibility](reproducibility.md)
- [Testing](testing.md)
- [Artifact contract](artifact_contract.md)
- [Artifacts](artifacts.md)
- [Experiments](experiments.md)
- [Rebuild log](rebuild_log.md)
- [Performance](performance.md)
- [Troubleshooting](troubleshooting.md)
- [Refactor log](refactor_log.md)
- [Refactor completion audit](refactor_completion_audit.md)
- [Archive](archive/README.md)

## Also useful

- [Training logs](training_logs.md)
- [Rebuild plan](../RL_REBUILD_PLAN.md)

## Suggested reading order

1. `getting_started.md`
2. `thesis_workflow.md`
3. `architecture.md`
4. `configuration.md`
5. `training.md`
6. `evaluation.md`
7. `artifact_contract.md`
8. `rebuild_log.md`

If you are only trying to verify a local checkout, run `uv run python python/scripts/verify_repo.py`. `make verify` remains a convenience wrapper, and the Bash parity wrapper is still available on Unix-like shells as `bash scripts/run_local_ci_parity.sh`.
