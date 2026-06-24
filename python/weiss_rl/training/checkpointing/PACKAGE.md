# Checkpointing Package Map

Use this package for checkpoint IO, promotion decisions, checkpoint guardrails,
and publication flows. Direct owner imports live under the subfolders below.

## Subpackages

- `aliases/`: current/best alias decisions, alias mutation, and publication
  helpers.
- `guards/`: checkpoint guard metrics, structured guardrails, periodic dev-eval
  guard flows, and promotion snapshots.
- `lifecycle/`: lifecycle plans, decisions, transitions, effects, finalization,
  and tracker state.
- `interpolation/`: checkpoint interpolation CLI, runtime, reporting, and core
  interpolation logic.
- `publishing/`: checkpoint publishing CLI, runtime, reporting, and entrypoint.
- `storage/`: checkpoint IO, load/write helpers, restore state, restore runtime,
  and path resolution.
