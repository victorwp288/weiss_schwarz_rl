# Eval Support Workflow Package Map

Use this package for the evaluation CLI/workflow support layer. The root package
is intentionally thin; direct owner imports live under the subfolders below.

## Subpackages

- `dispatch/`: dependency assembly, dispatch requests, route adapters, routes,
  and execution dispatch.
- `modes/`: public-demo, summary, and shared evaluation-mode handling.
- `parser/`: CLI parser construction, parser arguments, and validation.
- `policy_selection/`: final policy set resolution, manifest selection, and
  policy-selection result records.
- `reports/`: report IO, scaffolding, updates, update payloads, and report
  helpers.
- `startup/`: startup dependencies, preparation, state, validation, and startup
  orchestration.
