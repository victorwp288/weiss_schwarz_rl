# Models Package Map

Use this package for policy/value model implementation code. The stable public
surface remains `weiss_rl.model`; direct owner imports live under the subfolders
below.

## Public Surface

- `weiss_rl.model`: stable facade for model classes, factory helpers, and
  compatibility exports used outside the implementation package.
- `architecture_map.py`: reader-facing map for model component ownership.

## Subpackages

- `policy/`: model classes, model factory, loading, state-dict compatibility,
  policy facade mixins, and opponent-context helpers.
- `backbone/`: base model behavior, recurrent/trunk forwarding, sequence
  helpers, tensor utilities, layers, and state validation.
- `observations/`: typed observation encoder, structured observation contract,
  observation context, and feature gathering.
- `actions/`: structured action tables, action plans, and candidate projection
  or partitioning helpers.
- `heads/`: structured legal-action head assembly, dimensions, setup records,
  blueprint/build-plan helpers, and scoring-surface metadata.
- `scoring/`: dense, packed, factorized, sampled, and structured legal-action
  scoring paths.
- `public_heuristic/`: transparent public-board heuristic features, scoring,
  slot preferences, and logit-bias composition.
