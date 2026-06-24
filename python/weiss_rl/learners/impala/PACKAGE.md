# IMPALA Learner Package Map

Use this package for IMPALA learner update mechanics. The stable public surface
is `weiss_rl.learners.impala.ImpalaLearner`; direct owner imports live under the
subfolders below.

## Public Surface

- `learner.py`: learner state and update entrypoint.
- `__init__.py`: public `ImpalaLearner` export.

## Subpackages

- `batching/`: learner batch access, field validation, loss-batch inputs, and
  packed auxiliary batch views.
- `losses/`: loss assembly, loss-input preparation, objective stages, policy
  anchor/teacher/V-trace stages, metrics, and loss-plan metadata.
- `auxiliary/`: structured-teacher, paired-outcome, and paired-swing auxiliary
  loss inputs, candidates, outputs, and request helpers.
- `updates/`: normal and scoped optimizer updates, auxiliary update dispatch,
  optimizer steps, update bookkeeping, logging, and training-input validation.
- `support/`: learner support mixins for forward passes, faults, optimizer
  context, public-heuristic support, policy anchors, metrics, and structured
  summaries.
