# Diagnostics Package Map

Use this package for repo diagnostics, training-progress reports, hygiene
checks, probes, profiling entrypoints, and telemetry helpers.

## Subpackages

- `b2_audit/`: B2 disagreement audits, aggregation, source loading, reporting,
  and summary math.
- `progress/`: learning-progress metrics, sections, warnings, sync, artifact
  writing, and teacher-guidance reporting.
- `trajectory/`: trajectory audit comparison and policy-drift diagnostics.
- `hygiene/`: artifact hygiene, repo hygiene, artifact scans, and core
  placeholder checks.
- `profiling/`: profile fixtures and structured/train-job profiling entrypoints.
- `probes/`: action diagnostics, checkpoint-family bias, heuristic sanity, and
  reward-component probes.
- `logging/`: CLI banners, job telemetry, tensorboard logging, and training
  logger helpers.
