# Reproducibility

Reproducibility comes from explicit hashes, pinned seed files, stable policy
sets, and run manifests. It does not require every training mode to be bitwise
identical.

## Recorded Per Run

Canonical runs record:

- simulator spec hash and spec bundle
- canonical config hash
- run ID and run label
- git commit and dirty flag
- seed-file hashes
- hardware summary
- runtime mode
- policy-set selection details
- checkpoint tracker state

## Deterministic Surfaces

These surfaces should remain stable unless a behavior change is intentional:

- `paper_eval_pinned` evaluation behavior
- seed-file parsing and paired seed order
- policy-set ordering and tie-breaks
- payoff folding and uncertainty summaries
- config canonicalization and hash calculation
- artifact paths used by paper-readiness checks

## Throughput-Oriented Surfaces

`train_async_fast` records provenance and seeds, but host scheduling can affect
collection order. Use `train_ordered` for order-sensitive debugging and
regression isolation.

## Refactor Rule

If a refactor changes a hash, output order, seed use, checkpoint schema, run
manifest field, or summary shape, assume behavior changed until a test and log
entry prove otherwise.
