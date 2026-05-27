# Reproducibility

Reproducibility is built from explicit hashes and pinned evaluation protocols.

## Recorded Per Run

- simulator spec hash
- config hash
- run IDs
- git commit and dirty flag
- seed-file hashes
- hardware summary
- runtime mode
- policy-set selection details
- checkpoint tracker state

## Deterministic Surfaces

- `paper_eval_pinned` evaluation behavior
- seed-file parsing and paired seed order
- policy-set ordering and tie-breaks
- payoff folding and uncertainty summaries
- config canonicalization and hash calculation

## Non-Bitwise Surfaces

`train_async_fast` is throughput-oriented. It records provenance and seeds, but host scheduling can affect ordering. Use `train_ordered` for debugging order-sensitive regressions.

## Refactor Rule

If a refactor changes a hash, output ordering, seed use, checkpoint schema, or summary shape, assume behavior changed until a test and log entry prove otherwise.
