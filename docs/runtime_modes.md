# Runtime Modes

This repo uses explicit modes so determinism and throughput do not get conflated.

## `paper_eval_pinned`

Use this for evaluation, selection, metagame analysis, figures, and readiness.

- fixed committed seeds
- pinned evaluation RNG
- stable policy ordering and tie-breaks
- CPU evaluation by default
- canonical artifact writing only

This is the mode that should produce the exact thesis-facing outputs.

## `train_ordered`

Use this when you want reproducible thesis runs with deterministic merge order.

- queue-based actor/learner runtime
- barriered policy sync
- unrolls merged in stable order
- useful for debugging regressions and reproducing paper claims

This mode favors reproducibility over raw throughput.

## `train_async_fast`

Use this when you want the same runtime shape but with higher throughput.

- fixed seeds and full provenance
- queue-based collection
- policy lag and queue lag are recorded
- update ordering may vary with scheduling

This mode is performance-oriented, but it is not intended to be bitwise identical across arbitrary host scheduling.

## `public_demo`

Use this for public-safe CI/demo artifacts.

- synthetic, labeled demo outputs
- does not claim thesis-grade training or proprietary evaluation
- useful for artifact checks and docs examples

## Simulator installation modes

- `uv sync --extra dev` uses the repo's regular dev dependencies.
- `uv sync --extra dev --extra sim` adds the published `weiss-sim` package.
- If you prefer a sibling checkout, point `WEISS_SIM_PYTHONPATH` at the simulator source tree.

## Practical rule

When the result needs to be thesis-grade, use `paper_eval_pinned` and canonical artifacts. When the result needs to be fast, use `train_async_fast` but keep provenance and seeds fixed.
