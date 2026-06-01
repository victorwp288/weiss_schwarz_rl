# League

League behavior controls opponent exposure, promotion, and snapshot selection. It is part of training semantics.

## Core Concepts

- `SnapshotRegistry`: persistent snapshot metadata and stable lookup.
- PFSP: prioritized fictitious self-play sampling over eligible snapshots.
- Promotion gate: deterministic evaluation gate used to decide whether a checkpoint becomes champion/best.
- Baseline anchors: fixed references such as `B0 RandomLegal`, `B1 NoLeague baseline`, and heuristic public policies.
- Imported snapshots: snapshots copied or resolved from another completed run.

Pure promotion-anchor resolution helpers live in `weiss_rl.training.promotion`. They cover canonical B0/B1 names, legacy B1 alias fallback, B2 heuristic anchors, and symbolic latest/previous champion or recent snapshot labels. The simulator-backed promotion gate runner lives in `weiss_rl.training.promotion_gate_runner` and is wired through the training compatibility hooks.

## Safety Rules

- Preserve opponent IDs and display names.
- Preserve snapshot path resolution relative to the source registry.
- Preserve promotion thresholds, truncation guards, confidence gates, and payoff aggregation.
- Preserve fixed baseline and heuristic behavior unless a bug is proven.

## Tests to Run After League Changes

```powershell
uv run python -m pytest -q python/weiss_rl/tests/test_snapshot_registry.py
uv run python -m pytest -q python/weiss_rl/tests/test_promotion_gate.py
uv run python -m pytest -q python/weiss_rl/tests/test_opponent_pool.py
uv run python -m pytest -q python/weiss_rl/tests/test_heuristic_public.py
```
