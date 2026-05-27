# Performance

Performance work is welcome only when it preserves behavior and is benchmarked.

## Scope Claims Precisely

Always state:

- runtime mode: `train_ordered` or `train_async_fast`
- league enabled or no-league
- simulator profile and legality layout
- device, env count, unroll length, batch sizing, and update count
- whether eval paths inherit the change

## Cheap Profiling

```powershell
uv run python python/scripts/profile_structured_hotpaths.py --mode structured --time-steps 8 --batch-size 8 --iterations 3
uv run python python/scripts/profile_structured_hotpaths.py --mode heuristic --rows 256 --iterations 3
```

For training throughput, use `python/scripts/profile_train_job.py` with an explicit preset and runtime shape.

## Refactor Rule

Do not trade determinism or readability for a theoretical speedup. Add tests first, benchmark the exact mode, and record the result with the change or generated artifact bundle.
