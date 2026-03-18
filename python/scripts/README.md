Entry-point scripts for the thesis RL pipeline.

Expected future scripts:
- train.py
- eval.py
- make_figures.py

## Quick checks

Manifest smoke test (<2 minutes on CPU):

```bash
python python/scripts/train.py --stack-config configs/stack_smoke.yaml --run-label smoke_local
```

`--run-label` controls the human-friendly run directory name only. The immutable computed run identity still comes from the runtime spec/config/git/nonce inputs and is reported separately in the startup banner and manifest. `--run-id` remains accepted as a deprecated compatibility alias for the label override.

This writes a scaffold manifest with the real simulator `export_spec_bundle()` payload and provenance. If `weiss_sim` is not installed in the active interpreter, the script falls back to a working local simulator interpreter/check-out when available. Configure `WEISS_SIM_PYTHONPATH` (and optionally `WEISS_SIM_PYTHON`) if your simulator lives elsewhere.
