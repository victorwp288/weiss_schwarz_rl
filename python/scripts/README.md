Entry-point scripts for the thesis RL pipeline.

Expected future scripts:
- train.py
- eval.py
- make_figures.py

## Quick checks

Smoke test (<2 minutes on CPU):

```bash
python python/scripts/train.py --stack-config configs/minimal_loop.yaml --run-id smoke_local