# Training

Use the package CLI for thesis work:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke
uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke
```

The canonical configs are `configs/thesis/b1_noleague.yaml` and
`configs/thesis/main_league.yaml`. `configs/thesis/main_league_auto_gpu.yaml`
is the server-oriented main lane.

Compatibility scripts remain under `python/scripts/`, but new work should start
from `python -m weiss_rl.cli`.

Canonical runs write checkpoints, snapshot registries, training metrics, eval
outputs, replays, and figure artifacts under `runs/<run_label>/`.
