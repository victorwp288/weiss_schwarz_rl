# weiss_schwarz_rl

Thesis RL pipeline scaffold for Weiss Schwarz.

## Scope

- RL training, evaluation, league, metagame analysis, replay tooling, and paper figures.
- Simulator integration is provided by `weiss_sim`; this repo owns the RL pipeline and artifacts.

## Layout

- `python/weiss_rl/`: RL package modules (env wrappers, learner, eval, metagame, replay, plotting).
- `python/scripts/`: runnable entrypoint scripts (`train.py`, `eval.py`, `make_figures.py`).
- `configs/`: locked config stack and committed seed sets.
- `tests/`: top-level test entrypoint notes.
- `runs/`: generated run artifacts (kept out of git except placeholders).

## Setup

### uv (recommended)

```bash
uv sync --extra dev
```

### Editable install with pip extras

```bash
python -m pip install -e ".[dev]"
```

### Torch / CUDA note

This project depends on `torch` without forcing CPU-only builds.
If you want a CUDA wheel, install the matching PyTorch build for your platform after syncing, for example:

```bash
uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

### Pre-commit

```bash
uv run --extra dev pre-commit install
uv run --extra dev pre-commit run -a
```

## Quick checks

```bash
python -c "import weiss_rl; print(weiss_rl.__all__)"
uv run --extra dev pytest -q python/weiss_rl/tests
python -m py_compile $(find python -name '*.py')
```

## Schema validation line
1. uv run python -c "from weiss_rl.trajectory.schema import TRAJ_SCHEMA_VERSION, TrajectoryStep; assert TRAJ_SCHEMA_VERSION==1; _=TrajectoryStep; print('ok')"

2. uv run python -c "import weiss_rl.trajectory.schema as s; print('schema_version', s.TRAJ_SCHEMA_VERSION); print('schema_path', s.__file__)"
