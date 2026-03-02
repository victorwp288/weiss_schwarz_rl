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

## Quick checks

```bash
python -c "import weiss_rl; print(weiss_rl.__all__)"
uv run --extra dev pytest -q python/weiss_rl/tests
python -m py_compile $(find python -name '*.py')
```
