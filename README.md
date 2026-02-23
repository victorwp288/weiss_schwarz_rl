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

## Quick checks

```bash
PYTHONPATH=python python3 -m pytest -q python/weiss_rl/tests
python3 -m py_compile $(find python -name '*.py')
```
