# Simulator Compatibility

This repo's simulator-backed training and evaluation paths target the published
`weiss-sim==1.1.0` package, or a sibling simulator checkout that exposes the
same contract.

## Phase 0 Contract

The RL repo expects these stable simulator fields:

| Surface | Expected value or method |
| --- | --- |
| Package version | semver-style `weiss_sim.__version__ >= "1.1.0"` |
| Observation length | `weiss_sim.OBS_LEN == 378` |
| Action space | `weiss_sim.ACTION_SPACE_SIZE == 527` |
| Compatibility hash | `weiss_sim.SPEC_HASH == 8590000130` |
| Spec export | `weiss_sim.export_spec_bundle()` |
| High-level envs | `weiss_sim.fast(...)`, `weiss_sim.inspect(...)` |
| Low-level pool | `weiss_sim.make_pool(...)`, `weiss_sim.EnvPoolBuffers` |
| RL reset/step | `weiss_sim.rl.reset_rl(...)`, `weiss_sim.rl.step_rl(...)` |
| Fused sampling | `weiss_sim.rl.step_rl_sample_from_logits_with_logp(...)` |
| Buffer sampling | `EnvPoolBuffers.step_sample_from_logits_with_logp(...)` |
| Dynamic legal context | `EnvPoolBuffers.legal_action_context_v1(...)` |

The canonical hot layouts are:

| Layout | Use |
| --- | --- |
| `mask` | Debug and compatibility paths that need dense legal masks. |
| `nomask` | Compatibility surface for fetching packed ids through buffers. |
| `i16_legal_ids` | Current RL packed-id path with action metadata. |
| `i16_legal_ids_nometa` | Simulator 1.1 low-level hot path when metadata is not consumed. |

## Published Presets

`weiss-sim==1.1.0` publishes these bundled deck preset names:

- `starter_deck_ws02_v1`
- `main_deck_5hy_yotsuba_v1`
- `aggro_deck_5hy_nino_v1`
- `control_deck_jj_s66_v1`

Use them with the `preset:` prefix in RL configs, for example
`preset:main_deck_5hy_yotsuba_v1`.

## RL Deck Decision

The canonical `standard` and `standard-thesis-eval` surfaces train and evaluate
the focal model, B0, B1, and B2 on `preset:main_deck_5hy_yotsuba_v1`.

The final-eval scheduler assigns profile decks only to the themed public
heuristics:

- `B3 HeuristicPublicAggro` uses `preset:aggro_deck_5hy_nino_v1`
- `B4 HeuristicPublicControl` uses `preset:control_deck_jj_s66_v1`

This keeps the main comparison same-deck while preserving aggro/control rows as
explicit robustness checks.

## Validation Commands

```powershell
uv sync --extra dev --extra sim
uv run --extra dev --extra sim python -c "import weiss_sim; print(weiss_sim.__version__, weiss_sim.__file__, weiss_sim.SPEC_HASH)"
uv run --extra dev --extra sim python -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_training_startup.py python/weiss_rl/tests/test_pool_factory.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py
```

For sibling-checkout parity, point at an importable simulator build. A raw source
tree is not enough unless the Rust extension is also built and importable.

```powershell
$env:WEISS_SIM_PYTHONPATH="C:\Users\Bruger\Desktop\this one\weiss-schwarz-simulator\python"
uv run --extra dev python -m pytest -q python/weiss_rl/tests/test_simulator_contract.py python/weiss_rl/tests/test_rl_step_layout_contract_smoke.py
```

The sibling checkout should be treated as compatible only when these same tests
pass and the recorded spec bundle matches the expected hash.
