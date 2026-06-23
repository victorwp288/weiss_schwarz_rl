# Thesis Workflow

This is the operator page. Use `python -m weiss_rl.cli` for thesis work; use
lower-level `weiss_rl.*` modules only for labeled diagnostics and verifiers.

## Setup

```powershell
uv sync --extra dev --extra sim
```

The simulator-backed path expects `weiss-sim>=1.2.0,<2`. Startup checks the
active simulator version, required stepping APIs, spec bundle, and thesis deck
presets.

## Public Commands

| Task | Command |
| --- | --- |
| B1 smoke | `uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_smoke --profile smoke` |
| B1 thesis launch | `uv run --extra dev --extra sim python -m weiss_rl.cli train-b1 --run-label b1_thesis_seed1 --profile thesis-local` |
| Main smoke | `uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_smoke --b1-run runs/b1_smoke --profile smoke` |
| Main thesis launch | `uv run --extra dev --extra sim python -m weiss_rl.cli train-main --run-label main_thesis_seed1 --b1-run runs/b1_thesis_seed1 --profile thesis-local` |
| Smoke eval | `uv run --extra dev --extra sim python -m weiss_rl.cli smoke-eval --run-dir runs/main_smoke --b1-run runs/b1_smoke` |
| Final eval | `uv run --extra dev --extra sim python -m weiss_rl.cli eval-final --run-dir runs/main_thesis_seed1 --b1-run runs/b1_thesis_seed1` |
| Figures | `uv run --extra dev python -m weiss_rl.cli figures --run-dir runs/main_thesis_seed1 --format png --format pdf` |

`train-main --b1-run` resolves the explicit `b1_noleague_baseline` anchor. It
does not use chronological `latest` as a substitute.

Smoke eval runs the fixed B0-B4 policy panel with a tiny seed surface and skips
metagame, figure, and readiness outputs. Final eval uses the selected
`configs/thesis/final_eval.yaml` contract.

## Selected-Run Reproduction

```powershell
$env:PYTHONHASHSEED='0'; uv run --extra dev --extra sim python -m weiss_rl.cli eval-final `
  --run-dir runs/main_champion_hardneg_interp_u10_repair_a015_20260517 `
  --b1-run runs/guarded_recontinue2_from_selected_u15_anchor_nopublic_u20_20260516_seg01
```

Selected runs, hashes, and evidence are recorded in [artifacts.md](artifacts.md).

## Profiles

| Profile | Shape | Use |
| --- | --- | --- |
| `smoke` | CPU, 2 envs, unroll 4, 1 update | Plumbing checks only. |
| `gpu-probe` | CUDA, 32 envs, unroll 16, 2 updates | Local GPU sanity before a longer run. |
| `league-probe` | CUDA, 288 envs, unroll 64, 50 updates | Early-collapse checks before a thesis launch. |
| `thesis-local` | CUDA, 288 envs, unroll 64, 200 updates | Local thesis run shape. |
| `thesis-server` | CUDA, 4096 envs, unroll 64, 200 updates | Server run shape. |

Long runs should use explicit labels, preserved logs, and saved checkpoints. Do
not treat smoke results as model-quality evidence.

## Config Surface

Public thesis configs:

- `configs/thesis/b1_noleague.yaml`
- `configs/thesis/main_league.yaml`
- `configs/thesis/main_league_auto_gpu.yaml`
- `configs/thesis/final_eval.yaml`
- `configs/thesis/final_eval_gpu.yaml`
- `configs/thesis/multideck_exploratory.yaml`
- `configs/thesis/ablations/no_gru.yaml`
- `configs/thesis/ablations/ppo_lite.yaml`
- `configs/thesis/ablations/terminal_only_reward.yaml`

Shared fragments under `configs/thesis/_shared/` and
`configs/thesis/base_fixed_deck_structured.yaml` are not normal launch targets.
Config overrides are for diagnostics; repeated workflows should become named
configs or profiles.

## Model And Deck Policy

Canonical B1 and main league thesis configs use the medium64 structured model:

- `gru_hidden_size: 64`
- `encoder_mlp_width: 64`
- `typed_feature_width: 16`

Deck policy:

- Focal model, B0, B1, and B2: `preset:main_deck_5hy_yotsuba_v1`
- B3 aggro: `preset:aggro_deck_5hy_nino_v1`
- B4 control: `preset:control_deck_jj_s66_v1`
- Multideck results are exploratory and must be labeled separately.

## Diagnostics

B2 disagreement audit:

```powershell
uv run --extra dev --extra sim python -m weiss_rl.cli b2-audit `
  --run-dir runs/main_thesis_seed1 `
  --episodes-jsonl runs/main_thesis_seed1/eval/final_eval/episodes.jsonl `
  --policy-id policy_000200
```

Reward-component probe:

```powershell
uv run --extra dev python -m weiss_rl.diagnostics.reward_component_probe_entrypoint `
  --stack-config configs/thesis/b1_noleague.yaml `
  --num-envs 64 `
  --steps 256 `
  --output-json runs/diagnostics/reward_components/b1_noleague.json
```

Learning-progress diagnostic:

```powershell
uv run --extra dev python -m weiss_rl.diagnostics.learning_progress `
  --run-dir runs/b1_reward_full_shaping_probe100_20260513
```

B1 candidate selection is manual in this checkout: use periodic dev-eval,
targeted confirmation, and snapshot registry artifacts to choose the explicit
`b1_noleague_baseline` source run before launching `train-main`.

## Artifact Contract

A paper-grade run tree must contain the canonical run files, training outputs,
final eval outputs, diagnostics, replay artifacts, metagame outputs, and paper
figure outputs. Smoke/demo output is plumbing evidence, not paper evidence.

`artifact_contract_entrypoint` exercises the synthetic public-demo
train/eval/figure path, then writes and checks the dedicated
`runs/paper_readiness_fixture_ci` readiness fixture.

```powershell
uv run python -m weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint --dry-run
uv run python -m weiss_rl.workflows.artifact_contract.artifact_contract_entrypoint
```

## Validation

```powershell
uv run python -m weiss_rl.workflows.verify_repo_entrypoint
uv run python -m pytest -q tests/weiss_rl
uv run python -m ruff check python tests
uv run python -m ruff format --check python tests
uv run python -m mypy python/weiss_rl/workflows/thesis_wrapper.py python/weiss_rl/workflows/eval_entrypoint.py python/weiss_rl/human_play/play_vs_model_entrypoint.py
```

Simulator boundary check:

```powershell
uv run --extra dev --extra sim python -m pytest -q tests/weiss_rl/test_simulator_contract.py tests/weiss_rl/test_rl_step_supported_layouts.py tests/weiss_rl/test_weiss_sim_contract_surface.py tests/weiss_rl/test_env_pool_buffers_nometa.py tests/weiss_rl/test_decision_boundary_env_layout_equivalence.py tests/weiss_rl/test_heuristic_public_simulator_parity.py
```

Docs/config link and public-config check:

```powershell
uv run python -m pytest -q tests/weiss_rl/test_public_config_surface_docs.py
```

## Troubleshooting

Simulator import or spec failure:

```powershell
uv sync --extra dev --extra sim
uv run --extra dev --extra sim python -c "import weiss_sim; print(weiss_sim.__version__, weiss_sim.__file__, weiss_sim.SPEC_HASH)"
```

Verification failure: run the reported command directly from the repository
root. If a refactor touched a behavior boundary, check [architecture.md](architecture.md)
before treating the failure as incidental.

Artifact or readiness failure: do not use smoke/demo output as a substitute for
a paper-grade run tree. Run the artifact-contract dry run above and compare the
run layout to the paper-grade contract.
