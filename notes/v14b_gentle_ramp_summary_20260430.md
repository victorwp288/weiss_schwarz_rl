# v14b Gentle Ramp Summary - 2026-04-30

Run: `runs/thesis_main_candidate_v14b_u160_gentle_ramp_20260430`
Config: `configs/main_impala_league_server_v14b_u160_gentle_ramp.yaml`
Start checkpoint: `runs/thesis_main_candidate_v14_b1init_anchor_stabilize_20260430/training/checkpoints/checkpoint_160.pt`

## Purpose

Test whether the v14 u160 checkpoint could be gently improved without falling out of the B1-safe basin.

## Launch

```bash
cd /workspace/weiss_schwarz_rl
ulimit -n 1048576
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

RUN_LABEL=thesis_main_candidate_v14b_u160_gentle_ramp_20260430 \
MAX_UPDATES=320 \
STACK_CONFIG=configs/main_impala_league_server_v14b_u160_gentle_ramp.yaml \
RESUME_FROM=runs/thesis_main_candidate_v14_b1init_anchor_stabilize_20260430/training/checkpoints/checkpoint_160.pt \
RESUME_RESET_OPTIMIZER=1 \
scripts/run_thesis_main_v13_b1init_long_variant_20260430.sh
```

## Changes vs v14

- LR: `7.5e-6 -> 5e-6`
- RL pressure: `policy_loss_coef: 0.50 -> 0.35`
- B1 baseline mix: `0.45 -> 0.42`
- base public heuristic mix: `0.03 -> 0.04`
- heuristic public variant mix: `0.08 -> 0.12`
- unchanged: actor bias follows learner, exact B1 BC `0.25`, family B1 BC `0.40`, CF `0.10`, damage reward `0.0`, no champions/hard negatives/recents.

## Runtime Sanity

- u162: actor bias active `2.0`, family BC active `0.40`, B1 mix `0.42`, variant mix `0.12`, `vtrace_rho_p99` about `132`.
- u240: `vtrace_rho_p99` about `86`, exact BC loss about `0.489`, family BC loss about `4.013`.
- u320: `vtrace_rho_p99` about `14`, exact BC loss about `0.489`, family BC loss about `3.736`.

## Results

- Manual confirm64 B1-only at u180: B1 `0.4921875`.
- Automatic 16-pair dev eval at u240: B1 `0.500`, B3 `0.750`, B4 `0.96875`, aggregate `0.6958333333333333`.
- Manual confirm64 B1-only at u240: B1 `0.484375`.
- Manual confirm64 B1-only at u320: B1 `0.4140625`.

## Interpretation

v14b did not produce a better main candidate than v14 u160. The off-policy instability stayed fixed, but B1 preservation still decayed under even a gentle RL/variant continuation. Do not use u320. u240 is also below the B1 safety line on confirm64.

Best current main candidate remains:

`runs/thesis_main_candidate_v14_b1init_anchor_stabilize_20260430/training/checkpoints/checkpoint_160.pt`

Best overall thesis anchor remains:

`runs/thesis_b1_candidate_v5_aggressive_bc015_lowlr_20260429/training/checkpoints/checkpoint_120.pt`
