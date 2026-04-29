# Weiss RL Thesis Run Plan - Vast Box

Date: 2026-04-29

## Current Launch Facts

- Remote workspace: `/workspace/weiss_schwarz_rl`
- Hardware: 4x RTX PRO 6000 Blackwell Max-Q, about 98 GB VRAM each; 96 CPU cores; about 566 GiB RAM.
- Required launch prelude: `ulimit -n 1048576`
- Use `--ddp-backend gloo` for repo DDP on this Vast image.
- Avoid `--ddp-backend nccl` for now: standalone NCCL all-reduce works, but the repo workload hits CUDA illegal memory access under NCCL.
- Use `--autoscale --hardware-profile local --torchrun-nproc 4`.

## High-Level Order

1. Benchmark the training envelope.
2. Freeze a canonical model/topology surface.
3. Generate a fresh matching B1 no-league anchor.
4. Mine labels / counterfactual or leak-related artifacts if the frozen config requires them.
5. Train the main league thesis model.
6. Train comparable baselines and ablations with the same model/topology surface.
7. Run final evals, figures, and paper-readiness summaries.

## Benchmark Goals

Find the largest model/topology that is stable and fast enough for the remaining thesis matrix.

Primary knobs:

- Model width / GRU size: start at 512, then adjust down or up if justified.
- Encoder MLP width: match GRU size unless benchmark shows a reason not to.
- Unroll length: start with 64. Consider 32 for faster feedback or 96/128 if throughput improves without instability.
- Env scale: start with `training.scaling.target_envs_per_gpu=512`; test 768/1024 if actors are not saturated and memory/CPU allow it.
- Actor cap: start with 64; test higher only if the runtime allows it and CPU remains underused.
- Batch unrolls: keep comparable across models unless throughput or memory clearly argues otherwise.

Selection rule:

- Prefer the largest model that loses only modest speed versus the current 248 baseline.
- Do not spend the box on a larger model if it substantially reduces update throughput or blocks the ablation matrix.
- Apply the selected model/topology to every comparable run. Main may receive more max updates; baselines/ablations should otherwise match.

## Frozen Envelope Candidate

Benchmark result notes:

- `notes/vast_envelope_20260429_121838_results.md`
- `notes/vast_envelope_pass2_20260429_122509_results.md`

Recommended main-family envelope after short benchmarks:

- GRU hidden size: 512
- Encoder MLP width: 512
- Target envs/GPU: 512
- Max actor process count: 64
- Max envs/actor: 64
- DDP backend: Gloo
- GPU count: 4

Unroll choice:

- Conservative: unroll 64. This is closest to the existing config and was only about 3 percent slower than the 248 model at the same env scale.
- Aggressive: unroll 128. This gave the best raw throughput for width 512, about 58k mean samples/s, with about 45 GB peak VRAM on 98 GB GPUs.

Current recommendation:

- Use width 512 for the fresh B1 anchor, main run, baselines, and ablations.
- After the fixed-512 runtime sweep, the best robust setting is env/GPU 384, unroll 160, batch unrolls 64.
- The fastest raw setting is env/GPU 512, unroll 128, batch unrolls 128, but it uses more memory and changes the effective update batch size more.
- If we want minimum schedule risk, use width 512, env/GPU 512, unroll 64 instead, but this leaves significant throughput on the table.

Avoid:

- Target envs/GPU 768+ for now. It slowed throughput or hit shared-memory failures.
- Max envs/actor 32. It badly reduced throughput.
- NCCL DDP in this repo on this image.
- Batch unrolls 256. It OOMed near 89 GB/GPU in the fixed-512 sweep.

## Comparable Runs

Use the frozen surface for:

- `configs/baselines/noleague_impala.yaml`
- `configs/main_impala_league_server.yaml`
- `configs/ablations/no_tactical_bias.yaml`
- `configs/ablations/teacher_fade.yaml`
- `configs/ablations/no_b1_cutoff.yaml`
- `configs/ablations/reward_shaping.yaml`
- `configs/ablations/multideck.yaml`

Exceptions:

- `configs/baselines/norecurrence_impala.yaml`: keep model width comparable but no recurrent core.
- `configs/baselines/ppo_lite.yaml`: keep model surface comparable where possible, but PPO optimizer/precision may remain PPO-specific.

## Provisional Update Budget

To revise after benchmarking:

- B1 anchor: 200-400 updates.
- Main thesis model: 800-1200+ updates if stable.
- Baselines: 300-500 updates.
- Ablations: 300-500 updates.

If time gets tight, prioritize:

1. Main thesis model.
2. No tactical bias.
3. Teacher fade.
4. No B1 cutoff.
5. Reward shaping / multideck as time permits.
