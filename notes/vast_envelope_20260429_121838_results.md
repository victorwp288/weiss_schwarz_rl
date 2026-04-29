# Vast Benchmark Envelope - vast_envelope_20260429_121838

Date: 2026-04-29

Launch:

- ulimit: 1048576
- CUDA_VISIBLE_DEVICES: 0,1,2,3
- torchrun nproc: 4
- DDP backend: gloo
- Base config: configs/baselines/noleague_impala.yaml
- Updates per case: 30
- Wall-clock cap per case: 12 minutes
- Periodic dev eval disabled for raw throughput comparison.
- Checkpoint/snapshot intervals raised to avoid benchmark overhead.

| Label | Exit | Width | Target envs/GPU | Unroll | Actor cap | Records | Last update | Mean samples/s | Max samples/s | Mean updates/s | Max GPU mem MB | Max GPU util % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vast_envelope_20260429_121838_w512_e512_u64_a64` | 0 | 512 | 512 | 64 | 64 | 30 | 30 | 34303.01654631865 | 49593.855522771955 | 0.4868365747441929 | 28803.0 | 41.0 |
| `vast_envelope_20260429_121838_w512_e768_u64_a64` | 0 | 512 | 768 | 64 | 64 | 30 | 30 | 27363.720216822487 | 41363.94224322001 | 0.38911622743594526 | 31742.0 | 23.0 |
| `vast_envelope_20260429_121838_w512_e1024_u64_a64` | 1 | 512 | 1024 | 64 | 64 | 0 |  |  |  |  | 5.0 | 0.0 |
| `vast_envelope_20260429_121838_w384_e768_u64_a64` | 0 | 384 | 768 | 64 | 64 | 30 | 30 | 27959.148082739554 | 42342.8666012397 | 0.3976682501446301 | 26018.0 | 30.0 |
| `vast_envelope_20260429_121838_w248_e512_u64_a64` | 0 | 248 | 512 | 64 | 64 | 30 | 30 | 35486.3636519761 | 51760.52669774099 | 0.5038278068842095 | 20139.0 | 77.0 |
