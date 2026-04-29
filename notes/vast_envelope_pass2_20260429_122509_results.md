# Vast Benchmark Envelope Pass 2 - vast_envelope_pass2_20260429_122509

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

| Label | Exit | Width | Target envs/GPU | Unroll | Actor cap | Max envs/actor | Batch unrolls | Records | Last update | Mean samples/s | Max samples/s | Mean updates/s | Max GPU mem MB | Max GPU util % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vast_envelope_pass2_20260429_122509_w512_e384_u64_a64_m64_b64` | 0 | 512 | 384 | 64 | 64 | 64 | 64 | 30 | 30 | 39482.51070088619 | 55749.67534215966 | 0.559562237759975 | 28314.0 | 95.0 |
| `vast_envelope_pass2_20260429_122509_w512_e512_u96_a64_m64_b64` | 0 | 512 | 512 | 96 | 64 | 64 | 64 | 30 | 30 | 47360.8870512995 | 66857.09157467347 | 0.4475182742386051 | 40189.0 | 99.0 |
| `vast_envelope_pass2_20260429_122509_w512_e512_u128_a64_m64_b64` | 0 | 512 | 512 | 128 | 64 | 64 | 64 | 30 | 30 | 58133.5394083881 | 80295.99323510379 | 0.41143897246450756 | 44743.0 | 99.0 |
| `vast_envelope_pass2_20260429_122509_w512_e512_u64_a64_m32_b64` | 0 | 512 | 512 | 64 | 64 | 32 | 64 | 30 | 30 | 12121.893657938657 | 19324.92643358816 | 0.34549221453302364 | 23938.0 | 12.0 |
| `vast_envelope_pass2_20260429_122509_w640_e512_u64_a64_m64_b64` | 1 | 640 | 512 | 64 | 64 | 64 | 64 | 0 |  |  |  |  | 5.0 | 0.0 |
| `vast_envelope_pass2_20260429_122509_w768_e512_u64_a64_m64_b64` | 0 | 768 | 512 | 64 | 64 | 64 | 64 | 30 | 30 | 32768.07258208744 | 47023.1194124979 | 0.4649207584888353 | 38431.0 | 48.0 |
