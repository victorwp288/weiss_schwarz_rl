# Vast Runtime Geometry Benchmark - vast_runtime_geometry_20260429_123740

Date: 2026-04-29

Launch:

- ulimit: 1048576
- CUDA_VISIBLE_DEVICES: 0,1,2,3
- torchrun nproc: 4
- DDP backend: gloo
- Base config: configs/baselines/noleague_impala.yaml
- Fixed model width / GRU: 512
- Updates per case: 50
- Wall-clock cap per case: 15 minutes
- Periodic dev eval disabled for raw throughput comparison.
- Checkpoint/snapshot intervals raised to avoid benchmark overhead.

| Label | Exit | Env/GPU | Unroll | Max env/actor | Batch unrolls | Records | Last update | Mean samples/s | Max samples/s | Mean updates/s | Mean GPU util % | Max GPU util % | Mean VRAM MB | Max VRAM MB | Mean CPU % | Max CPU % | Max procs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vast_runtime_geometry_20260429_123740_e256_u128_m64_b64` | 1 | 256 | 128 | 64 | 64 | 0 |  |  |  |  | 0.00 | 0.00 | 5.00 | 5.00 | 0.00 | 0.00 | 1.00 |
| `vast_runtime_geometry_20260429_123740_e384_u96_m64_b64` | 0 | 384 | 96 | 64 | 64 | 50 | 50 | 63851.00 | 83252.28 | 0.62 | 27.38 | 99.00 | 22457.88 | 35470.00 | 1613.27 | 2635.62 | 33.00 |
| `vast_runtime_geometry_20260429_123740_e384_u128_m64_b64` | 0 | 384 | 128 | 64 | 64 | 50 | 50 | 76870.22 | 98139.13 | 0.56 | 14.67 | 73.00 | 28751.89 | 43084.00 | 1942.31 | 3017.66 | 33.00 |
| `vast_runtime_geometry_20260429_123740_e384_u160_m64_b64` | 0 | 384 | 160 | 64 | 64 | 50 | 50 | 88039.41 | 110076.13 | 0.51 | 46.90 | 79.00 | 34711.50 | 50138.00 | 2167.06 | 3197.39 | 33.00 |
| `vast_runtime_geometry_20260429_123740_e512_u160_m64_b64` | 0 | 512 | 160 | 64 | 64 | 50 | 50 | 79965.26 | 102941.57 | 0.46 | 15.64 | 67.00 | 36390.36 | 51893.00 | 1951.68 | 3217.38 | 41.00 |
| `vast_runtime_geometry_20260429_123740_e512_u192_m64_b64` | 0 | 512 | 192 | 64 | 64 | 50 | 50 | 88547.28 | 112381.71 | 0.43 | 51.91 | 100.00 | 43922.64 | 63025.00 | 2307.81 | 3349.37 | 41.00 |
| `vast_runtime_geometry_20260429_123740_e640_u128_m64_b64` | 0 | 640 | 128 | 64 | 64 | 50 | 50 | 64628.48 | 87606.15 | 0.47 | 47.00 | 100.00 | 28427.60 | 46208.00 | 1877.41 | 3001.06 | 49.00 |
| `vast_runtime_geometry_20260429_123740_e512_u128_m64_b128` | 0 | 512 | 128 | 64 | 128 | 50 | 50 | 109440.31 | 136795.99 | 0.40 | 34.75 | 100.00 | 54845.50 | 75421.00 | 2884.08 | 4239.56 | 41.00 |
| `vast_runtime_geometry_20260429_123740_e512_u128_m64_b256` | 1 | 512 | 128 | 64 | 256 | 0 |  |  |  |  | 0.00 | 0.00 | 2243.25 | 6589.00 | 168.09 | 433.02 | 41.00 |
