#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ulimit -n 1048576
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1

GROUP_LABEL="${BENCH_GROUP_LABEL:-vast_runtime_geometry_20260429_$(date +%H%M%S)}"
BENCH_UPDATES="${BENCH_UPDATES:-50}"
BENCH_MINUTES="${BENCH_MINUTES:-15}"
SEED="${SEED:-20260429}"
RESULTS_PATH="notes/${GROUP_LABEL}_results.md"

mkdir -p notes
cat > "$RESULTS_PATH" <<EOF
# Vast Runtime Geometry Benchmark - ${GROUP_LABEL}

Date: 2026-04-29

Launch:

- ulimit: $(ulimit -n)
- CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}
- torchrun nproc: 4
- DDP backend: gloo
- Base config: configs/baselines/noleague_impala.yaml
- Fixed model width / GRU: 512
- Updates per case: ${BENCH_UPDATES}
- Wall-clock cap per case: ${BENCH_MINUTES} minutes
- Periodic dev eval disabled for raw throughput comparison.
- Checkpoint/snapshot intervals raised to avoid benchmark overhead.

| Label | Exit | Env/GPU | Unroll | Max env/actor | Batch unrolls | Records | Last update | Mean samples/s | Max samples/s | Mean updates/s | Mean GPU util % | Max GPU util % | Mean VRAM MB | Max VRAM MB | Mean CPU % | Max CPU % | Max procs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
EOF

run_case() {
  local target_envs_per_gpu="$1"
  local unroll_length="$2"
  local max_envs_per_actor="$3"
  local batch_unrolls="$4"
  local label="${GROUP_LABEL}_e${target_envs_per_gpu}_u${unroll_length}_m${max_envs_per_actor}_b${batch_unrolls}"
  local exit_code=0

  echo "==> Running ${label}"
  set +e
  .venv/bin/python python/scripts/profile_train_job.py \
    --run-label "$label" \
    --stack-config configs/baselines/noleague_impala.yaml \
    --seed "$SEED" \
    --runtime-mode train_async_fast \
    --unroll-length "$unroll_length" \
    --max-updates "$BENCH_UPDATES" \
    --max-wall-clock-minutes "$BENCH_MINUTES" \
    --autoscale \
    --hardware-profile local \
    --torchrun-nproc 4 \
    --ddp-backend gloo \
    --ddp-timeout-seconds 1800 \
    --sample-interval-seconds 10 \
    --override "model.gru_hidden_size=512" \
    --override "model.encoder_mlp_width=512" \
    --override "training.scaling.target_envs_per_gpu=${target_envs_per_gpu}" \
    --override "training.scaling.max_actor_process_count=64" \
    --override "training.scaling.max_envs_per_actor=${max_envs_per_actor}" \
    --override "training.rollout.batch_unrolls_per_update=${batch_unrolls}" \
    --override "evaluation.periodic_dev_eval_interval_updates=0" \
    --override "training.checkpointing.checkpoint_interval_updates=100000" \
    --override "training.checkpointing.snapshot_interval_updates=100000"
  exit_code="$?"
  set -e

  .venv/bin/python - "$RESULTS_PATH" "$label" "$exit_code" "$target_envs_per_gpu" "$unroll_length" "$max_envs_per_actor" "$batch_unrolls" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
label = sys.argv[2]
exit_code = int(sys.argv[3])
target_envs = sys.argv[4]
unroll = sys.argv[5]
max_envs_per_actor = sys.argv[6]
batch_unrolls = sys.argv[7]
run_dir = Path("runs") / label
summary_path = run_dir / "job_telemetry_summary.json"
metrics_path = run_dir / "training" / "logs" / "training_metrics.jsonl"

summary = {}
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
training = summary.get("training_metrics", {}) if isinstance(summary, dict) else {}
telemetry = summary.get("telemetry", {}) if isinstance(summary, dict) else {}

last_update = ""
if metrics_path.exists():
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines:
        last_update = str(json.loads(lines[-1]).get("update_count", ""))

def stat(group: str, name: str) -> str:
    body = telemetry.get(group, {})
    if not isinstance(body, dict):
        return ""
    value = body.get(name, "")
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

def train_stat(group: str, name: str) -> str:
    body = training.get(group, {})
    if not isinstance(body, dict):
        return ""
    value = body.get(name, "")
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

row = (
    f"| `{label}` | {exit_code} | {target_envs} | {unroll} | {max_envs_per_actor} | {batch_unrolls} | "
    f"{training.get('record_count', '')} | {last_update} | "
    f"{train_stat('throughput_samples_per_sec', 'mean')} | "
    f"{train_stat('throughput_samples_per_sec', 'max')} | "
    f"{train_stat('throughput_updates_per_sec', 'mean')} | "
    f"{stat('gpu_util', 'mean')} | {stat('gpu_util', 'max')} | "
    f"{stat('gpu_mem_used_mb', 'mean')} | {stat('gpu_mem_used_mb', 'max')} | "
    f"{stat('cpu_percent_total', 'mean')} | {stat('cpu_percent_total', 'max')} | "
    f"{stat('process_count', 'max')} |\n"
)
with results_path.open("a", encoding="utf-8") as handle:
    handle.write(row)
PY
}

# Fixed width 512. Explore runtime geometry around the best known point.
run_case 256 128 64 64
run_case 384 96 64 64
run_case 384 128 64 64
run_case 384 160 64 64
run_case 512 160 64 64
run_case 512 192 64 64
run_case 640 128 64 64
run_case 512 128 64 128
run_case 512 128 64 256

echo "Runtime geometry benchmark complete: ${RESULTS_PATH}"
