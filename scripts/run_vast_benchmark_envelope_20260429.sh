#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ulimit -n 1048576
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1

GROUP_LABEL="${BENCH_GROUP_LABEL:-vast_envelope_20260429_$(date +%H%M%S)}"
BENCH_UPDATES="${BENCH_UPDATES:-30}"
BENCH_MINUTES="${BENCH_MINUTES:-12}"
SEED="${SEED:-20260429}"
RESULTS_PATH="notes/${GROUP_LABEL}_results.md"

mkdir -p notes
cat > "$RESULTS_PATH" <<EOF
# Vast Benchmark Envelope - ${GROUP_LABEL}

Date: 2026-04-29

Launch:

- ulimit: $(ulimit -n)
- CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}
- torchrun nproc: 4
- DDP backend: gloo
- Base config: configs/baselines/noleague_impala.yaml
- Updates per case: ${BENCH_UPDATES}
- Wall-clock cap per case: ${BENCH_MINUTES} minutes
- Periodic dev eval disabled for raw throughput comparison.
- Checkpoint/snapshot intervals raised to avoid benchmark overhead.

| Label | Exit | Width | Target envs/GPU | Unroll | Actor cap | Records | Last update | Mean samples/s | Max samples/s | Mean updates/s | Max GPU mem MB | Max GPU util % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
EOF

run_case() {
  local width="$1"
  local target_envs_per_gpu="$2"
  local unroll_length="$3"
  local actor_cap="$4"
  local label="${GROUP_LABEL}_w${width}_e${target_envs_per_gpu}_u${unroll_length}_a${actor_cap}"
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
    --override "model.gru_hidden_size=${width}" \
    --override "model.encoder_mlp_width=${width}" \
    --override "training.scaling.target_envs_per_gpu=${target_envs_per_gpu}" \
    --override "training.scaling.max_actor_process_count=${actor_cap}" \
    --override "evaluation.periodic_dev_eval_interval_updates=0" \
    --override "training.checkpointing.checkpoint_interval_updates=100000" \
    --override "training.checkpointing.snapshot_interval_updates=100000"
  exit_code="$?"
  set -e

  .venv/bin/python - "$RESULTS_PATH" "$label" "$exit_code" "$width" "$target_envs_per_gpu" "$unroll_length" "$actor_cap" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
label = sys.argv[2]
exit_code = int(sys.argv[3])
width = sys.argv[4]
target_envs = sys.argv[5]
unroll = sys.argv[6]
actor_cap = sys.argv[7]
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

row = (
    f"| `{label}` | {exit_code} | {width} | {target_envs} | {unroll} | {actor_cap} | "
    f"{training.get('record_count', '')} | {last_update} | "
    f"{training.get('throughput_samples_per_sec', {}).get('mean', '')} | "
    f"{training.get('throughput_samples_per_sec', {}).get('max', '')} | "
    f"{training.get('throughput_updates_per_sec', {}).get('mean', '')} | "
    f"{telemetry.get('gpu_mem_used_mb', {}).get('max', '')} | "
    f"{telemetry.get('gpu_util', {}).get('max', '')} |\n"
)
with results_path.open("a", encoding="utf-8") as handle:
    handle.write(row)
PY
}

# Start with the requested large model, then test env scale, then add smaller references.
run_case 512 512 64 64
run_case 512 768 64 64
run_case 512 1024 64 64
run_case 384 768 64 64
run_case 248 512 64 64

echo "Benchmark envelope complete: ${RESULTS_PATH}"
