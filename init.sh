#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python_bin="${PYTHON_BIN:-python3}"
uv_bin="${UV_BIN:-}"
bootstrap_dir="${UV_BOOTSTRAP_DIR:-.uv-bootstrap}"

log_step() {
  echo
  echo "==> $1"
}

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "ERROR: Python executable '$python_bin' was not found." >&2
  exit 127
fi

log_step "Python version"
"$python_bin" - <<'PY'
import sys

major, minor = sys.version_info[:2]
print(f"python_executable={sys.executable}")
print(f"python_version={sys.version.split()[0]}")
if (major, minor) < (3, 10) or (major, minor) >= (3, 13):
    raise SystemExit("ERROR: Python 3.10, 3.11, or 3.12 is required for this repo.")
PY

if [[ -n "$uv_bin" ]]; then
  if [[ ! -x "$uv_bin" ]]; then
    echo "ERROR: UV_BIN was set but is not executable: $uv_bin" >&2
    exit 127
  fi
else
  if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
  else
    log_step "Bootstrap local uv"
    "$python_bin" -m venv "$bootstrap_dir"
    bootstrap_python="$bootstrap_dir/bin/python"
    if [[ ! -x "$bootstrap_python" ]]; then
      echo "ERROR: Failed to create bootstrap Python environment at $bootstrap_dir" >&2
      exit 1
    fi
    "$bootstrap_python" -m pip install --upgrade pip uv
    uv_bin="$bootstrap_dir/bin/uv"
  fi
fi

log_step "uv executable"
echo "uv_bin=$uv_bin"
if ! "$uv_bin" --version; then
  echo "ERROR: Unable to execute uv from $uv_bin" >&2
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu124}"

log_step "Sync RL dependencies and published simulator package"
"$uv_bin" sync --extra dev --extra sim

RUN=("$uv_bin" run --extra dev --extra sim python)

log_step "Environment summary"
"${RUN[@]}" - <<'PY'
from __future__ import annotations

import pathlib
import re
import sys
import tomllib

import torch
import weiss_sim

platform_name = sys.platform
data = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
sim_entries = data["project"]["optional-dependencies"].get("sim", [])
expected_spec = None
for entry in sim_entries:
    match = re.match(r"(weiss-sim[<>=!~].+)", entry)
    if match:
        expected_spec = match.group(1)
        break

print(f"torch_version={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"weiss_sim_version={weiss_sim.__version__}")
print(f"weiss_sim_location={weiss_sim.__file__}")
if expected_spec is not None:
    print(f"expected_sim_dependency={expected_spec}")

if platform_name in {"linux", "win32"} and torch.version.cuda is None:
    raise SystemExit(
        "ERROR: init.sh expected a CUDA-enabled torch build on Linux/Windows, but the installed torch is CPU-only."
    )
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  log_step "GPU summary"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
fi

echo
echo "Environment setup complete."
