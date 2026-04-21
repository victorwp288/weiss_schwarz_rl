#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_SIM_EXTRA="${USE_SIM_EXTRA:-0}"

log_step() {
  echo
  echo "==> $1"
}

if command -v uv >/dev/null 2>&1; then
  log_step "Sync dependencies"
  if [[ "$USE_SIM_EXTRA" == "1" ]]; then
    uv sync --extra dev --extra sim
    RUN=(uv run --extra dev --extra sim python)
  else
    uv sync --extra dev
    RUN=(uv run --extra dev python)
  fi
else
  if [[ -x ".venv/bin/python" ]]; then
    RUN=(.venv/bin/python)
  else
    python3 -m venv .venv
    RUN=(.venv/bin/python)
  fi

  log_step "Check editable environment"
  "${RUN[@]}" -c "import sys; print(sys.executable)"

  log_step "Install local editable dependencies"
  if [[ "$USE_SIM_EXTRA" == "1" ]]; then
    "${RUN[@]}" -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e ".[dev,sim]"
  else
    "${RUN[@]}" -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e ".[dev]"
  fi
fi

run_python_file() {
  local label="$1"
  shift
  log_step "$label"
  "${RUN[@]}" "$@"
}

run_module() {
  local label="$1"
  shift
  log_step "$label"
  "${RUN[@]}" -m "$@"
}

run_python_file "Cross-platform verify" python/scripts/verify_repo.py

echo
echo "Local CI parity checks completed."
