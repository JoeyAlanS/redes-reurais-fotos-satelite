#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site="$(find "$root/.venv/lib" -mindepth 2 -maxdepth 2 -type d -name site-packages -print -quit)"
if [[ -z "$site" ]]; then
  echo 'Ambiente .venv ausente' >&2
  exit 1
fi
cuda_libs="$(find "$site/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' | sed 's/:$//')"
export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$site/nvidia/cuda_nvcc/bin:$PATH"
exec "$root/.venv/bin/python" "$@"
