#!/usr/bin/env bash
# Activate only this demo's isolated environment; never modifies OpenPI's environment.
set -euo pipefail
LIBERO_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$LIBERO_DEMO_ROOT/artifacts/astra_libero/numba_cache"
export LIBERO_CONFIG_PATH="$LIBERO_DEMO_ROOT/artifacts/astra_libero/libero_config"
export LD_LIBRARY_PATH="$LIBERO_DEMO_ROOT/.venv-libero/lib/native${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LP_NUM_THREADS=4
export TMPDIR=/dev/shm
exec "$LIBERO_DEMO_ROOT/.venv-libero/bin/python" "$@"
