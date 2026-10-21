#!/bin/bash
# Single-GPU job: sbatch [--job-name=X --time=HH:MM:SS --gres=gpu:a100:1] slurm/run.sbatch <command...>
# Runs <command...> through the NFS uv env after a GPU gate. Build the env on the login node first:
#   UV_PYTHON_INSTALL_DIR=$NFS/.uv/python UV_CACHE_DIR=$NFS/.uv/cache $NFS/bin/uv sync --python 3.12
#
#SBATCH --job-name=sdsb
#SBATCH --output=/mnt/data0/shared/colin/Projects/SchrodingerDistillation/slurm/logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=express
#SBATCH --exclude=192-222-56-117,192-222-56-240,192-222-51-248,bacon
set -euo pipefail
NFS=/mnt/data0/shared/colin
PROJECT="$NFS/Projects/SchrodingerDistillation"
export PATH="$NFS/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$NFS/.uv/python"
export UV_CACHE_DIR="$NFS/.uv/cache"
export MPLCONFIGDIR="$NFS/.mpl"; mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg
export TMPDIR="$PROJECT/.scratch/tmp"; mkdir -p "$TMPDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
test -x "$NFS/bin/uv" || { echo "ERROR: $NFS/bin/uv missing"; exit 1; }
cd "$PROJECT"
echo "=== node ==="
echo "host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "=== GPU gate ==="
uv run --no-sync python - <<'PY'
import sys, jax
devices, backend = jax.devices(), jax.default_backend()
print(f"[gpu_check] jax {jax.__version__} backend={backend} devices={devices}", flush=True)
if backend != "gpu":
    print("[gpu_check] FAIL: jax is on CPU", file=sys.stderr); sys.exit(1)
print(f"[gpu_check] PASS -> {devices[0].device_kind}", flush=True)
PY
echo "=== run: $* ==="
uv run --no-sync "$@"
echo "=== done $(date) ==="
