#!/bin/bash
# CPU-only job through the NFS uv env: sbatch slurm/cpu.sbatch <command...>
#SBATCH --job-name=sdsb-cpu
#SBATCH --output=/mnt/data0/shared/colin/Projects/SchrodingerDistillation/slurm/logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24gb
#SBATCH --time=02:00:00
#SBATCH --partition=express
#SBATCH --exclude=192-222-56-117,192-222-56-240,192-222-51-248,bacon
set -euo pipefail
NFS=/mnt/data0/shared/colin
PROJECT="$NFS/Projects/SchrodingerDistillation"
export PATH="$NFS/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$NFS/.uv/python"
export UV_CACHE_DIR="$NFS/.uv/cache"
export TMPDIR="$PROJECT/.scratch/tmp"; mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$PROJECT"
echo "host=$(hostname) arch=$(uname -m) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} date=$(date)"
echo "=== run: $* ==="
uv run --no-sync "$@"
echo "=== done $(date) ==="
