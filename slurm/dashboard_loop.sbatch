#!/bin/bash
# The dashboard regenerator on a compute node: sbatch slurm/dashboard_loop.sbatch
# Rewrites /mnt/data0/shared/colin/Projects/SchrodingerDistillation/dashboard every 120 s
# (python -m sparse_dsb.dashboard --loop 120); scripts/serve_dashboard.sh serves that directory from fokker.
#SBATCH --job-name=sdsb-dashboard
#SBATCH --output=/mnt/data0/shared/colin/Projects/SchrodingerDistillation/slurm/logs/%x-%j.out
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=192-222-51-248,192-222-56-117,192-222-56-240,bacon
set -euo pipefail
NFS=/mnt/data0/shared/colin
PROJECT="$NFS/Projects/SchrodingerDistillation"
export PATH="$NFS/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$NFS/.uv/python"
export UV_CACHE_DIR="$NFS/.uv/cache"
export MPLCONFIGDIR="$NFS/.mpl"; mkdir -p "$MPLCONFIGDIR"
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$PROJECT/.scratch/cache"; mkdir -p "$XDG_CACHE_HOME"
export TMPDIR="$PROJECT/.scratch/tmp"; mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
test -x "$NFS/bin/uv" || { echo "ERROR: $NFS/bin/uv missing"; exit 1; }
cd "$PROJECT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -u) out=$PROJECT/dashboard"
exec uv run --no-sync python -m sparse_dsb.dashboard --out "$PROJECT/dashboard" --loop 120
