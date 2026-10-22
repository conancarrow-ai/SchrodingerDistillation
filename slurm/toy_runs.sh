#!/bin/bash
# Submit the toy-distribution validation: every distribution of sparse_dsb/toy.py with the reference
# score network (dense) and with its linear layers replaced by sparse-transformers layers.
#   bash slurm/toy_runs.sh            # all 16 runs
#   bash slurm/toy_runs.sh moon       # one distribution
# Each run is one single-GPU job of slurm/run.sbatch writing runs/toy_<distribution>_<variant>/.
set -euo pipefail
cd /mnt/data0/shared/colin/Projects/SchrodingerDistillation
DISTRIBUTIONS=${1:-"moon scurve circle mixture"}
declare -A VARIANT=(
  [dense]="model.sparse_linear=dense"
  [maskfull]="model.sparse_linear=mask model.linear_fan_in=null"
  [mask32]="model.sparse_linear=mask model.linear_fan_in=32"
  [gather32]="model.sparse_linear=gather model.linear_fan_in=32"
)
for distribution in $DISTRIBUTIONS; do
  for variant in dense maskfull mask32 gather32; do
    out="runs/toy_${distribution}_${variant}"
    if [ -e "$out/logs/metrics.csv" ]; then echo "skip $out (exists)"; continue; fi
    sbatch --job-name="toy-${distribution}-${variant}" --time=03:00:00 --mem=16gb slurm/run.sbatch \
      python -m sparse_dsb.train --config configs/toy.yaml --out "$out" \
      "data.distribution=${distribution}" ${VARIANT[$variant]}
  done
done
