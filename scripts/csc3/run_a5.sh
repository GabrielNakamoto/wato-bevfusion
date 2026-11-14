#!/bin/bash
# Launch the A5 staged dual-head run on 2x RTX PRO 6000 (csc3).
#
#   ./run_a5.sh                 # fresh run
#   RUN=... ./run_a5.sh         # explicit run directory
#   EXTRA="evaluation.interval=1" ./run_a5.sh
set -euo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
CONFIG="${CONFIG:-configs/nuscenes/seg/resnet50_dualhead/a5_staged.yaml}"
RUN="${RUN:-$ROOT/runs/a5_$(date +%Y%m%d_%H%M)}"
EXTRA="${EXTRA:-}"

mkdir -p "$RUN"
cd "$REPO"

export PATH="$ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$PWD"
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "run dir: $RUN"
echo "config:  $CONFIG"
echo "cuda:    $CUDA_VISIBLE_DEVICES"

nohup "$ENV/bin/torchpack" dist-run -np 2 "$ENV/bin/python" tools/train.py \
  "$CONFIG" --run-dir "$RUN" $EXTRA \
  > "$RUN/train.log" 2>&1 &

echo "pid $!"
echo "tail -f $RUN/train.log"
