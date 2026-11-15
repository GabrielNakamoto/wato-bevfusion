#!/bin/bash
# Launch the A5 staged dual-head run on 2x RTX PRO 6000 (csc3).
#
#   ./run_a5.sh                 # fresh run
#   RUN=... ./run_a5.sh         # explicit run directory
#   CONFIG=... ./run_a5.sh      # smoke or alternate config
#   EXTRA="evaluation.interval=1" ./run_a5.sh
#
# Uses plain `mpiexec` rather than `torchpack dist-run`: torchpack passes
# `--allow-run-as-root -bind-to none -map-by slot`, which OpenMPI 5 / PRRTE
# rejects ("prterun interprets the first unrecognized command line token as the
# executable"). torchpack only needs MASTER_HOST plus mpi4py ranks, both of
# which this provides.
set -euo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
CONFIG="${CONFIG:-configs/dualhead/a5_staged.yaml}"
RUN="${RUN:-$ROOT/runs/a5_$(date +%Y%m%d_%H%M)}"
EXTRA="${EXTRA:-}"
NP="${NP:-2}"
VALIDATE_FLAGS="--no-validate"
if [ "${VALIDATE:-0}" = "1" ]; then VALIDATE_FLAGS=""; fi

mkdir -p "$RUN"
cd "$REPO"

export PATH="$ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$PWD"
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_HOST="localhost:$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')"

echo "run dir:  $RUN"
echo "config:   $CONFIG"
echo "devices:  $CUDA_VISIBLE_DEVICES"
echo "master:   $MASTER_HOST"

setsid nohup mpiexec -np "$NP" \
  -x MASTER_HOST -x PYTHONPATH -x CUDA_VISIBLE_DEVICES -x NCCL_DEBUG \
  -x OMPI_MCA_btl_vader_single_copy_mechanism \
  "$ENV/bin/python" tools/train.py "$CONFIG" --run-dir "$RUN" $VALIDATE_FLAGS $EXTRA \
  > "$RUN/train.log" 2>&1 &

echo "pid $!"
echo "tail -f $RUN/train.log"
