#!/bin/bash
# Evaluate stage-boundary checkpoints inside a live SLURM allocation.
#
# On csc3 the GPUs are only reachable from within a job, so the eval runs as an
# overlapping step of the training allocation (`srun --jobid ... --overlap`).
#
#   ./watch_eval_slurm.sh <RUN> <JOBID> 2 15 22
set -uo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
CONFIG="${CONFIG:-configs/dualhead/a5_staged.yaml}"

RUN="${1:?usage: watch_eval_slurm.sh RUN JOBID [epochs...]}"
JOBID="${2:?usage: watch_eval_slurm.sh RUN JOBID [epochs...]}"
shift 2
EPOCHS=("${@:-2 15 22}")
POLL="${POLL:-120}"
GPU="${GPU:-1}"

cd "$REPO"

for epoch in ${EPOCHS[@]}; do
  ckpt="$RUN/epoch_${epoch}.pth"
  out="$RUN/eval_epoch_${epoch}.json"
  [ -f "$out" ] && { echo "$(date '+%F %T') skip epoch $epoch (already evaluated)"; continue; }
  while [ ! -f "$ckpt" ]; do
    if ! squeue -j "$JOBID" -h -o "%T" | grep -qE "RUNNING|PENDING"; then
      echo "$(date '+%F %T') job $JOBID gone before epoch $epoch; stopping watcher"
      exit 0
    fi
    sleep "$POLL"
  done
  echo "$(date '+%F %T') evaluating $ckpt"
  srun --jobid="$JOBID" --overlap --ntasks=1 --cpus-per-task=8 --mem=32G \
    env PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUDA_MPS_PIPE_DIRECTORY=/dev/shm/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/dev/shm/nvidia-log \
    "$ENV/bin/python" tools/eval_map.py "$CONFIG" \
    --checkpoint "$ckpt" --split val --out "$out" \
    > "$RUN/eval_epoch_${epoch}.log" 2>&1
  echo "$(date '+%F %T') epoch $epoch done: $(tr -d '\r' < "$RUN/eval_epoch_${epoch}.log" | tail -2 | tr '\n' ' ')"
done
echo "$(date '+%F %T') watcher done"
