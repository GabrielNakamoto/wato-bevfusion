#!/bin/bash
# Evaluate stage-boundary checkpoints from a running (or finished) A5 run.
#
# Runs single-process on one GPU, so it is independent of the training DDP
# process. Polls for the checkpoints and writes one JSON per epoch.
#
#   ./watch_eval.sh /users/gnakamot/wato/runs/a5_YYYYmmdd_HHMM 2 15 22
set -uo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
RUN="${1:?usage: watch_eval.sh RUN_DIR [epochs...]}"
shift || true
EPOCHS=("${@:-2 15 22}")
GPU="${GPU:-1}"
CONFIG="${CONFIG:-configs/dualhead/a5_staged.yaml}"
POLL="${POLL:-120}"

cd "$REPO"
export PYTHONPATH="$PWD"

for epoch in ${EPOCHS[@]}; do
  ckpt="$RUN/epoch_${epoch}.pth"
  out="$RUN/eval_epoch_${epoch}.json"
  [ -f "$out" ] && { echo "skip epoch $epoch (already evaluated)"; continue; }
  while [ ! -f "$ckpt" ]; do
    if ! pgrep -f "tools/train.py" > /dev/null; then
      echo "training finished before epoch $epoch; stopping watcher"
      exit 0
    fi
    sleep "$POLL"
  done
  echo "evaluating $ckpt"
  CUDA_VISIBLE_DEVICES="$GPU" "$ENV/bin/python" tools/eval_map.py \
    "$CONFIG" --checkpoint "$ckpt" --split val --out "$out" \
    > "$RUN/eval_epoch_${epoch}.log" 2>&1
  echo "epoch $epoch: $(tail -c 300 "$RUN/eval_epoch_${epoch}.log" | tr -d '\r' | tail -2)"
done
echo "watcher done"
