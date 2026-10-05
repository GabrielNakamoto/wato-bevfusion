#!/bin/bash
# BEVFusion inference entrypoint
# Usage: ./run.sh [det|seg]
# Works outside the container — dispatches into docker automatically.

MODE="${1:-det}"

if [ "$MODE" = "seg" ]; then
  CONFIG="configs/nuscenes/seg/fusion-bev256d2-lss.yaml"
  CHECKPOINT="pretrained/bevfusion-seg.pth"
  EVAL="map"
else
  CONFIG="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
  CHECKPOINT="pretrained/bevfusion-det.pth"
  EVAL="bbox"
fi

MASTER_PORT="${MASTER_PORT:-29501}"

docker exec -e MASTER_PORT="$MASTER_PORT" bevfusion-dev bash -c \
  "cd /home/bevfusion && \
   MASTER_HOST=localhost:\$MASTER_PORT PYTHONPATH=. \
   mpirun -np 1 --allow-run-as-root \
   python tools/test.py '$CONFIG' '$CHECKPOINT' --eval '$EVAL' \
   --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=1"