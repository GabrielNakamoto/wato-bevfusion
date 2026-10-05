#!/bin/bash
# BEVFusion inference entrypoint
# Usage: ./run.sh [det|seg] [--viz] [--out-dir path] [--num-samples N]
#
# Examples:
#   ./run.sh seg                        # seg evaluation on val split
#   ./run.sh det                        # detection evaluation on val split
#   ./run.sh seg --viz                  # seg eval + dump visualizations to ./viz/
#   ./run.sh seg --viz --out-dir /tmp/viz_out --num-samples 300
#
# Port forwarding for local viewing of visualizations:
#   docker exec bevfusion-dev bash -c "cd /home/bevfusion/viz && python3 -m http.server 8080"
#   ssh -L 8080:trpro-slurm1:8080 raine@thor.cluster.watonomous.ca
#   Open http://localhost:8080

set -e

MODE="${1:-det}"
VIZ=""
OUT_DIR="${VIZ_DIR:-./viz}"
NUM_SAMPLES=""

shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --viz) VIZ="--viz" ;;
    --out-dir) OUT_DIR="$2"; shift ;;
    --num-samples) NUM_SAMPLES="$2"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

if [ "$MODE" = "seg" ]; then
  CONFIG="configs/nuscenes/seg/fusion-bev256d2-lss.yaml"
  CHECKPOINT="pretrained/bevfusion-seg.pth"
  EVAL="map"
else
  CONFIG="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
  CHECKPOINT="pretrained/bevfusion-det.pth"
  EVAL="bbox"
fi

if [ -n "$NUM_SAMPLES" ]; then
  TRUNC_PKL="/tmp/nuscenes_infos_val_trunc.pkl"
  docker exec bevfusion-dev python -c "
import pickle
src = '/home/bevfusion/data/nuscenes/nuscenes_infos_val.pkl'
dst = '$TRUNC_PKL'
with open(src, 'rb') as f:
    data = pickle.load(f)
k = 'infos' if 'infos' in data else list(data.keys())[0]
data[k] = data[k][:$NUM_SAMPLES]
with open(dst, 'wb') as f:
    pickle.dump(data, f)
print(f'Truncated to {len(data[k])} samples -> {dst}')
"
  CFG_OVERRIDES="data.test.ann_file=$TRUNC_PKL data.samples_per_gpu=1 data.workers_per_gpu=1"
else
  CFG_OVERRIDES="data.samples_per_gpu=1 data.workers_per_gpu=1"
fi

docker exec bevfusion-dev bash -c "
  PORT=\$(( ( RANDOM % 50000 )  + 1025 ))
  export MASTER_HOST=localhost:\$PORT
  cd /home/bevfusion
  echo 'Using port \$PORT'
  if [ \"$VIZ\" != \"\" ]; then
    mkdir -p '$OUT_DIR' '$OUT_DIR/camera' '$OUT_DIR/lidar' '$OUT_DIR/map'
    PYTHONPATH=. mpirun -np 1 --allow-run-as-root \
      python tools/test.py '$CONFIG' '$CHECKPOINT' \
      --eval '$EVAL' \
      --show-dir '$OUT_DIR' \
      --cfg-options $CFG_OVERRIDES
  else
    PYTHONPATH=. mpirun -np 1 --allow-run-as-root \
      python tools/test.py '$CONFIG' '$CHECKPOINT' \
      --eval '$EVAL' \
      --cfg-options $CFG_OVERRIDES
  fi
"