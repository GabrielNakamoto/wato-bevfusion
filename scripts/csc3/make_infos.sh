#!/bin/bash
# Build nuScenes infos + groundtruth database for the staged run.
#
# Expects the dataset staged at $ROOT/data/nuscenes (see stage_nuscenes.sh) and
# a symlink at $REPO/data/nuscenes so the relative paths in the configs work.
set -euo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
DATA="${DATA:-$ROOT/data/nuscenes}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
PY="$ENV/bin/python"
WORKERS="${WORKERS:-16}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "checking dataset layout"
for path in samples sweeps maps v1.0-trainval; do
  if [ ! -d "$DATA/$path" ]; then
    echo "missing $DATA/$path" >&2
    exit 1
  fi
done
if [ ! -d "$DATA/maps/expansion" ]; then
  echo "missing map expansion at $DATA/maps/expansion" >&2
  exit 1
fi
echo "keyframes: $(find "$DATA/samples/CAM_FRONT" -name '*.jpg' | wc -l)"
echo "lidar keyframes: $(find "$DATA/samples/LIDAR_TOP" -name '*.pcd.bin' | wc -l)"
echo "lidar sweeps: $(find "$DATA/sweeps/LIDAR_TOP" -name '*.pcd.bin' | wc -l)"

mkdir -p "$REPO/data"
if [ ! -e "$REPO/data/nuscenes" ]; then
  ln -s "$DATA" "$REPO/data/nuscenes"
fi

cd "$REPO"
log "create_data (infos + dbinfos, max_sweeps=10)"
"$PY" tools/create_data.py nuscenes \
  --root-path "$DATA" \
  --out-dir "$DATA" \
  --extra-tag nuscenes \
  --version v1.0 \
  --max-sweeps 10 \
  --workers "$WORKERS"

log "counts"
"$PY" - "$DATA" <<'PY'
import pickle
import sys

root = sys.argv[1]
expected = {"train": 28130, "val": 6019}
for split, want in expected.items():
    path = f"{root}/nuscenes_infos_{split}.pkl"
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    status = "ok" if len(infos) == want else f"EXPECTED {want}"
    print(f"{split}: {len(infos)} ({status})")
    sample = infos[0]
    for key in ("lidar2ego", "ego2global", "location"):
        if key not in sample:
            print(f"  WARNING: missing {key} in infos[0]")
PY

log "done"
