#!/bin/bash
# Build nuscenes_dbinfos_train.pkl + nuscenes_gt_database for ObjectPaste.
# Separate from make_infos.sh so a failed/incomplete GT database does not force
# the ~90 minute info conversion to run again.
set -euo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
DATA="${DATA:-$ROOT/data/nuscenes}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
PY="$ENV/bin/python"

cd "$REPO"
"$PY" - "$DATA" <<'PY'
import sys

sys.path.insert(0, "tools")
from data_converter.create_gt_database import create_groundtruth_database

data = sys.argv[1]
create_groundtruth_database(
    "NuScenesDataset",
    data,
    "nuscenes",
    f"{data}/nuscenes_infos_train.pkl",
)
print("dbinfos done")
PY
ls -la "$DATA/nuscenes_dbinfos_train.pkl"
