#!/bin/bash
# Keep the A5 run alive: resume from the latest checkpoint if the training
# process disappears before the final epoch.
#
# The first launch was killed by a SIGINT to mpiexec (OpenMPI "Abort is in
# progress"), so training is wrapped in setsid as well as supervised.
set -uo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
RUN="${1:?usage: supervise_a5.sh RUN_DIR}"
CONFIG="${CONFIG:-configs/dualhead/a5_staged.yaml}"
MAX_EPOCHS="${MAX_EPOCHS:-22}"
SLEEP="${SLEEP:-60}"

cd "$REPO"
mkdir -p "$RUN"

build_resume_config() {
  local ckpt="$1" out="$2"
  "$ENV/bin/python" - "$CONFIG" "$ckpt" "$out" <<'PY'
import sys, yaml
base, ckpt, out = sys.argv[1:4]
cfg = yaml.safe_load(open(base))
cfg["resume_from"] = ckpt
cfg["load_from"] = None
yaml.safe_dump(cfg, open(out, "w"), sort_keys=True, default_flow_style=False)
PY
}

while true; do
  if [ -f "$RUN/epoch_${MAX_EPOCHS}.pth" ]; then
    echo "$(date '+%F %T') final checkpoint present; supervisor exiting"
    exit 0
  fi

  if pgrep -f "tools/train.py $CONFIG" > /dev/null || pgrep -f "tools/train.py configs/dualhead" > /dev/null; then
    sleep "$SLEEP"
    continue
  fi

  if [ -f "$RUN/latest.pth" ]; then
    RESUME="$RUN/resume_config.yaml"
    build_resume_config "$RUN/latest.pth" "$RESUME"
    echo "$(date '+%F %T') training not running; resuming from $RUN/latest.pth"
    RUN="$RUN" CONFIG="$RESUME" bash scripts/csc3/run_a5.sh >> "$RUN/supervisor.log" 2>&1
  else
    echo "$(date '+%F %T') training not running and no checkpoint; launching fresh"
    RUN="$RUN" CONFIG="$CONFIG" bash scripts/csc3/run_a5.sh >> "$RUN/supervisor.log" 2>&1
  fi
  sleep 30
done
