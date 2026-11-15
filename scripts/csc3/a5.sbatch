#!/bin/bash
#SBATCH --job-name=a5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=23:00:00

# A5 staged dual-head run under SLURM.
#
# csc3/tiramisu is SLURM-managed: GPU device access is granted per allocation,
# so training must run inside a job. This script also resumes from
# <RUN>/latest.pth if the process dies, and exits once epoch_22.pth exists.
#
#   sbatch --chdir=<repo> --output=<RUN>/slurm-%j.log \
#          --export=ALL,RUN=<RUN> scripts/csc3/a5.sbatch
set -uo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
RUN="${RUN:-$ROOT/runs/a5_slurm_$(date +%Y%m%d_%H%M)}"
CONFIG="${CONFIG:-configs/dualhead/a5_staged.yaml}"
MAX_EPOCHS="${MAX_EPOCHS:-22}"
HOURS="${HOURS:-22}"

mkdir -p "$RUN"
cd "$REPO"

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$PWD"
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "[sbatch] job=${SLURM_JOB_ID:-local} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} run=$RUN"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

deadline=$(( $(date +%s) + HOURS * 3600 ))

while [ ! -f "$RUN/epoch_${MAX_EPOCHS}.pth" ] && [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -f "$RUN/latest.pth" ]; then
    config="$RUN/resume_config.yaml"
    "$ENV/bin/python" - "$CONFIG" "$RUN/latest.pth" "$config" <<'PY'
import sys
import yaml

base, ckpt, out = sys.argv[1:4]
cfg = yaml.safe_load(open(base))
cfg["resume_from"] = ckpt
cfg["load_from"] = None
yaml.safe_dump(cfg, open(out, "w"), sort_keys=True, default_flow_style=False)
print(f"[sbatch] resuming from {ckpt}")
PY
  else
    config="$CONFIG"
    echo "[sbatch] fresh start"
  fi

  export MASTER_HOST="localhost:$("$ENV/bin/python" -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')"
  echo "[sbatch] $(date '+%F %T') launching (config=$config master=$MASTER_HOST)"

  mpiexec -np 2 \
    -x MASTER_HOST -x PYTHONPATH -x CUDA_VISIBLE_DEVICES -x NCCL_DEBUG \
    -x OMPI_MCA_btl_vader_single_copy_mechanism \
    "$ENV/bin/python" tools/train.py "$config" --run-dir "$RUN" --no-validate
  rc=$?
  echo "[sbatch] $(date '+%F %T') training exited rc=$rc"
  if [ $rc -eq 0 ]; then
    break
  fi
  sleep 30
done

echo "[sbatch] $(date '+%F %T') done; checkpoints:"
ls -1 "$RUN"/epoch_*.pth 2>/dev/null | tail -3
