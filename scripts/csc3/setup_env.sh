#!/bin/bash
# Reproducible CUDA 12.8 / Blackwell (sm_120) environment for the BEVFusion
# dual-head training runs on csc3. No sudo, no docker (the account is not in
# the docker group), so everything lives in a conda env.
#
# Usage:
#   ./setup_env.sh deps     # conda toolchain + pip packages + patches
#   ./setup_env.sh ext      # build mmdet3d CUDA extensions
#   ./setup_env.sh verify   # import + GPU smoke
#   ./setup_env.sh all
set -euo pipefail

ROOT="${ROOT:-/users/gnakamot/wato}"
ENV="${ENV:-$ROOT/envs/bf}"
REPO="${REPO:-$ROOT/repo/wato-bevfusion}"
PY="$ENV/bin/python"
CONDA="$HOME/miniforge3/bin/conda"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

deps() {
  if [ ! -x "$PY" ]; then
    log "creating conda env $ENV"
    "$CONDA" create -y -p "$ENV" python=3.12
  fi

  log "conda toolchain: CUDA 12.8 + gcc 12"
  "$CONDA" install -y -p "$ENV" -c conda-forge \
    cuda-toolkit=12.8 gxx_linux-64=12 openmpi mpi4py

  log "pip base"
  "$PY" -m pip install --upgrade pip "setuptools<70" wheel
  "$PY" -m pip install "numpy==1.26.4"

  log "torch 2.9.1 + torchvision (cu128, sm_120)"
  "$PY" -m pip install torch==2.9.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

  log "mmcv / mmdet (pure python mmcv: no ops)"
  MMCV_WITH_OPS=0 "$PY" -m pip install --no-build-isolation "mmcv==1.4.0"
  "$PY" -m pip install mmdet==2.20.0 nuscenes-devkit opencv-python-headless \
    "Pillow>=10,<12" pycocotools terminaltables torchpack tqdm addict \
    "yapf<0.40.2" packaging "numba>=0.61,<0.62" pyyaml
  "$PY" -m pip install "numpy==1.26.4"

  log "site-packages patches (shapely 2.x, mmcv logger, mmdet DDP eval)"
  "$PY" "$ROOT/repo/wato-bevfusion/scripts/csc3/patch_site_packages.py" --env "$ENV"

  if [ -f "$ROOT/repo/wato-bevfusion/docker/mmcv_ext_stub.py" ]; then
    cp "$ROOT/repo/wato-bevfusion/docker/mmcv_ext_stub.py" \
      "$($PY -c 'import mmcv, os; print(os.path.dirname(mmcv.__file__))')/_ext.py"
  fi

  log "deps done"
}

ext() {
  log "building CUDA extensions for sm_120 (this takes ~10-20 min)"
  export CUDA_HOME="$ENV"
  export PATH="$ENV/bin:/usr/local/cuda/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="12.0"
  export FORCE_CUDA=1
  export MAX_JOBS="${MAX_JOBS:-8}"
  export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX"
  export CPATH="$ENV/targets/x86_64-linux/include:$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')"
  export LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

  cd "$REPO"
  "$PY" -m pip install -e . --no-build-isolation

  log "ext done"
}

verify() {
  log "imports"
  "$PY" - <<'PY'
import torch, torchvision, mmcv, mmdet, numpy, cv2
import mmdet3d
from mmdet3d.ops import Voxelization
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("mmcv", mmcv.__version__, "mmdet", mmdet.__version__, "numpy", numpy.__version__)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(i, torch.cuda.get_device_name(i), f"sm_{cap[0]}{cap[1]}", torch.cuda.get_device_properties(i).total_memory // 2**30, "GiB")
print("torchvision nms:", torchvision.ops.nms is not None)
PY
  log "verify done"
}

case "${1:-all}" in
  deps) deps ;;
  ext) ext ;;
  verify) verify ;;
  all) deps; ext; verify ;;
  *) echo "usage: $0 [deps|ext|verify|all]"; exit 1 ;;
esac
