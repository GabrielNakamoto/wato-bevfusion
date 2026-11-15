#!/usr/bin/env python
"""Materialise the released-paper configs into standalone YAML for the harness gate.

The repo's configs are layered with ``${...}`` references and no ``_base_``, so
torchpack's `configs` object is loaded sequentially and dumped as one resolved
file. ``init_cfg`` is stripped: the released checkpoints provide every weight,
and the Swin ImageNet URL must not be fetched during eval.

    python scripts/csc3/build_official_configs.py
"""
import os

import yaml
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.utils import recursive_eval

SEG_CHAIN = [
    "configs/default.yaml",
    "configs/nuscenes/default.yaml",
    "configs/nuscenes/seg/default.yaml",
    "configs/nuscenes/seg/fusion-bev256d2-lss.yaml",
]

DET_CHAIN = [
    "configs/default.yaml",
    "configs/nuscenes/default.yaml",
    "configs/nuscenes/det/default.yaml",
    "configs/nuscenes/det/transfusion/default.yaml",
    "configs/nuscenes/det/transfusion/secfpn/default.yaml",
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/default.yaml",
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
]


def plain(value):
    """PyYAML cannot represent mmcv Config / ConfigDict directly."""
    if hasattr(value, "_cfg_dict"):
        value = value._cfg_dict
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def build(chain, out):
    # torchpack's `configs` is a process-global; reload from scratch each time.
    configs.clear()
    for path in chain:
        configs.load(path, recursive=True)
    cfg = Config(recursive_eval(configs))
    backbone = cfg.model.encoders.camera.backbone
    if backbone.get("init_cfg") is not None:
        backbone["init_cfg"] = None
    cfg.load_from = None
    cfg.pop("lr_config", None)
    cfg.pop("momentum_config", None)
    cfg.pop("runner", None)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        yaml.safe_dump(plain(cfg), handle, sort_keys=True, default_flow_style=False)
    print(f"wrote {out}")
    print("  image_size:", cfg.model.encoders.camera.vtransform.image_size)
    print("  point_cloud_range:", cfg.point_cloud_range, "voxel:", cfg.voxel_size)
    print("  heads:", {k: (v is not None) for k, v in cfg.model.heads.items()})


def build_det_resnet50(out):
    """Det-only view of the A5 architecture (our released det checkpoint is ResNet50)."""
    configs.clear()
    configs.load("configs/nuscenes/seg/resnet50_dualhead/a5_staged.yaml", recursive=True)
    cfg = Config(recursive_eval(configs))
    cfg.model.heads.map = None
    cfg.model.pop("training_stages", None)
    cfg.load_from = None
    for key in ("lr_config", "momentum_config", "runner", "optimizer", "optimizer_config"):
        cfg.pop(key, None)
    with open(out, "w") as handle:
        yaml.safe_dump(plain(cfg), handle, sort_keys=True, default_flow_style=False)
    print(f"wrote {out}")
    print("  map head:", cfg.model.heads.map, "object head:", cfg.model.heads.object is not None)


if __name__ == "__main__":
    build(SEG_CHAIN, "configs/nuscenes/seg/official/seg_check.yaml")
    build(DET_CHAIN, "configs/nuscenes/det/official/det_check.yaml")
    build_det_resnet50("configs/nuscenes/det/official/det_resnet50_check.yaml")
