#!/usr/bin/env python
"""Generate the A5 staged training config from the dual-head correction baseline.

Single source of truth for the stage schedule: keep `STAGES` in sync with
ADR-002 §5.4. The model config and the lr hook config both embed it; the
training entry point asserts they agree.

Usage:
  python scripts/csc3/build_a5_config.py \
      --base configs/nuscenes/seg/resnet50_dualhead/dualhead_base.yaml \
      --out  configs/nuscenes/seg/resnet50_dualhead/a5_staged.yaml \
      --load-from models/dualhead_epoch10.pth \
      --weights-json artifacts/bev_map_balance_train.json
"""
import argparse
import copy
import json

import yaml

# (until_epoch, trainable specs, bn_eval specs, per-group lr)
STAGES = [
    dict(
        until_epoch=2,
        trainable=["heads.map"],
        bn_eval=["encoders", "fuser", "decoder"],
        lr={"heads": 3.0e-4},
    ),
    dict(
        until_epoch=15,
        trainable=[
            "heads",
            "decoder",
            "encoders.camera.neck",
            "encoders.camera.backbone",
        ],
        bn_eval=["encoders.lidar"],
        lr={
            "heads": 1.0e-4,
            "decoder": 1.0e-4,
            "camera_neck": 1.0e-4,
            "camera_backbone": 3.0e-5,
        },
    ),
    dict(
        until_epoch=22,
        trainable=[
            "heads",
            "decoder",
            "encoders.camera",
            "encoders.lidar",
            "fuser",
        ],
        bn_eval=["encoders.lidar"],
        lr={
            "heads": 3.0e-5,
            "decoder": 3.0e-5,
            "camera_neck": 1.0e-5,
            "camera_backbone": 1.0e-5,
            "camera_vtransform": 1.0e-5,
            "lidar": 1.0e-5,
            "fuser": 1.0e-5,
        },
    ),
]

DEFAULT_WEIGHTS = [1.0, 2.0, 1.3, 2.5, 1.7, 1.7]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--load-from", required=True)
    parser.add_argument("--weights-json", default=None)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--samples-per-gpu", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=2)
    return parser.parse_args()


def resolve_weights(args):
    if args.weights_json:
        payload = json.load(open(args.weights_json))
        suggested = payload["suggested_class_weights"]
        order = list(payload["positive_share"].keys())
        weights = [round(float(suggested[name]), 4) for name in order]
        return weights, order
    return DEFAULT_WEIGHTS, None


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.base))
    weights, order = resolve_weights(args)

    # --- model: stages, loss, hi-res branch -------------------------------
    model = cfg["model"]
    model.pop("frozen_modules", None)
    model.pop("trainable_modules", None)
    model["training_stages"] = copy.deepcopy(STAGES)

    seg = model["heads"]["map"]
    # Pixel-mean focal + inverse-share class weights is algebraically the same
    # rebalancing as positive normalisation, with a bounded loss scale.
    seg["normalize_by_positives"] = False
    seg["dice_weight"] = 0.5
    seg["gamma"] = 2.0
    seg["alpha"] = -1.0
    seg["class_weights"] = weights
    seg["hi_res"] = dict(enabled=True, mid_channels=128, dilation=2)

    # --- optimizer: named per-region groups -------------------------------
    cfg["optimizer"] = {
        "type": "AdamW",
        "lr": 1.0e-4,
        "weight_decay": 0.01,
        "lr_mults": {
            "heads": 1.0,
            "decoder": 0.1,
            "camera_neck": 0.1,
            "camera_backbone": 0.03,
            "camera_vtransform": 0.01,
            "lidar": 0.01,
            "fuser": 0.01,
        },
    }
    cfg["lr_config"] = {
        "type": "StagedTrainableHook",
        "by_epoch": True,
        "warmup_iters": 200,
        "warmup_ratio": 0.1,
        "stages": copy.deepcopy(STAGES),
    }
    cfg["momentum_config"] = None
    cfg["find_unused_parameters"] = True

    # --- schedule ---------------------------------------------------------
    cfg["max_epochs"] = args.epochs
    cfg["runner"]["max_epochs"] = args.epochs
    cfg["checkpoint_config"] = {"interval": 1, "max_keep_ckpts": 100}
    cfg.setdefault("evaluation", {})["interval"] = args.eval_interval

    # --- data: plain repeat (raw dataset, no CBGS) ------------------------
    train = cfg["data"]["train"]
    inner = train.get("dataset", train)
    cfg["data"]["train"] = {"type": "NuScenesDataset", **copy.deepcopy(inner)}
    cfg["data"]["samples_per_gpu"] = args.samples_per_gpu
    cfg["data"]["workers_per_gpu"] = args.workers_per_gpu

    # --- checkpointing ----------------------------------------------------
    cfg["load_from"] = args.load_from
    cfg["resume_from"] = None

    with open(args.out, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True, default_flow_style=False)

    print(f"wrote {args.out}")
    print(f"class weights ({order or 'default order'}): {weights}")
    print("stages:")
    for stage in STAGES:
        print(f"  <{stage['until_epoch']:>2}  {stage['trainable']}  lr={stage['lr']}")


if __name__ == "__main__":
    main()
