#!/usr/bin/env python
"""Build a 4-epoch smoke config from the A5 config.

Keeps the real model/pipeline/optimizer/stages but:
  * uses the strided mini info pickles,
  * compresses the stage schedule to [1, 3, 4] so all three stages are exercised
    in a few minutes (stage 0: epoch 0, stage 1: epochs 1-2, stage 2: epoch 3),
  * evaluates and checkpoints every epoch.

    python scripts/csc3/build_smoke_config.py \\
        --base configs/dualhead/a5_staged.yaml \\
        --out configs/nuscenes/seg/resnet50_dualhead/a5_smoke.yaml
"""
import argparse
import copy

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--samples-per-gpu", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.base))
    bounds = [1, 3, args.epochs]

    stages = cfg["model"]["training_stages"]
    assert len(stages) == len(bounds)
    for stage, bound in zip(stages, bounds):
        stage["until_epoch"] = bound
    cfg["lr_config"]["stages"] = copy.deepcopy(stages)

    cfg["max_epochs"] = args.epochs
    cfg["runner"]["max_epochs"] = args.epochs
    cfg["evaluation"]["interval"] = 1
    cfg["checkpoint_config"] = {"interval": 1, "max_keep_ckpts": 10}

    for split in ("train", "val", "test"):
        block = cfg["data"][split]
        block["ann_file"] = block["ann_file"].replace(".pkl", "_mini.pkl")
    cfg["data"]["samples_per_gpu"] = args.samples_per_gpu
    cfg["data"]["workers_per_gpu"] = args.workers

    with open(args.out, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True, default_flow_style=False)
    print(f"wrote {args.out} (stage bounds {bounds})")


if __name__ == "__main__":
    main()
