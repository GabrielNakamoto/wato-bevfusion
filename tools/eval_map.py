#!/usr/bin/env python
"""Deterministic single-process eval for the dual-head model.

Reports both the paper-comparable metric (``iou@max``, best threshold per class)
and ``iou@0.5`` (what the deployed binary threshold actually produces), plus
detection mAP/NDS when an object head is present. Writes a JSON next to the
checkpoint so runs are comparable without digging through logs.

Example:
    python tools/eval_map.py configs/nuscenes/seg/resnet50_dualhead/a5_staged.yaml \
        --checkpoint models/dualhead_epoch10.pth --out eval/dualhead_epoch10.json
"""
import argparse
import hashlib
import json
import os
import warnings

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack.utils.config import configs

from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
import mmdet3d.datasets  # noqa: F401  (registers NuScenesDataset)
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def parse_args():
    parser = argparse.ArgumentParser(description="single-process map/bbox eval")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--samples-per-gpu", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def per_class_table(metrics, classes):
    table = {}
    for name in classes:
        ious = {
            f"{t:.2f}": metrics.get(f"map/{name}/iou@{t:.2f}") for t in THRESHOLDS
        }
        available = {k: v for k, v in ious.items() if v is not None}
        best_threshold = max(available, key=available.get)
        table[name] = {
            "iou_max": available[best_threshold],
            "best_threshold": float(best_threshold),
            "iou_at_0.5": available.get("0.50"),
            "per_threshold": available,
        }
    return table


def main():
    args = parse_args()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    data_cfg = cfg.data[args.split]
    data_cfg.test_mode = True
    data_cfg.pop("samples_per_gpu", None)
    if args.samples_per_gpu > 1:
        data_cfg.pipeline = replace_ImageToTensor(data_cfg.pipeline)

    dataset = build_dataset(data_cfg)
    if args.max_samples:
        dataset.data_infos = dataset.data_infos[: args.max_samples]
        if hasattr(dataset, "flag"):
            dataset.flag = dataset.flag[: args.max_samples]

    workers = args.workers if args.workers is not None else cfg.data.get("workers_per_gpu", 4)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False,
    )

    cfg.model.pretrained = None
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    model.CLASSES = dataset.CLASSES
    if hasattr(dataset, "map_classes"):
        model.MAP_CLASSES = dataset.map_classes

    # mmcv 1.4 returns the checkpoint dict, not a missing/unexpected report;
    # the loader logs those itself.
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDataParallel(model.cuda(), device_ids=[torch.device("cuda", 0)])
    model.eval()
    with torch.no_grad():
        results = single_gpu_test(model, loader, show=False, out_dir=None)

    metrics = {}
    try:
        metrics.update(dataset.evaluate_map(results))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"map eval failed: {exc}")

    det = {}
    if any("boxes_3d" in result for result in results[:1]):
        try:
            det = dataset.evaluate(results, metric="bbox")
            det = {k: v for k, v in det.items() if isinstance(v, (int, float))}
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"bbox eval failed: {exc}")

    classes = getattr(dataset, "map_classes", [])
    table = per_class_table(metrics, classes)
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": os.path.abspath(args.config),
        "split": args.split,
        "num_samples": len(dataset),
        "map": {
            "miou_max": metrics.get("map/mean/iou@max"),
            "miou_at_0.5": sum(
                v["iou_at_0.5"] for v in table.values() if v["iou_at_0.5"] is not None
            )
            / max(1, len(table)),
            "per_class": table,
        },
        "det": det,
        "raw_metrics": metrics,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps({k: summary["map"][k] for k in ("miou_max", "miou_at_0.5")}, indent=2))
    if det:
        print(json.dumps({k: det[k] for k in det if k in ("bbox_mAP", "nds", "bbox_mAP_50")}, indent=2))


if __name__ == "__main__":
    main()
