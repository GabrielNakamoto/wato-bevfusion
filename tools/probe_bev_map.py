#!/usr/bin/env python
"""Probes for the BEV map labels themselves.

Two questions that decide how much of the thin-class gap is learnable:

``balance``
    What fraction of the 200x200 map is positive per class? Feeds the loss
    class weights (``w_c = clamp(median/pos_share, 0.5, 20)``).

``ceiling``
    How stable is the 0.5 m raster under a finer rendering? Renders at 0.25 m,
    max-pools to 0.5 m and compares with the native 0.5 m raster. Large
    disagreement on divider/stop_line means part of their IoU ceiling is label
    quantisation, not model error.

Example:
    python tools/probe_bev_map.py --mode balance --infos data/nuscenes/nuscenes_infos_train.pkl --samples 500
    python tools/probe_bev_map.py --mode ceiling --infos data/nuscenes/nuscenes_infos_val.pkl --samples 200
"""
import argparse
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from mmdet3d.datasets.pipelines.loading import LoadBEVSegmentation

MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("balance", "ceiling"), required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--dataset-root", default="data/nuscenes/")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load_infos(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload["infos"] if isinstance(payload, dict) else payload


def sample_input(info):
    ego2global = np.array(info["ego2global"], dtype=np.float64)
    lidar2ego = np.array(info["lidar2ego"], dtype=np.float64)
    return {
        "lidar_aug_matrix": np.eye(4, dtype=np.float64),
        "lidar2ego": lidar2ego,
        "ego2global": ego2global,
        "location": info["location"],
    }


def render(transform, info):
    data = sample_input(info)
    return transform(data)["gt_masks_bev"]  # (C, H, W) long


def main():
    args = parse_args()
    infos = load_infos(args.infos)[: args.samples]

    coarse = LoadBEVSegmentation(
        dataset_root=args.dataset_root,
        xbound=(-50.0, 50.0, 0.5),
        ybound=(-50.0, 50.0, 0.5),
        classes=MAP_CLASSES,
    )

    if args.mode == "balance":
        counts = np.zeros(len(MAP_CLASSES), dtype=np.float64)
        total = 0
        for info in infos:
            mask = render(coarse, info)
            counts += mask.reshape(len(MAP_CLASSES), -1).sum(axis=1)
            total += mask.shape[-1] * mask.shape[-2]
        share = counts / max(1, total)
        result = {
            "samples": len(infos),
            "positive_share": dict(zip(MAP_CLASSES, share.tolist())),
        }
        median = float(np.median(share[share > 0])) if np.any(share > 0) else 1.0
        weights = np.clip(median / np.maximum(share, 1e-9), 0.5, 20.0)
        result["suggested_class_weights"] = dict(zip(MAP_CLASSES, weights.tolist()))
    else:
        fine = LoadBEVSegmentation(
            dataset_root=args.dataset_root,
            xbound=(-50.0, 50.0, 0.25),
            ybound=(-50.0, 50.0, 0.25),
            classes=MAP_CLASSES,
        )
        inter = np.zeros(len(MAP_CLASSES))
        union = np.zeros(len(MAP_CLASSES))
        pos_coarse = np.zeros(len(MAP_CLASSES))
        pos_fine = np.zeros(len(MAP_CLASSES))
        for info in infos:
            lo = render(coarse, info).astype(bool)
            hi = render(fine, info).astype(np.float32)
            hi = (
                F.max_pool2d(torch.from_numpy(hi).unsqueeze(0), 2)
                .squeeze(0)
                .numpy()
                .astype(bool)
            )
            for index in range(len(MAP_CLASSES)):
                inter[index] += np.logical_and(lo[index], hi[index]).sum()
                union[index] += np.logical_or(lo[index], hi[index]).sum()
                pos_coarse[index] += lo[index].sum()
                pos_fine[index] += hi[index].sum()
        iou = inter / np.maximum(union, 1)
        result = {
            "samples": len(infos),
            "raster_agreement_iou": dict(zip(MAP_CLASSES, iou.tolist())),
            "positives_native_0.5m": dict(zip(MAP_CLASSES, pos_coarse.tolist())),
            "positives_fine_0.25m_pooled": dict(zip(MAP_CLASSES, pos_fine.tolist())),
            "ratio_fine_over_native": dict(
                zip(MAP_CLASSES, (pos_fine / np.maximum(pos_coarse, 1)).tolist())
            ),
        }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
