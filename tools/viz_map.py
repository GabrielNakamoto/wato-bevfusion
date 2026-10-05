#!/usr/bin/env python
"""Run BEVFusion seg inference on a few val samples and save BEV map PNGs."""
import argparse, os, sys, pickle
import numpy as np
import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from mmdet.apis import multi_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

sys.setrecursionlimit(10000)

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = dict(configs) if hasattr(configs, "__getitem__") else {k: v for k, v in globals().items() if not k.startswith("_")}
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)
    return obj

def main():
    dist.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", default="map_viz")
    parser.add_argument("--split", default="val")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    dataset = build_dataset(cfg.data[args.split])
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1, dist=True, shuffle=False,
    )

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDistributedDataParallel(
        model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False,
    )
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    for i, data in enumerate(data_loader):
        if i >= 10:
            break
        with torch.inference_mode():
            result = model(return_loss=False, rescale=True, **data)
        metas = data["metas"].data[0][0]
        name = f"{metas['timestamp']}_{metas['token']}"
        if "masks_bev" in result[0]:
            masks = result[0]["masks_bev"]
            # masks shape: [num_classes, H, W]
            # Save a color composite of the map
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            num_classes = masks.shape[0]
            fig, axes = plt.subplots(1, num_classes + 1, figsize=(4*(num_classes+1), 4))
            # Composite all classes
            composite = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))[:, :3] * 255
            for c in range(num_classes):
                composite[masks[c] >= 0.5] = colors[c].astype(np.uint8)
                axes[c].imshow(masks[c], cmap="gray", vmin=0, vmax=1)
                axes[c].set_title(cfg.map_classes[c] if hasattr(cfg, "map_classes") and c < len(cfg.map_classes) else f"class {c}")
                axes[c].axis("off")
            axes[-1].imshow(composite)
            axes[-1].set_title("composite")
            axes[-1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"{name}.png"), bbox_inches="tight")
            plt.close()
            if (i + 1) % 5 == 0:
                print(f"  saved {i+1}/{min(10, len(data_loader))}")
        else:
            print(f"  no maske_bev in output for sample {i}")

    print(f"Done. Maps saved to {args.out_dir}/")

if __name__ == "__main__":
    main()