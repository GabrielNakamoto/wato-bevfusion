#!/usr/bin/env python
"""Render BEV map segmentation PNGs from test.py output pickle."""
import pickle, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pkl_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_results.pkl"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/map_pngs"

os.makedirs(out_dir, exist_ok=True)

with open(pkl_path, "rb") as f:
    results = pickle.load(f)

classes = ["drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider"]
num_classes = len(classes)
colors = (plt.cm.tab20(np.linspace(0, 1, num_classes))[:, :3] * 255).astype(np.uint8)

for i, res in enumerate(results):
    masks = res.get("masks_bev")
    if masks is None:
        print(f"  {i}: no masks_bev")
        continue
    # masks shape: [num_classes, H, W]
    H, W = masks.shape[1], masks.shape[2]
    composite = np.zeros((H, W, 3), dtype=np.uint8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for c in range(num_classes):
        binary = masks[c] >= 0.5
        composite[binary] = colors[c]
        axes[c].imshow(masks[c], cmap="gray", vmin=0, vmax=1)
        axes[c].set_title(classes[c])
        axes[c].axis("off")
    axes[-2].imshow(composite)
    axes[-2].set_title("composite")
    axes[-2].axis("off")
    gt = res.get("gt_masks_bev")
    if gt is not None:
        gt_composite = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(num_classes):
            gt_composite[gt[c] > 0] = colors[c]
        axes[-1].imshow(gt_composite)
        axes[-1].set_title("ground truth")
        axes[-1].axis("off")
    else:
        axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{i:04d}.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved sample_{i:04d}.png")

print(f"Done. {len(results)} maps in {out_dir}/")