#!/usr/bin/env python
"""Pre-flight check for the staged dual-head model, before touching the dataset.

Validates, on the real GPU:
  * the model builds from the generated config (dual head + hi-res branch),
  * every stage produces the expected trainable-parameter set and BN modes,
  * the optimizer covers every parameter exactly once,
  * the staged lr hook emits one lr per group and zero for frozen regions,
  * the map head forward/backward works for each stage.

    python scripts/csc3/check_model.py configs/nuscenes/seg/resnet50_dualhead/a5_staged.yaml
"""
import argparse
import sys

import torch
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.core.hook.staged import StagedTrainableHook
from mmdet3d.models import build_model
from mmdet3d.models.utils import (
    build_param_groups,
    sanity_check_groups,
    trainable_parameter_counts,
)
from mmdet3d.utils import recursive_eval


class FakeRunner:
    def __init__(self, model, optimizer, epoch=0):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.iter = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    stages = cfg.model.get("training_stages")
    if not stages:
        sys.exit("config has no model.training_stages")
    if [s["until_epoch"] for s in stages] != [
        s["until_epoch"] for s in cfg.lr_config["stages"]
    ]:
        sys.exit("model.training_stages and lr_config.stages disagree")

    model = build_model(cfg.model)
    device = "cpu" if args.no_cuda else "cuda"
    model = model.to(device)

    base_lr = cfg.optimizer["lr"]
    optimizer = torch.optim.AdamW(
        build_param_groups(
            model,
            base_lr=base_lr,
            weight_decay=cfg.optimizer.get("weight_decay", 0.01),
            lr_mults=cfg.optimizer.get("lr_mults"),
        ),
        lr=base_lr,
    )
    sanity_check_groups(model, optimizer)
    print("optimizer groups:", [g["name"] for g in optimizer.param_groups])

    head = model.heads["map"]
    print("map head:", type(head).__name__, "hi_res:", head.hi is not None)
    print("loss:", head.loss, "normalize_by_positives:", head.normalize_by_positives,
          "dice:", head.dice_weight, "weights:", head.class_weight_tensor.tolist())

    for index, stage in enumerate(stages):
        model.set_stage(stage)
        model.train()
        counts = trainable_parameter_counts(model)
        total = sum(counts.values())
        print(f"\n[stage {index}] <{stage['until_epoch']} trainable {total/1e6:.1f}M {counts}")

        # BN policy: every BN under bn_eval must be in eval mode.
        for spec in stage.get("bn_eval", []):
            for name, module in model.named_modules():
                if name == spec or name.startswith(spec + "."):
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        assert not module.training, f"{name} BN should be eval in stage {index}"

        # Forward/backward through the map head only (the trunk path is covered
        # by the freeze unit tests and the dataset smoke run).
        feats = torch.randn(1, 512, 180, 180, device=device, requires_grad=True)
        target = torch.zeros(1, 6, 200, 200, device=device)
        target[:, :, 40:60, 40:60] = 1
        losses = head(feats, target)
        loss = sum(losses.values())
        loss.backward()

        missing = [
            name
            for name, param in head.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        assert not missing, f"head params without grad: {missing[:3]}"
        grad_params = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"  head loss {float(loss):.4f}  head params with grad {grad_params}")
        optimizer.zero_grad(set_to_none=True)

        runner = FakeRunner(model, optimizer, epoch=int(stage["until_epoch"]) - 1)
        hook = StagedTrainableHook(stages=stages, warmup_iters=0)
        lrs = hook.get_lr(runner, None)
        assert len(lrs) == len(optimizer.param_groups)
        for group, lr in zip(optimizer.param_groups, lrs):
            expected_frozen = group["name"] not in stage.get("lr", {})
            assert (lr == 0.0) == expected_frozen, (group["name"], lr, expected_frozen)
        print("  lrs:", {g["name"]: round(lr, 8) for g, lr in zip(optimizer.param_groups, lrs)})

    print("\nOK: staged model pre-flight passed")


if __name__ == "__main__":
    main()
