"""Trainable-parameter and stage control for multi-stage fine-tuning.

The public BEVFusion tree has no freeze/unfreeze machinery: configs that set
``frozen_modules`` / ``trainable_modules`` are silently ignored. These helpers
make the policy explicit and re-appliable, because ``nn.Module.train()`` is
called at the start of every epoch and would otherwise re-enable BatchNorm
statistics updates inside frozen subtrees.

Conventions
-----------
* A *spec* is a dotted module prefix (``"encoders.lidar"``) or an exact module
  name (``"heads"``). Matching is by dotted path, so ``"heads"`` covers
  ``heads.map`` and ``heads.object``.
* Parameter groups are tagged with a stable ``name`` so a stage hook can set a
  learning rate per region without rebuilding the optimizer.
"""
from typing import Dict, Iterable, List, Sequence

import torch.nn as nn

try:  # torch >= 1.12
    _BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
except AttributeError:  # pragma: no cover
    _BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

# Longest prefix first: "encoders.camera.backbone" must win over "encoders.camera".
GROUP_PREFIXES = (
    ("heads", "heads"),
    ("decoder", "decoder"),
    ("encoders.camera.backbone", "camera_backbone"),
    ("encoders.camera.neck", "camera_neck"),
    ("encoders.camera.vtransform", "camera_vtransform"),
    ("encoders.camera", "camera"),
    ("encoders.lidar", "lidar"),
    ("encoders.radar", "radar"),
    ("fuser", "fuser"),
)

# Deterministic order; the optimizer and the lr hook both rely on it.
GROUP_ORDER = (
    "heads",
    "decoder",
    "camera_backbone",
    "camera_neck",
    "camera_vtransform",
    "camera",
    "lidar",
    "radar",
    "fuser",
    "other",
)


def module_matches(qualified_name: str, spec: str) -> bool:
    """True when ``qualified_name`` is ``spec`` or lives under it."""
    return qualified_name == spec or qualified_name.startswith(spec + ".")


def param_group_name(qualified_name: str) -> str:
    for prefix, group in GROUP_PREFIXES:
        if module_matches(qualified_name, prefix):
            return group
    return "other"


def freeze_all(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def set_trainable(module: nn.Module, specs: Sequence[str]) -> None:
    """Freeze everything, then re-enable parameters under ``specs``."""
    freeze_all(module)
    if not specs:
        return
    specs = tuple(specs)
    for name, param in module.named_parameters():
        if any(module_matches(name, spec) for spec in specs):
            param.requires_grad_(True)


def set_bn_eval(module: nn.Module, specs: Sequence[str]) -> None:
    """Force BatchNorm layers under ``specs`` into eval mode (frozen statistics)."""
    if not specs:
        return
    specs = tuple(specs)
    for name, submodule in module.named_modules():
        if isinstance(submodule, _BN_TYPES):
            if any(module_matches(name, spec) for spec in specs):
                submodule.eval()


def trainable_parameter_counts(module: nn.Module) -> Dict[str, int]:
    counts = {name: 0 for name in GROUP_ORDER}
    for name, param in module.named_parameters():
        if param.requires_grad:
            counts[param_group_name(name)] += param.numel()
    return {k: v for k, v in counts.items() if v}


def build_param_groups(
    module: nn.Module,
    base_lr: float,
    weight_decay: float = 0.01,
    lr_mults: Dict[str, float] = None,
) -> List[dict]:
    """One AdamW group per region, tagged with a stable ``name``.

    ``lr`` is the group's stage-independent base; the staged lr hook overrides
    it every iteration from the active stage. ``lr_mults`` is only a starting
    point so that a run without the hook still trains sanely.
    """
    lr_mults = lr_mults or {}
    buckets: Dict[str, List[nn.Parameter]] = {name: [] for name in GROUP_ORDER}
    for name, param in module.named_parameters():
        buckets[param_group_name(name)].append(param)

    groups = []
    for name in GROUP_ORDER:
        params = buckets[name]
        if not params:
            continue
        groups.append(
            dict(
                name=name,
                params=params,
                lr=float(base_lr) * float(lr_mults.get(name, 1.0)),
                weight_decay=float(weight_decay),
            )
        )
    if not groups:
        raise RuntimeError("build_param_groups found no parameters")
    return groups


def group_lrs(optimizer) -> Dict[str, float]:
    return {g.get("name", "other"): g["lr"] for g in optimizer.param_groups}


def sanity_check_groups(module: nn.Module, optimizer) -> None:
    """Assert every parameter appears in exactly one optimizer group."""
    seen = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in seen:
                raise RuntimeError(
                    f"parameter in two groups: {group.get('name')} and {seen[id(param)]}"
                )
            seen[id(param)] = group.get("name")
    missing = [
        name
        for name, param in module.named_parameters()
        if id(param) not in seen
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} parameters missing from optimizer, e.g. {missing[:3]}")
