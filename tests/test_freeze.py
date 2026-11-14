import pytest
import torch
from torch import nn

from mmdet3d.models.utils.freeze import (
    build_param_groups,
    group_lrs,
    module_matches,
    param_group_name,
    sanity_check_groups,
    set_bn_eval,
    set_trainable,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                "camera": nn.ModuleDict(
                    {
                        "backbone": nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4)),
                        "neck": nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4)),
                        "vtransform": nn.Conv2d(4, 4, 1),
                    }
                ),
                "lidar": nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4)),
            }
        )
        self.fuser = nn.Conv2d(8, 4, 1)
        self.decoder = nn.ModuleDict(
            {"backbone": nn.Conv2d(4, 4, 1), "neck": nn.Conv2d(4, 4, 1)}
        )
        self.heads = nn.ModuleDict(
            {"map": nn.Conv2d(4, 6, 1), "object": nn.Conv2d(4, 8, 1)}
        )


def test_module_matching_is_path_aware():
    assert module_matches("heads.map", "heads")
    assert module_matches("heads", "heads")
    assert not module_matches("heads_up", "heads")
    assert module_matches("encoders.camera.backbone.0", "encoders.camera.backbone")


def test_group_names_are_ordered_by_specificity():
    assert param_group_name("encoders.camera.backbone.0.weight") == "camera_backbone"
    assert param_group_name("encoders.camera.neck.0.weight") == "camera_neck"
    assert param_group_name("encoders.lidar.0.weight") == "lidar"
    assert param_group_name("heads.map.weight") == "heads"
    assert param_group_name("something.else") == "other"


def test_set_trainable_only_enables_specs():
    model = TinyModel()
    set_trainable(model, ["heads"])
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(name.startswith("heads.") for name in trainable)

    set_trainable(model, ["heads", "decoder"])
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("decoder.") for name in trainable)
    assert not any(name.startswith("encoders.") for name in trainable)


def test_set_bn_eval_only_touches_matching_bn():
    model = TinyModel()
    model.train()
    set_bn_eval(model, ["encoders.lidar"])
    assert model.encoders.lidar[1].training is False
    # camera BNs untouched
    assert model.encoders.camera.backbone[1].training is True


def test_param_groups_cover_every_parameter_once():
    model = TinyModel()
    set_trainable(model, ["heads", "decoder"])
    groups = build_param_groups(model, base_lr=1e-4, weight_decay=0.01, lr_mults={"heads": 3.0})
    names = [g["name"] for g in groups]
    assert names == sorted(names, key=names.index)
    assert "heads" in names and "decoder" in names

    class Opt:
        param_groups = groups

    sanity_check_groups(model, Opt())
    assert group_lrs(Opt())["heads"] == pytest.approx(3e-4)


def test_frozen_parameters_stay_in_groups():
    model = TinyModel()
    set_trainable(model, ["heads"])
    groups = build_param_groups(model, base_lr=1e-4)
    total = sum(p.numel() for p in model.parameters())
    grouped = sum(p.numel() for g in groups for p in g["params"])
    assert total == grouped
