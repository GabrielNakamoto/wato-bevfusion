from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS

__all__ = ["BEVSegmentationHead"]


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def soft_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1.0,
) -> torch.Tensor:
    """Per-sample soft dice over the spatial dims of one class channel."""
    probs = probs.float().flatten(1)
    targets = targets.float().flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


@HEADS.register_module()
class BEVSegmentationHead(nn.Module):
    """Binary per-class BEV map head.

    Two knobs matter for rare/thin classes and are off by default so existing
    checkpoints keep their behaviour:

    ``normalize_by_positives``
        Divide the focal sum by the number of positive pixels instead of by all
        map pixels. A mean over 200x200 cells lets a stop-line positive
        contribute ~1/40000 of the loss.

    ``hi_res``
        Residual 2x upsampled dilated branch before the classifier. Adds
        boundary detail without changing the output resolution (200x200 @ 0.5 m)
        and keeps the classifier weights intact, which matters when warm
        starting from a checkpoint.
    """

    def __init__(
        self,
        in_channels: int,
        grid_transform: Dict[str, Any],
        classes: List[str],
        loss: str,
        class_weights: Optional[List[float]] = None,
        gamma: float = 2.0,
        alpha: float = -1.0,
        dice_weight: float = 0.0,
        normalize_by_positives: bool = False,
        hi_res: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.dice_weight = float(dice_weight)
        self.normalize_by_positives = bool(normalize_by_positives)

        if class_weights is None:
            self.register_buffer("class_weight_tensor", torch.zeros(0))
            self.use_class_weights = False
        else:
            assert len(class_weights) == len(classes)
            self.register_buffer(
                "class_weight_tensor",
                torch.tensor(class_weights, dtype=torch.float32),
            )
            self.use_class_weights = True

        self.transform = BEVGridTransform(**grid_transform)

        self.hi: Optional[nn.Module] = None
        if hi_res and hi_res.get("enabled", True):
            mid = int(hi_res.get("mid_channels", 128))
            dilation = int(hi_res.get("dilation", 2))
            self.hi = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    mid,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(mid),
                nn.ReLU(True),
                nn.Conv2d(
                    mid,
                    mid,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm2d(mid),
                nn.ReLU(True),
                nn.Conv2d(mid, in_channels, 1),
            )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),
        )

    def _class_loss(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=logit.dtype)
        if self.loss == "xent":
            raw = sigmoid_xent_loss(logit, target, reduction="none")
            if self.normalize_by_positives:
                return raw.sum() / target.sum().clamp(min=1.0)
            return raw.mean()
        if self.loss == "focal":
            raw = sigmoid_focal_loss(
                logit, target, alpha=self.alpha, gamma=self.gamma, reduction="none"
            )
            if self.normalize_by_positives:
                return raw.sum() / target.sum().clamp(min=1.0)
            return raw.mean()
        raise ValueError(f"unsupported loss: {self.loss}")

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = self.transform(x)
        if self.hi is not None:
            h = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            h = self.hi(h)
            h = F.interpolate(
                h, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + h
        x = self.classifier(x)

        if self.training:
            assert target is not None
            num_classes = len(self.classes)
            losses = {}
            for index, name in enumerate(self.classes):
                loss = self._class_loss(x[:, index], target[:, index])
                if self.dice_weight > 0:
                    dice = soft_dice_loss(
                        torch.sigmoid(x[:, index]), target[:, index]
                    ).mean()
                    loss = loss + self.dice_weight * dice
                if self.use_class_weights:
                    loss = loss * self.class_weight_tensor[index].to(loss.dtype)
                losses[f"{name}/{self.loss}"] = loss / num_classes
            return losses
        else:
            return torch.sigmoid(x)
