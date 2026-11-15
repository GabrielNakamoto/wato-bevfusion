import pytest
import torch

from mmdet3d.models.heads.segm.vanilla import (
    BEVSegmentationHead,
    sigmoid_focal_loss,
    soft_dice_loss,
)

GRID = dict(
    input_scope=[(-50.0, 50.0, 1.0), (-50.0, 50.0, 1.0)],
    output_scope=[(-50.0, 50.0, 1.0), (-50.0, 50.0, 1.0)],
)


def make_head(**kwargs):
    defaults = dict(
        in_channels=8,
        grid_transform=GRID,
        classes=["a", "b"],
        loss="focal",
    )
    defaults.update(kwargs)
    return BEVSegmentationHead(**defaults)


def test_loss_keys_and_finiteness():
    head = make_head(normalize_by_positives=True, dice_weight=0.5)
    head.train()
    x = torch.randn(2, 8, 100, 100)
    target = torch.zeros(2, 2, 100, 100)
    target[:, 0, 40:60, 40:60] = 1
    losses = head(x, target)
    assert set(losses) == {"a/focal", "b/focal"}
    for value in losses.values():
        assert torch.isfinite(value)


def test_focal_is_lower_for_correct_predictions():
    head = make_head(normalize_by_positives=True)
    target = torch.zeros(1, 100, 100)
    target[0, 40:60, 40:60] = 1
    good = torch.where(target > 0, torch.full_like(target, 10.0), torch.full_like(target, -10.0))
    bad = -good
    assert head._class_loss(good, target) < head._class_loss(bad, target)


def test_positive_normalisation_respects_cap():
    head = make_head(normalize_by_positives=True, normalize_cap=7.0)
    target = torch.zeros(1, 100, 100)
    target[0, 10:12, 10:12] = 1  # 4 positives -> raw ratio 2500, capped at 7
    logits = torch.randn(1, 100, 100)
    raw = sigmoid_focal_loss(logits, target, gamma=2.0, reduction="none")
    expected = raw.mean() * 7.0
    assert float(head._class_loss(logits, target)) == pytest.approx(
        float(expected), rel=1e-5
    )


def test_class_weights_scale_loss():
    plain = make_head()
    weighted = make_head(class_weights=[1.0, 2.0])
    target = torch.zeros(1, 2, 100, 100)
    target[:, 0, 40:60, 40:60] = 1
    logits = torch.randn(1, 2, 100, 100)
    plain_loss = plain._class_loss(logits[:, 1], target[:, 1], index=1)
    weighted_loss = weighted._class_loss(logits[:, 1], target[:, 1], index=1)
    assert float(weighted_loss) == pytest.approx(2.0 * float(plain_loss), rel=1e-5)


def test_dice_rewards_overlap():
    target = torch.zeros(1, 50, 50)
    target[0, 10:20, 10:20] = 1
    perfect = target.clone()
    empty = torch.zeros_like(target)
    # eps=1.0 smooths small masks; the ordering is what matters
    assert soft_dice_loss(perfect, target).item() < 1e-5
    assert soft_dice_loss(empty, target).item() == pytest.approx(1.0, abs=0.02)


def test_hi_res_branch_keeps_output_shape():
    head = make_head(hi_res=dict(enabled=True, mid_channels=4, dilation=2))
    head.eval()
    x = torch.randn(1, 8, 180, 180)
    with torch.no_grad():
        out = head(x)
    assert out.shape == (1, 2, 100, 100)  # GRID is 1 m/cell
    assert head.classifier[0].in_channels == 8  # classifier untouched
    assert head.hi is not None


def test_hi_res_branch_is_identity_at_init():
    """A warm-started checkpoint must evaluate unchanged when the branch is added."""
    head = make_head(hi_res=dict(enabled=True, mid_channels=4, dilation=2))
    head.eval()
    x = torch.randn(1, 8, 180, 180)
    with torch.no_grad():
        grad = head.hi(torch.randn(1, 8, 360, 360))
        out_with = head(x)
    assert torch.count_nonzero(grad) == 0
    assert head.hi[-1].weight.abs().sum().item() == 0.0
    assert torch.isfinite(out_with).all()


def test_default_head_matches_upstream_behaviour():
    """Normalisation and dice must be opt-in."""
    head = make_head()
    assert head.normalize_by_positives is False
    assert head.dice_weight == 0.0
    assert head.hi is None
    assert head.use_class_weights is False
