import pytest

from mmdet3d.core.hook.staged import StagedTrainableHook

STAGES = [
    dict(until_epoch=2, trainable=["heads"], lr={"heads": 3e-4}),
    dict(
        until_epoch=5,
        trainable=["heads", "decoder"],
        lr={"heads": 1e-4, "decoder": 1e-4},
    ),
    dict(
        until_epoch=8,
        trainable=["heads", "decoder", "lidar"],
        lr={"heads": 3e-5, "decoder": 3e-5, "lidar": 1e-5},
        cosine=False,
    ),
]


class FakeOptimizer:
    def __init__(self, names):
        self.param_groups = [dict(name=name, lr=0.0) for name in names]


class FakeRunner:
    def __init__(self, epoch, iter_=0):
        self.epoch = epoch
        self.iter = iter_
        self.optimizer = FakeOptimizer(["heads", "decoder", "lidar"])


def test_stage_boundaries_are_half_open():
    hook = StagedTrainableHook(stages=STAGES)
    assert hook.stage_index(0) == 0
    assert hook.stage_index(1) == 0
    assert hook.stage_index(2) == 1
    assert hook.stage_index(4) == 1
    assert hook.stage_index(5) == 2
    assert hook.stage_index(99) == 2


def test_frozen_groups_get_zero_lr():
    hook = StagedTrainableHook(stages=STAGES)
    lrs = hook.get_lr(FakeRunner(epoch=0), None)
    assert lrs[0] > 0
    assert lrs[1] == 0.0
    assert lrs[2] == 0.0


def test_stage_two_enables_decoder_only():
    hook = StagedTrainableHook(stages=STAGES)
    lrs = hook.get_lr(FakeRunner(epoch=2), None)
    assert lrs[0] > 0
    assert lrs[1] > 0
    assert lrs[2] == 0.0


def test_cosine_decays_within_stage():
    hook = StagedTrainableHook(stages=STAGES)
    early = hook.get_lr(FakeRunner(epoch=2), None)[0]
    late = hook.get_lr(FakeRunner(epoch=4), None)[0]
    assert early > late


def test_cosine_disabled_holds_lr():
    hook = StagedTrainableHook(stages=STAGES)
    first = hook.get_lr(FakeRunner(epoch=5), None)[2]
    last = hook.get_lr(FakeRunner(epoch=7), None)[2]
    assert first == pytest.approx(last)


def test_warmup_scales_first_iterations():
    hook = StagedTrainableHook(stages=STAGES, warmup_iters=100, warmup_ratio=0.1)
    start = hook.get_lr(FakeRunner(epoch=0, iter_=0), None)[0]
    mid = hook.get_lr(FakeRunner(epoch=0, iter_=50), None)[0]
    end = hook.get_lr(FakeRunner(epoch=0, iter_=100), None)[0]
    assert start < mid < end
