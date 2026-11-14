"""Staged fine-tuning: per-region learning rates and freeze/unfreeze transitions.

mmcv's built-in lr policies assume one global schedule and a static parameter
set. A staged run needs both to change at epoch boundaries, so this hook owns
three things:

* which subtrees are trainable in the current stage (delegated to
  ``model.set_stage``),
* a per-parameter-group learning rate, with a cosine decay *within* each stage,
* optional linear warmup over the first N iterations.

The optimizer must be built by ``mmdet3d.apis.train.build_staged_optimizer`` so
that every parameter group carries a ``name`` matching the stage ``lr`` keys.
"""
import math

from mmcv.runner import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class StagedTrainableHook(LrUpdaterHook):
    def __init__(
        self,
        stages,
        by_epoch=True,
        warmup_iters=0,
        warmup_ratio=0.1,
        **kwargs,
    ):
        assert isinstance(stages, (list, tuple)) and len(stages) > 0
        # Warmup is handled here so that it composes with per-group lrs.
        super().__init__(by_epoch=by_epoch, warmup=None, **kwargs)
        self.stages = list(stages)
        self.warmup_iters = int(warmup_iters)
        self.warmup_ratio = float(warmup_ratio)
        self._stage_idx = None

    def stage_index(self, epoch: int) -> int:
        for index, stage in enumerate(self.stages):
            if epoch < int(stage["until_epoch"]):
                return index
        return len(self.stages) - 1

    def before_run(self, runner):
        super().before_run(runner)
        if self._stage_idx is None:
            self._apply_stage(runner, self.stage_index(runner.epoch))

    def before_train_epoch(self, runner):
        index = self.stage_index(runner.epoch)
        if index != self._stage_idx:
            self._apply_stage(runner, index)

    def _apply_stage(self, runner, index: int) -> None:
        stage = self.stages[index]
        model = getattr(runner.model, "module", runner.model)
        if not hasattr(model, "set_stage"):
            raise RuntimeError(
                "StagedTrainableHook requires a model with set_stage(); "
                "add training_stages to the model config"
            )
        model.set_stage(stage)
        self._stage_idx = index
        counts = model.stage_summary() if hasattr(model, "stage_summary") else {}
        pretty = ", ".join(f"{k}={v/1e6:.1f}M" for k, v in sorted(counts.items()))
        runner.logger.info(
            f"[stage {index}] epochs [{self._stage_start(index)}, "
            f"{stage['until_epoch']}) trainable: {pretty or 'none'}; "
            f"lr={stage.get('lr', {})}"
        )

    def _stage_start(self, index: int) -> int:
        return 0 if index == 0 else int(self.stages[index - 1]["until_epoch"])

    def _cosine_factor(self, runner, index: int) -> float:
        stage = self.stages[index]
        if not stage.get("cosine", True):
            return 1.0
        start = self._stage_start(index)
        span = max(1, int(stage["until_epoch"]) - start)
        # Full lr on the first epoch of the stage, ~0 on the last: never exactly
        # zero, otherwise the final epoch of every stage trains at lr=0.
        progress = (runner.epoch - start) / span
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_lr(self, runner, base_lr):
        index = self.stage_index(runner.epoch)
        stage = self.stages[index]
        factor = self._cosine_factor(runner, index)
        stage_lrs = stage.get("lr", {})
        lrs = []
        for group in runner.optimizer.param_groups:
            name = group.get("name", "other")
            value = stage_lrs.get(name, None)
            # A group absent from the stage lr map is frozen for that stage.
            lrs.append(0.0 if value is None else float(value) * factor)
        if self.warmup_iters > 0 and runner.iter < self.warmup_iters:
            ratio = self.warmup_ratio + (1.0 - self.warmup_ratio) * (
                runner.iter / max(1, self.warmup_iters)
            )
            lrs = [lr * ratio for lr in lrs]
        return lrs
