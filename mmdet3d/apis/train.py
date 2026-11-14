import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmdet3d.runner import CustomEpochBasedRunner

from mmdet3d.core.hook import StagedTrainableHook  # noqa: F401  (registry side effect)
from mmdet3d.models.utils import build_param_groups, sanity_check_groups
from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


def _install_sync_params_shim() -> None:
    """Torch 2.9+ removed ``DistributedDataParallel._sync_params``, which
    mmcv 1.4 calls from ``train_step`` / ``val_step``. Broadcast once for
    parameters and every step for buffers (frozen BatchNorm stats)."""
    from mmcv.parallel import MMDistributedDataParallel

    if hasattr(MMDistributedDataParallel, "_sync_params"):
        return

    def _sync_params_compat(self) -> None:
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        with torch.no_grad():
            if not getattr(self, "_wato_params_synced", False):
                for parameter in self.module.parameters():
                    torch.distributed.broadcast(parameter.data, src=0)
                self._wato_params_synced = True
            for buffer in self.module.buffers():
                torch.distributed.broadcast(buffer.data, src=0)

    MMDistributedDataParallel._sync_params = _sync_params_compat


def build_staged_optimizer(module, optimizer_cfg):
    """AdamW over named parameter groups, one group per model region."""
    opt_type = str(optimizer_cfg.get("type", "AdamW")).lower()
    if opt_type != "adamw":
        raise ValueError(f"staged training expects AdamW, got {optimizer_cfg.get('type')}")
    base_lr = float(optimizer_cfg.get("lr", 1e-4))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
    groups = build_param_groups(
        module,
        base_lr=base_lr,
        weight_decay=weight_decay,
        lr_mults=optimizer_cfg.get("lr_mults", None),
    )
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


_install_sync_params_shim()


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            None,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]

    # put model on gpus
    stages = getattr(model, "training_stages", None)
    find_unused_parameters = cfg.get("find_unused_parameters", stages is not None)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )

    # build runner
    if stages:
        lr_cfg = cfg.get("lr_config", None) or {}
        if lr_cfg.get("type") == "StagedTrainableHook":
            model_bounds = [int(s["until_epoch"]) for s in stages]
            hook_bounds = [int(s["until_epoch"]) for s in lr_cfg["stages"]]
            if model_bounds != hook_bounds:
                raise ValueError(
                    "model.training_stages and lr_config.stages disagree: "
                    f"{model_bounds} vs {hook_bounds}"
                )
        optimizer = build_staged_optimizer(model.module, cfg.optimizer)
        sanity_check_groups(model.module, optimizer)
        logger.info(
            "Staged optimizer groups: "
            + ", ".join(g["name"] for g in optimizer.param_groups)
        )
    else:
        optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )
    
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, [("train", 1)])
