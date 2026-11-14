"""Stub for ``mmcv._ext`` when mmcv is installed with ``MMCV_WITH_OPS=0``.

mmcv's pure-python op wrappers still import ``mmcv._ext`` and assert that the
compiled symbols exist. BEVFusion's training path does not use those ops (its
CUDA kernels come from ``mmdet3d.ops``), but the import must succeed.

Each attribute resolves to a callable that raises on *use*, so an accidental
dependency fails loudly at the call site instead of silently returning zeros.
"""


def _unavailable(name):
    def _fail(*args, **kwargs):
        raise NotImplementedError(
            f"mmcv._ext.{name} is unavailable: mmcv was built with MMCV_WITH_OPS=0. "
            "Install mmcv from source with CUDA ops if this op is required."
        )

    _fail.__name__ = name
    return _fail


def __getattr__(name):
    return _unavailable(name)
