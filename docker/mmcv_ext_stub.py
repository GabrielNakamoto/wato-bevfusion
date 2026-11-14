"""Stub for ``mmcv._ext`` when mmcv is installed with ``MMCV_WITH_OPS=0``.

BEVFusion's training path uses mmcv's Python ops only; the CUDA kernels come
from ``mmdet3d.ops`` (spconv, bev_pool, voxelization). Any code path that truly
needs the compiled mmcv extension should fail loudly rather than silently
returning zeros.
"""


def __getattr__(name):
    raise AttributeError(
        f"mmcv._ext.{name} is unavailable: mmcv was built with MMCV_WITH_OPS=0. "
        "Install mmcv from source with ops if this path is required."
    )
