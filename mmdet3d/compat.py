"""Runtime shims for torch versions newer than this tree was written against.

``torch.nn.parallel._functions._get_stream`` now requires a ``torch.device``,
but ``DataParallel``/``DistributedDataParallel`` normalise ``device_ids`` to
ints. Every ``scatter`` call therefore raises ``AttributeError: 'int' object has
no attribute 'type'`` (torch 2.9). Patch once, at import time, from both the
training and the eval entry points.
"""
import torch


def patch_parallel_stream() -> None:
    import torch.nn.parallel._functions as functions

    original = getattr(functions, "_get_stream", None)
    if original is None or getattr(functions, "_wato_stream_patched", False):
        return

    def _get_stream(device):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return original(device)

    functions._get_stream = _get_stream
    functions._wato_stream_patched = True


patch_parallel_stream()
