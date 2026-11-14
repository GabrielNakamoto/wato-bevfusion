#!/usr/bin/env python
"""Apply the site-package patches the pinned stack needs.

All three are version-pinned (mmcv 1.4.0, mmdet 2.20.0, nuscenes-devkit) and
idempotent: re-running is a no-op. Each patch prints what it did so a failed
environment build is obvious instead of surfacing as a confusing runtime error.

  1. nuscenes-devkit: shapely 2.x returns a GeometryCollection without
     iteration; ``mask_for_lines`` must use ``.geoms``.
  2. mmcv: TextLoggerHook raises KeyError('data_time') after an eval resets the
     log buffer.
  3. mmdet: ``multi_gpu_test`` forwards DataContainer objects to DDP; BEVFusion
     expects already-scattered inputs.
"""
import argparse
import glob
import os
import re


def site_packages(env: str) -> str:
    matches = glob.glob(os.path.join(env, "lib", "python3.*", "site-packages"))
    if not matches:
        raise SystemExit(f"no site-packages under {env}")
    return sorted(matches)[-1]


def patch(path: str, transform, marker: str, name: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"[missing] {name}: {path}")
    with open(path) as handle:
        source = handle.read()
    if marker in source:
        print(f"[skip] {name} (already patched)")
        return
    patched = transform(source)
    if patched == source:
        raise SystemExit(f"[fail] {name}: pattern not found in {path}")
    with open(path, "w") as handle:
        handle.write(patched)
    print(f"[ok] {name}")


def patch_nuscenes(source: str) -> str:
    anchor = source.find("def mask_for_lines")
    if anchor < 0:
        return source
    head, tail = source[:anchor], source[anchor:]
    tail = re.sub(
        r"(\n\s*)for line in lines:",
        r"\1for line in getattr(lines, 'geoms', lines):",
        tail,
        count=1,
    )
    return head + tail


def patch_mmcv_logger(source: str) -> str:
    pattern = re.compile(
        r"f'data_time: \{data_time:\.3f\}, '\s*\n(\s*)f'memory: \{memory\}, '"
    )

    def repl(match):
        indent = match.group(1)
        return (
            'f\'data_time: {log_dict.get("data_time", 0.0):.3f}, \'\n'
            + indent
            + 'f\'memory: {log_dict.get("memory", 0)}, \''
        )

    return pattern.sub(repl, source, count=1)


def patch_mmdet_ddp(source: str) -> str:
    pattern = re.compile(
        r"(?P<ind>[ \t]*)result = model\(return_loss=False, rescale=True, \*\*data\)"
    )

    def repl(match):
        indent = match.group("ind")
        return (
            f"{indent}if hasattr(model, 'scatter') and getattr(model, 'device_ids', None):\n"
            f"{indent}    inputs, kwargs = model.scatter(\n"
            f"{indent}        (), dict(return_loss=False, rescale=True, **data),\n"
            f"{indent}        model.device_ids)\n"
            f"{indent}    result = model.module(*inputs[0], **kwargs[0])\n"
            f"{indent}else:\n"
            f"{indent}    result = model(return_loss=False, rescale=True, **data)"
        )

    return pattern.sub(repl, source, count=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="conda env prefix")
    args = parser.parse_args()

    packages = site_packages(args.env)
    print(f"site-packages: {packages}")

    patch(
        os.path.join(packages, "nuscenes", "map_expansion", "map_api.py"),
        patch_nuscenes,
        marker="getattr(lines, 'geoms', lines)",
        name="nuscenes-devkit shapely 2.x",
    )
    patch(
        os.path.join(packages, "mmcv", "runner", "hooks", "logger", "text.py"),
        patch_mmcv_logger,
        marker='log_dict.get("data_time"',
        name="mmcv TextLoggerHook data_time",
    )
    patch(
        os.path.join(packages, "mmdet", "apis", "test.py"),
        patch_mmdet_ddp,
        marker="model.module(*inputs[0], **kwargs[0])",
        name="mmdet multi_gpu_test DataContainer scatter",
    )


if __name__ == "__main__":
    main()
