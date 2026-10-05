#!/usr/bin/env python
"""Fix hardcoded paths in the nuscenes pkl files."""
import pickle

OLD_PREFIX = "/drive_dataset"
NEW_PREFIX = "/dataset"

pkl_files = [
    "/home/bevfusion/data/nuscenes/nuscenes_infos_train.pkl",
    "/home/bevfusion/data/nuscenes/nuscenes_infos_val.pkl",
    "/home/bevfusion/data/nuscenes/nuscenes_infos_test.pkl",
    "/home/bevfusion/data/nuscenes/nuscenes_dbinfos_train.pkl",
]

for pkl_file in pkl_files:
    print(f"Processing {pkl_file}...", end=" ")
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        count = 0

        # Collect all items that may have path fields
        items_to_fix = []

        if isinstance(data, dict) and "infos" in data:
            items_to_fix.extend(data["infos"])
        elif isinstance(data, list):
            items_to_fix.extend(data)
        elif isinstance(data, dict):
            # dbinfos: {class: [{path:...}, ...]}
            for cls_list in data.values():
                if isinstance(cls_list, list):
                    items_to_fix.extend(cls_list)

        for info in items_to_fix:
            for key in ("lidar_path", "data_path", "path"):
                val = info.get(key, "")
                if OLD_PREFIX in val:
                    info[key] = val.replace(OLD_PREFIX, NEW_PREFIX)
                    count += 1
            for cam_val in info.get("cams", {}).values():
                dp = cam_val.get("data_path", "")
                if OLD_PREFIX in dp:
                    cam_val["data_path"] = dp.replace(OLD_PREFIX, NEW_PREFIX)
                    count += 1
            for sweep in info.get("sweeps", []):
                for key in ("lidar_path", "data_path"):
                    val = sweep.get(key, "")
                    if OLD_PREFIX in val:
                        sweep[key] = val.replace(OLD_PREFIX, NEW_PREFIX)
                        count += 1

        with open(pkl_file, "wb") as f:
            pickle.dump(data, f)
        print(f"fixed {count} paths")
    except Exception as e:
        print(f"ERROR: {e}")