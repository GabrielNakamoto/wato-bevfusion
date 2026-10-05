#!/usr/bin/env python
"""Create nuscenes ground truth database for test."""
import sys
sys.path.insert(0, "/home/bevfusion")
sys.path.insert(0, "/home/bevfusion/tools")

import os

# The actual file was created with _radar suffix
root_path = sys.argv[1] if len(sys.argv) > 1 else "/dataset"
info_path = sys.argv[2] if len(sys.argv) > 2 else "/dataset/nuscenes_infos_test_radar.pkl"

from data_converter.create_gt_database import create_groundtruth_database

create_groundtruth_database(
    "NuScenesDataset",
    root_path,
    "nuscenes",
    info_path,
    load_augmented=None
)
print("ground truth database done")