#!/usr/bin/env python
"""Create nuscenes test data infos only."""
import sys
import os

# Add tools dir to path for imports
sys.path.insert(0, "/home/bevfusion")
from tools.data_converter import nuscenes_converter as nuscenes_converter

root_path = sys.argv[1] if len(sys.argv) > 1 else "/dataset"
max_sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

print(f"Creating test infos from {root_path} with max_sweeps={max_sweeps}")
nuscenes_converter.create_nuscenes_infos(
    root_path, "nuscenes", version="v1.0-test", max_sweeps=max_sweeps
)
print("Done!")