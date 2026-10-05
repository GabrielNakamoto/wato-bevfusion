#!/usr/bin/env python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version="v1.0-trainval", dataroot="/dataset", verbose=False)
sd = nusc.get("sample_data", nusc.sample[0]["data"]["CAM_FRONT"])
cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
print("calibrated_sensor keys:", list(cs.keys()))