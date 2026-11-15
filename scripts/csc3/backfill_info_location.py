#!/usr/bin/env python
"""Add muScenes ``location`` to existing info pickles.

The BEVFusion-R converter merge dropped ``location`` from the info dict, but
``LoadBEVSegmentation`` needs it to pick the map JSON. Rather than re-running
the ~90 minute conversion, backfill it from the scene's log.
"""
import argparse
import pickle
from collections import Counter

from nuscenes import NuScenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--strip-radar", action="store_true")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.data, verbose=False)
    scene_location = {
        scene["token"]: nusc.get("log", scene["log_token"])["location"]
        for scene in nusc.scene
    }
    token_location = {
        sample["token"]: scene_location[sample["scene_token"]]
        for sample in nusc.sample
    }

    for split in ("train", "val"):
        path = f"{args.data}/nuscenes_infos_{split}.pkl"
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        infos = payload["infos"] if isinstance(payload, dict) else payload
        for info in infos:
            info["location"] = token_location[info["token"]]
            if args.strip_radar:
                info.pop("radars", None)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)
        counts = Counter(info["location"] for info in infos)
        print(f"{split}: {len(infos)} infos, locations={dict(counts)}")


if __name__ == "__main__":
    main()
