#!/usr/bin/env python
"""Write small train/val info pickles for smoke runs.

Strides across the full info list (rather than taking a prefix) so a smoke run
sees several scenes and all map classes.

    python scripts/csc3/make_mini_infos.py --root data/nuscenes --train 64 --val 32
"""
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/nuscenes")
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--val", type=int, default=32)
    return parser.parse_args()


def strided(infos, count):
    if count >= len(infos):
        return infos
    step = max(1, len(infos) // count)
    return infos[:: step][:count]


def main():
    args = parse_args()
    for split, count in (("train", args.train), ("val", args.val)):
        source = f"{args.root}/nuscenes_infos_{split}.pkl"
        with open(source, "rb") as handle:
            payload = pickle.load(handle)
        infos = payload["infos"] if isinstance(payload, dict) else payload
        subset = strided(infos, count)
        output = {
            "infos": subset,
            "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
        }
        target = f"{args.root}/nuscenes_infos_{split}_mini.pkl"
        with open(target, "wb") as handle:
            pickle.dump(output, handle)
        print(f"{split}: {len(infos)} -> {len(subset)} ({target})")


if __name__ == "__main__":
    main()
