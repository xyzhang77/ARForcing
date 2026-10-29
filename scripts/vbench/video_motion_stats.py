"""Pixel-space motion stats on matching mp4s across video dirs (no models); run with python -m scripts.vbench.video_motion_stats."""

import argparse
import json
from pathlib import Path
from statistics import fmean, median

import cv2
import numpy as np


def read_gray(path, scale=4):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"Need >= 2 frames: {path}")
    return np.stack(frames)


def motion_stats(frames):
    residual = frames - frames.mean(0, keepdims=True)
    return {
        "frame_diff_mae": float(np.abs(np.diff(frames, axis=0)).mean()),
        "motion_ratio": float(np.square(residual).sum() / max(np.square(frames).sum(), 1e-12)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", required=True, help="name=path pairs; first one is the reference")
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    dirs = dict(item.split("=", 1) for item in args.dirs)
    names = set.intersection(*({p.name for p in Path(d).glob("*.mp4")} for d in dirs.values()))
    if not names:
        raise ValueError("No mp4 filenames shared by all dirs")
    names = sorted(names)
    per_video = {key: {} for key in dirs}
    for index, name in enumerate(names):
        for key, d in dirs.items():
            per_video[key][name] = motion_stats(read_gray(Path(d) / name))
        if index % 20 == 0:
            print(f"{index + 1}/{len(names)}", flush=True)
    ref = next(iter(dirs))
    summary = {}
    for key in dirs:
        diffs = [per_video[key][n]["frame_diff_mae"] for n in names]
        ratio_to_ref = [per_video[key][n]["frame_diff_mae"] / max(per_video[ref][n]["frame_diff_mae"], 1e-12) for n in names]
        summary[key] = {
            "frame_diff_mae_mean": fmean(diffs),
            "motion_ratio_mean": fmean(per_video[key][n]["motion_ratio"] for n in names),
            f"frame_diff_vs_{ref}_median": median(ratio_to_ref),
            f"less_motion_than_{ref}_frac": fmean(r < 1 for r in ratio_to_ref),
        }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8") as f:
        json.dump({"dirs": dirs, "num_videos": len(names), "summary": summary, "per_video": per_video},
                  f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
