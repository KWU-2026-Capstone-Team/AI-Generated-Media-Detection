#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIR_NAMES = ["O_R", "O_S", "O_SR", "R_S", "R_SR", "S_SR"]


def topk_mean(x: np.ndarray, k_ratio: float) -> float:
    k = max(1, int(math.ceil(len(x) * k_ratio)))
    return float(np.mean(np.sort(x)[-k:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_csvs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="dinov2")
    ap.add_argument("--topk_ratio", type=float, default=0.2)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    merged_frame = outdir / f"{args.prefix}_frame_sims.csv"
    video_out = outdir / f"{args.prefix}_video_sims.csv"

    per_video = defaultdict(list)
    total_rows = 0
    header_written = False
    with merged_frame.open("w", newline="") as f_out:
        writer = None
        for frame_csv in args.frame_csvs:
            with Path(frame_csv).resolve().open(newline="") as f_in:
                reader = csv.DictReader(f_in)
                if writer is None:
                    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    key = (row["split"], row["method"], int(row["label"]), str(row["video_id"]).strip())
                    per_video[key].append([float(row[p]) for p in PAIR_NAMES])
                    total_rows += 1

    if not header_written:
        raise RuntimeError("No input rows found")

    with video_out.open("w", newline="") as f_v:
        fieldnames = ["split", "method", "label", "video_id"] + [f"mean_{n}" for n in PAIR_NAMES] + [
            f"median_{n}" for n in PAIR_NAMES
        ] + [f"topk_{n}" for n in PAIR_NAMES]
        writer = csv.DictWriter(f_v, fieldnames=fieldnames)
        writer.writeheader()
        for (split, method, label, vid), sim_list in per_video.items():
            arr = np.asarray(sim_list)
            row = {"split": split, "method": method, "label": label, "video_id": vid}
            for k, name in enumerate(PAIR_NAMES):
                row[f"mean_{name}"] = float(arr[:, k].mean())
                row[f"median_{name}"] = float(np.median(arr[:, k]))
                row[f"topk_{name}"] = float(topk_mean(arr[:, k], args.topk_ratio))
            writer.writerow(row)

    print(f"[rows] {total_rows}")
    print(f"[saved] {merged_frame}")
    print(f"[saved] {video_out}")


if __name__ == "__main__":
    main()
