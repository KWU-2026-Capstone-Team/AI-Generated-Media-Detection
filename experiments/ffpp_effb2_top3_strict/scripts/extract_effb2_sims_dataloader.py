#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset


PAIR_NAMES = ["O_R", "O_S", "O_SR", "R_S", "R_SR", "S_SR"]


def topk_mean(x: np.ndarray, k_ratio: float) -> float:
    k = max(1, int(math.ceil(len(x) * k_ratio)))
    return float(np.mean(np.sort(x)[-k:]))


class FourViewDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        with csv_path.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: str):
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return (
            row["split"],
            row["method"],
            int(row["label"]),
            str(row["video_id"]).strip(),
            row["frame"],
            self._load(row["path_O"]),
            self._load(row["path_R"]),
            self._load(row["path_S"]),
            self._load(row["path_SR"]),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--topk_ratio", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu") if args.ckpt else None
    model = timm.create_model("efficientnet_b2", pretrained=(ckpt is None), num_classes=0)
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)
    cfg = (ckpt or {}).get("cfg", resolve_data_config({}, model=model))
    tfm = create_transform(**cfg)

    dataset = FourViewDataset(Path(args.csv).resolve(), tfm)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    frame_out = outdir / "effb2_frame_sims.csv"
    video_out = outdir / "effb2_video_sims.csv"
    per_video = defaultdict(list)

    with frame_out.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["split", "method", "label", "video_id", "frame"] + PAIR_NAMES)
        writer.writeheader()

        for batch_idx, batch in enumerate(loader):
            splits, methods, labels, video_ids, frames, xO, xR, xS, xSR = batch
            bsz = xO.shape[0]
            x = torch.cat([xO, xR, xS, xSR], dim=0).to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                z = model(x)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-12)

            zO, zR, zS, zSR = z[:bsz], z[bsz : 2 * bsz], z[2 * bsz : 3 * bsz], z[3 * bsz : 4 * bsz]
            sims = torch.stack(
                [
                    (zO * zR).sum(dim=1),
                    (zO * zS).sum(dim=1),
                    (zO * zSR).sum(dim=1),
                    (zR * zS).sum(dim=1),
                    (zR * zSR).sum(dim=1),
                    (zS * zSR).sum(dim=1),
                ],
                dim=1,
            ).cpu().numpy()

            for i in range(bsz):
                row_out = {
                    "split": splits[i],
                    "method": methods[i],
                    "label": int(labels[i]),
                    "video_id": str(video_ids[i]).strip(),
                    "frame": frames[i],
                }
                for j, name in enumerate(PAIR_NAMES):
                    row_out[name] = float(sims[i, j])
                writer.writerow(row_out)
                per_video[(splits[i], methods[i], int(labels[i]), str(video_ids[i]).strip())].append(sims[i].tolist())

            if batch_idx % 200 == 0:
                print(f"[batch] {batch_idx} / approx {math.ceil(len(dataset) / args.batch)}", flush=True)

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
    print(f"[saved] {frame_out}")
    print(f"[saved] {video_out}")


if __name__ == "__main__":
    main()
