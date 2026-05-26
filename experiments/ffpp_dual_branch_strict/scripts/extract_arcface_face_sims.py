#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from insightface.app import FaceAnalysis


PAIR_NAMES = ["O_R", "O_S", "O_SR", "R_S", "R_SR", "S_SR"]


def topk_mean(x: np.ndarray, k_ratio: float) -> float:
    k = max(1, int(math.ceil(len(x) * k_ratio)))
    return float(np.mean(np.sort(x)[-k:]))


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def detect_face_box(detector: cv2.CascadeClassifier, gray: np.ndarray) -> tuple[int, int, int, int] | None:
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def square_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int, margin: float) -> tuple[int, int, int, int]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    s = max(w, h) * (1.0 + margin * 2.0)
    x1 = max(0, int(round(cx - s / 2.0)))
    y1 = max(0, int(round(cy - s / 2.0)))
    x2 = min(img_w, int(round(cx + s / 2.0)))
    y2 = min(img_h, int(round(cy + s / 2.0)))
    side = max(1, min(x2 - x1, y2 - y1))
    return x1, y1, x1 + side, y1 + side


def center_square(size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    s = max(1, min(w, h))
    x1 = max(0, (w - s) // 2)
    y1 = max(0, (h - s) // 2)
    return x1, y1, x1 + s, y1 + s


def resize_box(box: tuple[int, int, int, int], src_size: tuple[int, int], dst_size: tuple[int, int]) -> tuple[int, int, int, int]:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    x1, y1, x2, y2 = box
    rx = dst_w / float(src_w)
    ry = dst_h / float(src_h)
    out = (
        int(round(x1 * rx)),
        int(round(y1 * ry)),
        int(round(x2 * rx)),
        int(round(y2 * ry)),
    )
    x1o, y1o, x2o, y2o = out
    x1o = max(0, min(x1o, dst_w - 1))
    y1o = max(0, min(y1o, dst_h - 1))
    x2o = max(x1o + 1, min(x2o, dst_w))
    y2o = max(y1o + 1, min(y2o, dst_h))
    return x1o, y1o, x2o, y2o


def crop_resize_bgr(img: Image.Image, box: tuple[int, int, int, int], out_size: int = 112) -> np.ndarray:
    bgr = pil_to_bgr(img)
    x1, y1, x2, y2 = box
    patch = bgr[y1:y2, x1:x2]
    if patch.size == 0:
        h, w = bgr.shape[:2]
        cx1, cy1, cx2, cy2 = center_square((w, h))
        patch = bgr[cy1:cy2, cx1:cx2]
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--topk_ratio", type=float, default=0.2)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--face_margin", type=float, default=0.20)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    # When CUDA_VISIBLE_DEVICES is used per process, the visible GPU should be addressed as cuda:0.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    rec = app.models["recognition"]

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    frame_out = outdir / "face_arc_frame_sims.csv"
    video_out = outdir / "face_arc_video_sims.csv"
    per_video = defaultdict(list)

    with frame_out.open("w", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["split", "method", "label", "video_id", "frame", "face_detected"] + PAIR_NAMES,
        )
        writer.writeheader()

        for batch_idx, batch_rows in enumerate(batched(rows, args.batch)):
            o_imgs = [Image.open(r["path_O"]).convert("RGB") for r in batch_rows]

            crops_O, crops_R, crops_S, crops_SR = [], [], [], []
            meta = []
            for i, row in enumerate(batch_rows):
                img_O = o_imgs[i]
                img_R = Image.open(row["path_R"]).convert("RGB")
                img_S = Image.open(row["path_S"]).convert("RGB")
                img_SR = Image.open(row["path_SR"]).convert("RGB")

                gray_O = cv2.cvtColor(np.asarray(img_O), cv2.COLOR_RGB2GRAY)
                face_box = detect_face_box(detector, gray_O)
                detected = 1 if face_box is not None else 0
                if face_box is None:
                    box_O = center_square(img_O.size)
                else:
                    box_O = square_box(*face_box, img_O.size[0], img_O.size[1], margin=args.face_margin)

                box_R = resize_box(box_O, img_O.size, img_R.size)
                box_S = resize_box(box_O, img_O.size, img_S.size)
                box_SR = resize_box(box_O, img_O.size, img_SR.size)

                crops_O.append(crop_resize_bgr(img_O, box_O))
                crops_R.append(crop_resize_bgr(img_R, box_R))
                crops_S.append(crop_resize_bgr(img_S, box_S))
                crops_SR.append(crop_resize_bgr(img_SR, box_SR))
                meta.append((row, detected))

            zO = rec.get_feat(crops_O).astype(np.float32)
            zR = rec.get_feat(crops_R).astype(np.float32)
            zS = rec.get_feat(crops_S).astype(np.float32)
            zSR = rec.get_feat(crops_SR).astype(np.float32)

            def norm(x):
                return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

            zO, zR, zS, zSR = map(norm, [zO, zR, zS, zSR])
            sims = np.stack(
                [
                    np.sum(zO * zR, axis=1),
                    np.sum(zO * zS, axis=1),
                    np.sum(zO * zSR, axis=1),
                    np.sum(zR * zS, axis=1),
                    np.sum(zR * zSR, axis=1),
                    np.sum(zS * zSR, axis=1),
                ],
                axis=1,
            )

            for i, (row, detected) in enumerate(meta):
                out_row = {
                    "split": row["split"],
                    "method": row["method"],
                    "label": int(row["label"]),
                    "video_id": str(row["video_id"]).strip(),
                    "frame": row["frame"],
                    "face_detected": detected,
                }
                for j, name in enumerate(PAIR_NAMES):
                    out_row[name] = float(sims[i, j])
                writer.writerow(out_row)
                per_video[(row["split"], row["method"], int(row["label"]), str(row["video_id"]).strip())].append(
                    [float(detected)] + sims[i].tolist()
                )

            if batch_idx % 50 == 0:
                print(f"[batch] {batch_idx} / approx {math.ceil(len(rows) / args.batch)}", flush=True)

    with video_out.open("w", newline="") as f_v:
        fieldnames = ["split", "method", "label", "video_id", "detect_rate"] + [f"mean_{n}" for n in PAIR_NAMES] + [
            f"median_{n}" for n in PAIR_NAMES
        ] + [f"topk_{n}" for n in PAIR_NAMES]
        writer = csv.DictWriter(f_v, fieldnames=fieldnames)
        writer.writeheader()
        for (split, method, label, vid), values in per_video.items():
            arr = np.asarray(values, dtype=np.float32)
            detect_rate = float(arr[:, 0].mean())
            sims = arr[:, 1:]
            row = {"split": split, "method": method, "label": label, "video_id": vid, "detect_rate": detect_rate}
            for k, name in enumerate(PAIR_NAMES):
                row[f"mean_{name}"] = float(sims[:, k].mean())
                row[f"median_{name}"] = float(np.median(sims[:, k]))
                row[f"topk_{name}"] = float(topk_mean(sims[:, k], args.topk_ratio))
            writer.writerow(row)

    print(f"[saved] {frame_out}")
    print(f"[saved] {video_out}")


if __name__ == "__main__":
    main()
