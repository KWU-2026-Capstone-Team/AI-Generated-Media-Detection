#!/usr/bin/env python3

from __future__ import annotations

import math

import numpy as np
import pandas as pd


ALL_PAIRS = ["O_R", "O_S", "O_SR", "R_S", "R_SR", "S_SR"]
HGB_METHODS = ["Face2Face", "FaceSwap", "NeuralTextures"]
PCTS = [10, 25, 50, 75, 90]


def pair_to_two_ids(pair_id: str):
    s = str(pair_id).strip()
    if "_" not in s:
        return None
    return s.split("_", 1)


def make_drop_video(df: pd.DataFrame, pairs: list[str], agg: str = "median") -> pd.DataFrame:
    feat = {}
    base_m = df[f"{agg}_O_R"].astype(float).values
    base_t = df["topk_O_R"].astype(float).values
    for p in pairs:
        feat[f"drop_{agg}_O_R_minus_{p}"] = base_m - df[f"{agg}_{p}"].astype(float).values
        feat[f"drop_topk_O_R_minus_{p}"] = base_t - df[f"topk_{p}"].astype(float).values
    x = pd.DataFrame(feat)
    return x.replace([np.inf, -np.inf], np.nan).fillna(x.median(numeric_only=True))


def build_real_base_video(df_all: pd.DataFrame, pairs: list[str], agg: str) -> dict[str, np.ndarray]:
    real = df_all[(df_all["label"] == 0) & (df_all["method"] == "original")].copy()
    real["video_id"] = real["video_id"].astype(str).str.strip()
    xr = make_drop_video(real, pairs=pairs, agg=agg).reset_index(drop=True)
    real = real.reset_index(drop=True)
    return {vid: xr.loc[g.index].mean(axis=0).values.astype(np.float32) for vid, g in real.groupby("video_id")}


def make_pair_delta_video(
    df_part: pd.DataFrame,
    real_base: dict[str, np.ndarray],
    pairs: list[str],
    agg: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_part = df_part.copy()
    df_part["video_id"] = df_part["video_id"].astype(str).str.strip()
    xd = make_drop_video(df_part, pairs=pairs, agg=agg).reset_index(drop=True)
    df_part = df_part.reset_index(drop=True)
    keep = []
    xdelta = []
    for i, row in df_part.iterrows():
        vid = row["video_id"]
        two = pair_to_two_ids(vid)
        x = xd.iloc[i].values.astype(np.float32)
        if two is None:
            if vid not in real_base:
                continue
            xdelta.append(x - real_base[vid])
            keep.append(True)
            continue
        a, b = two
        if (a not in real_base) or (b not in real_base):
            continue
        xdelta.append(x - 0.5 * (real_base[a] + real_base[b]))
        keep.append(True)
    idx = np.where(keep)[0]
    cols = [f"delta_{c}" for c in xd.columns]
    return df_part.iloc[idx].copy(), pd.DataFrame(np.vstack(xdelta), columns=cols)


def summarize_pair(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    stats = {"mean": float(arr.mean()), "std": float(arr.std())}
    for p in PCTS:
        stats[f"p{p}"] = float(np.percentile(arr, p))
    stats["iqr"] = stats["p75"] - stats["p25"]
    stats["min"] = float(arr.min())
    stats["max"] = float(arr.max())
    thr = stats["p10"]
    stats["tail_frac"] = float((arr <= thr).mean())
    k = max(1, int(math.ceil(len(arr) * 0.05)))
    stats["bottom5_mean"] = float(np.mean(np.sort(arr)[:k]))
    return stats


def build_video_features_from_frames(df_frames: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    rows = []
    summary_pairs = ["O_R"] + [p for p in pairs if p != "O_R"]
    for (split, method, label, vid), g in df_frames.groupby(["split", "method", "label", "video_id"]):
        feat = {"split": split, "method": method, "label": int(label), "video_id": str(vid).strip()}
        for p in summary_pairs:
            stats = summarize_pair(g[p].values)
            for k, v in stats.items():
                feat[f"{p}_{k}"] = v
        for base_key in ["mean", "p50", "p10", "p90"]:
            base = feat[f"O_R_{base_key}"]
            for p in pairs:
                feat[f"drop_{base_key}_O_R_minus_{p}"] = base - feat[f"{p}_{base_key}"]
        rows.append(feat)
    return pd.DataFrame(rows)


def build_real_base_enhanced(dfv: pd.DataFrame, feat_cols: list[str]) -> dict[str, np.ndarray]:
    real = dfv[(dfv["label"] == 0) & (dfv["method"] == "original")].copy()
    return {vid: g[feat_cols].mean(axis=0).values.astype(np.float32) for vid, g in real.groupby("video_id")}


def make_pair_delta_enhanced(
    df_part: pd.DataFrame,
    real_base: dict[str, np.ndarray],
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_part = df_part.copy()
    df_part["video_id"] = df_part["video_id"].astype(str).str.strip()
    keep = []
    xdelta = []
    for _, row in df_part.iterrows():
        vid = row["video_id"]
        x = row[feat_cols].values.astype(np.float32)
        two = pair_to_two_ids(vid)
        if two is None:
            if vid not in real_base:
                continue
            xdelta.append(x - real_base[vid])
            keep.append(True)
            continue
        a, b = two
        if (a not in real_base) or (b not in real_base):
            continue
        xdelta.append(x - 0.5 * (real_base[a] + real_base[b]))
        keep.append(True)
    idx = np.where(keep)[0]
    return df_part.iloc[idx].copy(), pd.DataFrame(np.vstack(xdelta), columns=[f"delta_{c}" for c in feat_cols])
