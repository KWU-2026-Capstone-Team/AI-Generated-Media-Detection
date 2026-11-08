#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mixed_pair_helpers as TM


def annotate_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vid = df["video_id"].astype(str).str.strip()
    df["video_id"] = vid
    is_pair = vid.str.contains("_")
    df["id_a"] = vid.where(~is_pair, vid.str.split("_").str[0])
    df["id_b"] = vid.where(~is_pair, vid.str.split("_").str[1])
    return df


def select_subset(df: pd.DataFrame, orig_ids: set[str]) -> pd.DataFrame:
    mask_original = (df["label"] == 0) & df["video_id"].isin(orig_ids)
    mask_fake = (df["label"] == 1) & df["id_a"].isin(orig_ids) & df["id_b"].isin(orig_ids)
    return df[mask_original | mask_fake].copy()


def score_probs(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (proba >= threshold).astype(int)
    return {
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
        "pr": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, pred)),
        "cm": confusion_matrix(y, pred).tolist(),
    }


def select_threshold(y: np.ndarray, proba: np.ndarray) -> tuple[float, dict[str, object]]:
    best_thr = 0.5
    best_metrics = None
    best_key = None
    for thr in np.linspace(0.05, 0.95, 91):
        metrics = score_probs(y, proba, float(thr))
        key = (metrics["bacc"], metrics["f1"], metrics["auc"], -abs(thr - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
            best_metrics = metrics
    return best_thr, best_metrics


def fit_logreg(train_video: pd.DataFrame, pairs: list[str], agg: str) -> Pipeline:
    base = TM.build_real_base_video(train_video, pairs=pairs, agg=agg)
    part = train_video[train_video["method"].isin(["original", "Deepfakes"])].copy()
    tr_df, xtr = TM.make_pair_delta_video(part, base, pairs=pairs, agg=agg)
    ytr = tr_df["label"].astype(int).values
    clf = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=8000, class_weight="balanced"))])
    clf.fit(xtr.values, ytr)
    return clf


def eval_logreg(clf: Pipeline, subset_video: pd.DataFrame, pairs: list[str], agg: str) -> tuple[pd.DataFrame, np.ndarray]:
    base = TM.build_real_base_video(subset_video, pairs=pairs, agg=agg)
    df_eval, x_eval = TM.make_pair_delta_video(subset_video, base, pairs=pairs, agg=agg)
    proba = clf.predict_proba(x_eval.values)[:, 1]
    return df_eval.reset_index(drop=True), proba


def build_feat_cols(pairs: list[str]) -> list[str]:
    feat_cols = []
    for p in pairs:
        feat_cols.extend(
            [
                f"{p}_mean",
                f"{p}_std",
                f"{p}_p10",
                f"{p}_p25",
                f"{p}_p50",
                f"{p}_p75",
                f"{p}_p90",
                f"{p}_iqr",
                f"{p}_min",
                f"{p}_max",
                f"{p}_tail_frac",
                f"{p}_bottom5_mean",
            ]
        )
    for base_key in ["mean", "p50", "p10", "p90"]:
        for p in pairs:
            feat_cols.append(f"drop_{base_key}_O_R_minus_{p}")
    return feat_cols


def fit_hgb(train_enh: pd.DataFrame, feat_cols: list[str]) -> HistGradientBoostingClassifier:
    base = TM.build_real_base_enhanced(train_enh, feat_cols)
    tr_df, xtr = TM.make_pair_delta_enhanced(train_enh, base, feat_cols)
    ytr = tr_df["label"].astype(int).values
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=5,
        random_state=42,
    )
    clf.fit(xtr.values, ytr)
    return clf


def eval_hgb(clf: HistGradientBoostingClassifier, subset_enh: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    base = TM.build_real_base_enhanced(subset_enh, feat_cols)
    df_eval, x_eval = TM.make_pair_delta_enhanced(subset_enh, base, feat_cols)
    proba = clf.predict_proba(x_eval.values)[:, 1]
    return df_eval.reset_index(drop=True), proba


def combine_method_scores(
    df_ref: pd.DataFrame,
    deepfake_proba: np.ndarray,
    method_outputs: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> np.ndarray:
    key_to_idx = {(r.method, str(r.video_id).strip(), int(r.label)): i for i, r in df_ref.iterrows()}
    all_probs = [deepfake_proba]
    for method, (df_method, p_sub) in method_outputs.items():
        p_vec = np.zeros(len(df_ref), dtype=np.float32)
        for i, row in df_method.iterrows():
            key = (row["method"], str(row["video_id"]).strip(), int(row["label"]))
            if key in key_to_idx:
                p_vec[key_to_idx[key]] = p_sub[i]
        all_probs.append(p_vec)
    return np.vstack(all_probs).max(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_csv", required=True)
    ap.add_argument("--frame_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--pairs", nargs="+", default=TM.ALL_PAIRS)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--inner_val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agg", default="median")
    args = ap.parse_args()

    video_df = pd.read_csv(args.video_csv, dtype={"video_id": str})
    frame_df = pd.read_csv(args.frame_csv, dtype={"video_id": str}, low_memory=False)
    video_df = annotate_components(video_df)
    frame_df = annotate_components(frame_df)

    df_enh_all = TM.build_video_features_from_frames(frame_df, args.pairs)
    df_enh_all = annotate_components(df_enh_all)

    original_ids = sorted(video_df[video_df["method"] == "original"]["video_id"].astype(str).unique())
    outer_kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for fold_idx, (dev_idx, test_idx) in enumerate(outer_kf.split(original_ids), start=1):
        dev_ids = [original_ids[i] for i in dev_idx]
        test_ids = {original_ids[i] for i in test_idx}

        inner_split = ShuffleSplit(n_splits=1, test_size=args.inner_val_ratio, random_state=args.seed + fold_idx)
        train_rel_idx, val_rel_idx = next(inner_split.split(dev_ids))
        train_ids = {dev_ids[i] for i in train_rel_idx}
        val_ids = {dev_ids[i] for i in val_rel_idx}

        train_video = select_subset(video_df, train_ids)
        val_video = select_subset(video_df, val_ids)
        test_video = select_subset(video_df, test_ids)

        train_frames = select_subset(frame_df, train_ids)
        val_frames = select_subset(frame_df, val_ids)
        test_frames = select_subset(frame_df, test_ids)

        train_enh = select_subset(df_enh_all, train_ids)
        val_enh = select_subset(df_enh_all, val_ids)
        test_enh = select_subset(df_enh_all, test_ids)

        logreg_model = fit_logreg(train_video, pairs=args.pairs, agg=args.agg)
        df_val_ref, val_deepfake = eval_logreg(logreg_model, val_video, pairs=args.pairs, agg=args.agg)
        df_test_ref, test_deepfake = eval_logreg(logreg_model, test_video, pairs=args.pairs, agg=args.agg)

        feat_cols = build_feat_cols(args.pairs)
        val_method_outputs = {}
        test_method_outputs = {}
        for method in TM.HGB_METHODS:
            train_enh_m = train_enh[(train_enh["method"] == "original") | (train_enh["method"] == method)].copy()
            val_enh_m = val_enh[(val_enh["method"] == "original") | (val_enh["method"] == method)].copy()
            test_enh_m = test_enh[(test_enh["method"] == "original") | (test_enh["method"] == method)].copy()
            hgb_model = fit_hgb(train_enh_m, feat_cols)
            val_method_outputs[method] = eval_hgb(hgb_model, val_enh_m, feat_cols)
            test_method_outputs[method] = eval_hgb(hgb_model, test_enh_m, feat_cols)

        val_proba = combine_method_scores(df_val_ref, val_deepfake, val_method_outputs)
        test_proba = combine_method_scores(df_test_ref, test_deepfake, test_method_outputs)

        y_val = df_val_ref["label"].astype(int).values
        y_test = df_test_ref["label"].astype(int).values
        thr, val_metrics = select_threshold(y_val, val_proba)
        test_metrics = score_probs(y_test, test_proba, thr)

        rows.append(
            {
                "fold": fold_idx,
                "thr": thr,
                "val_bacc": val_metrics["bacc"],
                "val_auc": val_metrics["auc"],
                "val_pr": val_metrics["pr"],
                "val_f1": val_metrics["f1"],
                "test_bacc": test_metrics["bacc"],
                "test_auc": test_metrics["auc"],
                "test_pr": test_metrics["pr"],
                "test_f1": test_metrics["f1"],
                "bacc_gap_val_minus_test": val_metrics["bacc"] - test_metrics["bacc"],
                "auc_gap_val_minus_test": val_metrics["auc"] - test_metrics["auc"],
                "pr_gap_val_minus_test": val_metrics["pr"] - test_metrics["pr"],
                "val_cm": json.dumps(val_metrics["cm"]),
                "test_cm": json.dumps(test_metrics["cm"]),
                "n_train": int(len(train_video.drop_duplicates(subset=["method", "video_id"]))),
                "n_val": int(len(df_val_ref)),
                "n_test": int(len(df_test_ref)),
                "n_train_frames": int(len(train_frames)),
                "n_val_frames": int(len(val_frames)),
                "n_test_frames": int(len(test_frames)),
                "n_train_orig_ids": len(train_ids),
                "n_val_orig_ids": len(val_ids),
                "n_test_orig_ids": len(test_ids),
            }
        )
        print(f"[fold {fold_idx}] test_bacc={test_metrics['bacc']:.4f} test_auc={test_metrics['auc']:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    print(df_out[["test_bacc", "test_auc", "test_pr", "test_f1"]].mean().to_string())


if __name__ == "__main__":
    main()
