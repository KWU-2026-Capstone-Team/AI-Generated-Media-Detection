#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cv5_mixed_ensemble_strict as strict_mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_video_csv", required=True)
    ap.add_argument("--full_frame_csv", required=True)
    ap.add_argument("--face_video_csv", required=True)
    ap.add_argument("--face_frame_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--full_pairs", nargs="+", required=True)
    ap.add_argument("--face_pairs", nargs="+", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--inner_val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agg", default="median")
    args = ap.parse_args()

    video_full = strict_mod.annotate_components(pd.read_csv(args.full_video_csv, dtype={"video_id": str}))
    frame_full = strict_mod.annotate_components(pd.read_csv(args.full_frame_csv, dtype={"video_id": str}, low_memory=False))
    enh_full = strict_mod.annotate_components(strict_mod.TM.build_video_features_from_frames(frame_full, args.full_pairs))

    video_face = strict_mod.annotate_components(pd.read_csv(args.face_video_csv, dtype={"video_id": str}))
    frame_face = strict_mod.annotate_components(pd.read_csv(args.face_frame_csv, dtype={"video_id": str}, low_memory=False))
    enh_face = strict_mod.annotate_components(strict_mod.TM.build_video_features_from_frames(frame_face, args.face_pairs))

    original_ids = sorted(video_full[video_full["method"] == "original"]["video_id"].astype(str).unique())
    outer_kf = strict_mod.KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows = []

    for fold_idx, (dev_idx, test_idx) in enumerate(outer_kf.split(original_ids), start=1):
        dev_ids = [original_ids[i] for i in dev_idx]
        test_ids = {original_ids[i] for i in test_idx}
        inner_split = strict_mod.ShuffleSplit(n_splits=1, test_size=args.inner_val_ratio, random_state=args.seed + fold_idx)
        train_rel_idx, val_rel_idx = next(inner_split.split(dev_ids))
        train_ids = {dev_ids[i] for i in train_rel_idx}
        val_ids = {dev_ids[i] for i in val_rel_idx}

        def branch_scores(video_df, enh_df, pairs):
            train_video = strict_mod.select_subset(video_df, train_ids)
            val_video = strict_mod.select_subset(video_df, val_ids)
            test_video = strict_mod.select_subset(video_df, test_ids)
            train_enh = strict_mod.select_subset(enh_df, train_ids)
            val_enh = strict_mod.select_subset(enh_df, val_ids)
            test_enh = strict_mod.select_subset(enh_df, test_ids)

            logreg_model = strict_mod.fit_logreg(train_video, pairs=pairs, agg=args.agg)
            df_val_ref, val_deepfake = strict_mod.eval_logreg(logreg_model, val_video, pairs=pairs, agg=args.agg)
            df_test_ref, test_deepfake = strict_mod.eval_logreg(logreg_model, test_video, pairs=pairs, agg=args.agg)

            feat_cols = strict_mod.build_feat_cols(pairs)
            val_method_outputs = {}
            test_method_outputs = {}
            for method in strict_mod.TM.HGB_METHODS:
                train_enh_m = train_enh[(train_enh["method"] == "original") | (train_enh["method"] == method)].copy()
                val_enh_m = val_enh[(val_enh["method"] == "original") | (val_enh["method"] == method)].copy()
                test_enh_m = test_enh[(test_enh["method"] == "original") | (test_enh["method"] == method)].copy()
                hgb_model = strict_mod.fit_hgb(train_enh_m, feat_cols)
                val_method_outputs[method] = strict_mod.eval_hgb(hgb_model, val_enh_m, feat_cols)
                test_method_outputs[method] = strict_mod.eval_hgb(hgb_model, test_enh_m, feat_cols)

            val_proba = strict_mod.combine_method_scores(df_val_ref, val_deepfake, val_method_outputs)
            test_proba = strict_mod.combine_method_scores(df_test_ref, test_deepfake, test_method_outputs)
            return df_val_ref, val_proba, df_test_ref, test_proba

        df_val_full, val_full, df_test_full, test_full = branch_scores(video_full, enh_full, args.full_pairs)
        df_val_face, val_face, df_test_face, test_face = branch_scores(video_face, enh_face, args.face_pairs)

        key_val_full = {(r.method, str(r.video_id).strip(), int(r.label)): i for i, r in df_val_full.iterrows()}
        key_test_full = {(r.method, str(r.video_id).strip(), int(r.label)): i for i, r in df_test_full.iterrows()}
        val_face_aligned = np.zeros_like(val_full)
        test_face_aligned = np.zeros_like(test_full)
        for i, row in df_val_face.iterrows():
            key = (row["method"], str(row["video_id"]).strip(), int(row["label"]))
            if key in key_val_full:
                val_face_aligned[key_val_full[key]] = val_face[i]
        for i, row in df_test_face.iterrows():
            key = (row["method"], str(row["video_id"]).strip(), int(row["label"]))
            if key in key_test_full:
                test_face_aligned[key_test_full[key]] = test_face[i]

        y_val = df_val_full["label"].astype(int).values
        y_test = df_test_full["label"].astype(int).values

        best = None
        best_weight = None
        best_thr = None
        best_val = None
        for w in np.linspace(0.0, 1.0, 21):
            val_combo = w * val_full + (1.0 - w) * val_face_aligned
            thr, val_metrics = strict_mod.select_threshold(y_val, val_combo)
            key = (val_metrics["bacc"], val_metrics["f1"], val_metrics["auc"])
            if best is None or key > best:
                best = key
                best_weight = float(w)
                best_thr = float(thr)
                best_val = val_metrics

        test_combo = best_weight * test_full + (1.0 - best_weight) * test_face_aligned
        test_metrics = strict_mod.score_probs(y_test, test_combo, best_thr)
        rows.append(
            {
                "fold": fold_idx,
                "full_weight": best_weight,
                "face_weight": 1.0 - best_weight,
                "thr": best_thr,
                "val_bacc": best_val["bacc"],
                "val_auc": best_val["auc"],
                "val_pr": best_val["pr"],
                "val_f1": best_val["f1"],
                "test_bacc": test_metrics["bacc"],
                "test_auc": test_metrics["auc"],
                "test_pr": test_metrics["pr"],
                "test_f1": test_metrics["f1"],
                "val_cm": json.dumps(best_val["cm"]),
                "test_cm": json.dumps(test_metrics["cm"]),
                "n_val": int(len(df_val_full)),
                "n_test": int(len(df_test_full)),
            }
        )
        print(f"[fold {fold_idx}] weight_full={best_weight:.2f} test_bacc={test_metrics['bacc']:.4f} test_auc={test_metrics['auc']:.4f}")

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    print(df[["test_bacc", "test_auc", "test_pr", "test_f1"]].mean().to_string())


if __name__ == "__main__":
    main()
