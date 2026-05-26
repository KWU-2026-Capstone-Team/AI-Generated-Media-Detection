#!/usr/bin/env bash
set -euo pipefail

# Example only: replace these with your local paths.
SHARD1="/path/to/matched_4views_shard_1_of_3.csv"
SHARD2="/path/to/matched_4views_shard_2_of_3.csv"
SHARD3="/path/to/matched_4views_shard_3_of_3.csv"
WORKDIR="/path/to/workdir"

# 1) Full-frame DINOv2 similarities
python scripts/extract_timm_4view_sims.py --csv "$SHARD1" --outdir "$WORKDIR/dino_shard_1" --model vit_base_patch14_dinov2
python scripts/extract_timm_4view_sims.py --csv "$SHARD2" --outdir "$WORKDIR/dino_shard_2" --model vit_base_patch14_dinov2
python scripts/extract_timm_4view_sims.py --csv "$SHARD3" --outdir "$WORKDIR/dino_shard_3" --model vit_base_patch14_dinov2

python scripts/merge_timm_4view_sims.py \
  --frame_csvs \
  "$WORKDIR/dino_shard_1/frame_sims.csv" \
  "$WORKDIR/dino_shard_2/frame_sims.csv" \
  "$WORKDIR/dino_shard_3/frame_sims.csv" \
  --outdir "$WORKDIR/dinov2_full" \
  --prefix dinov2

# 2) Face ArcFace similarities
python scripts/extract_arcface_face_sims.py --csv "$SHARD1" --outdir "$WORKDIR/arcface_shard_1" --batch 128
python scripts/extract_arcface_face_sims.py --csv "$SHARD2" --outdir "$WORKDIR/arcface_shard_2" --batch 128
python scripts/extract_arcface_face_sims.py --csv "$SHARD3" --outdir "$WORKDIR/arcface_shard_3" --batch 128

python scripts/merge_face_arc_sims.py \
  --frame_csvs \
  "$WORKDIR/arcface_shard_1/face_arc_frame_sims.csv" \
  "$WORKDIR/arcface_shard_2/face_arc_frame_sims.csv" \
  "$WORKDIR/arcface_shard_3/face_arc_frame_sims.csv" \
  --outdir "$WORKDIR/arcface_full"

# 3) Best strict CV setting
python scripts/cv5_dual_branch_ensemble_strict.py \
  --full_video_csv "$WORKDIR/dinov2_full/dinov2_video_sims.csv" \
  --full_frame_csv "$WORKDIR/dinov2_full/dinov2_frame_sims.csv" \
  --face_video_csv "$WORKDIR/arcface_full/face_arc_video_sims.csv" \
  --face_frame_csv "$WORKDIR/arcface_full/face_arc_frame_sims.csv" \
  --out_csv "$WORKDIR/results/cv5_dual_branch_strict_fullORSR_faceAll6.csv" \
  --full_pairs O_R O_SR R_SR \
  --face_pairs O_R O_S O_SR R_S R_SR S_SR
