#!/usr/bin/env bash
set -euo pipefail

# Example only: replace these with your local paths.
MATCHED_CSV="/path/to/matched_4views.csv"
SHARD1="/path/to/matched_4views_shard_1_of_3.csv"
SHARD2="/path/to/matched_4views_shard_2_of_3.csv"
SHARD3="/path/to/matched_4views_shard_3_of_3.csv"
WORKDIR="/path/to/workdir"

python scripts/extract_effb2_sims_dataloader.py --csv "$SHARD1" --outdir "$WORKDIR/shard_1_of_3"
python scripts/extract_effb2_sims_dataloader.py --csv "$SHARD2" --outdir "$WORKDIR/shard_2_of_3"
python scripts/extract_effb2_sims_dataloader.py --csv "$SHARD3" --outdir "$WORKDIR/shard_3_of_3"

python scripts/merge_and_aggregate_effb2.py \
  --frame_csvs \
  "$WORKDIR/shard_1_of_3/effb2_frame_sims.csv" \
  "$WORKDIR/shard_2_of_3/effb2_frame_sims.csv" \
  "$WORKDIR/shard_3_of_3/effb2_frame_sims.csv" \
  --outdir "$WORKDIR/full_matched4views"

python scripts/cv5_mixed_ensemble_strict.py \
  --video_csv "$WORKDIR/full_matched4views/effb2_video_sims.csv" \
  --frame_csv "$WORKDIR/full_matched4views/effb2_frame_sims.csv" \
  --out_csv "$WORKDIR/results/cv5_mixed_ensemble_strict_top3.csv" \
  --pairs O_S R_S S_SR
