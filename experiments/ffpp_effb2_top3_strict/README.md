# FF++ EfficientNet-B2 Top-3 Strict CV

This folder contains the best `EfficientNet-B2`-based FF++ experiment we ran under a strict no-leakage evaluation protocol.

## Best result

- embedding model: `EfficientNet-B2`
- dataset source: `matched_4views.csv`
- views used: `O, R, S, SR`
- selected pairs: `O_S, R_S, S_SR`
- evaluation: strict 5-fold CV over original IDs

Mean test metrics across 5 folds:

- `BACC`: `0.9342`
- `AUC`: `0.8730`
- `PR-AUC`: `0.9147`
- `F1`: `0.9294`

This was the best-performing `EfficientNet-B2` setting among the strict CV runs.

## Included files

- `scripts/extract_effb2_sims_dataloader.py`
  - extracts 4-view frame similarities with EfficientNet-B2
- `scripts/merge_and_aggregate_effb2.py`
  - merges shard outputs and aggregates video-level similarities
- `scripts/mixed_pair_helpers.py`
  - feature engineering helpers for pair statistics and delta features
- `scripts/cv5_mixed_ensemble_strict.py`
  - strict no-leakage 5-fold mixed-ensemble evaluation
- `results/cv5_mixed_ensemble_strict_top3.csv`
  - fold-wise metrics for the best Top-3 setting
- `results/cv5_mixed_ensemble_strict_comparison.md`
  - comparison against `All-6` and `O,R,SR only`

## Evaluation protocol

- split unit: original video IDs
- outer loop: 5-fold CV on original IDs
- inner loop: validation split from the training original IDs only
- fake pair rule: a fake pair is kept only if both source original IDs belong to the same subset
- threshold tuning: validation only

This protocol is stricter than the earlier fixed-test setup and is the recommended one to report.

## Quick start

1. Re-embed `matched_4views.csv` with `extract_effb2_sims_dataloader.py`.
2. Merge shard outputs with `merge_and_aggregate_effb2.py`.
3. Run `cv5_mixed_ensemble_strict.py` with:

```bash
python scripts/cv5_mixed_ensemble_strict.py \
  --video_csv /path/to/effb2_video_sims.csv \
  --frame_csv /path/to/effb2_frame_sims.csv \
  --out_csv results/cv5_mixed_ensemble_strict_top3.csv \
  --pairs O_S R_S S_SR
```

## Dependencies

Recommended Python packages:

- `torch`
- `timm`
- `pandas`
- `numpy`
- `scikit-learn`
- `Pillow`
