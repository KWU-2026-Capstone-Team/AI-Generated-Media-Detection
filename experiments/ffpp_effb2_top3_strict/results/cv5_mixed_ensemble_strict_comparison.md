# Strict 5-Fold Mixed Ensemble Comparison

Dataset:
- `/home/tako/seelhwan/matched_4views.csv`
- re-embedded and aggregated into:
  - `/home/tako/seelhwan/ffpp_matched4views_full_experiment/cache/full_matched4views/effb2_frame_sims.csv`
  - `/home/tako/seelhwan/ffpp_matched4views_full_experiment/cache/full_matched4views/effb2_video_sims.csv`

Protocol:
- 5-fold CV over original video IDs
- inner validation split from the training original IDs
- fake pairs are kept only when both source original IDs belong to the same subset
- threshold tuned on validation only

## Mean test metrics across 5 folds

| setting | views needed | pairs | test BACC | test AUC | test PR-AUC | test F1 |
|---|---:|---|---:|---:|---:|---:|
| All-6 | 4 | `O_R,O_S,O_SR,R_S,R_SR,S_SR` | 0.9271 | 0.8601 | 0.9063 | 0.9212 |
| Top-3 | 4 | `O_S,R_S,S_SR` | 0.9342 | 0.8730 | 0.9147 | 0.9294 |
| O,R,SR only | 3 | `O_R,O_SR,R_SR` | 0.9288 | 0.8634 | 0.9092 | 0.9226 |

## Quick read

- `Top-3` is the best performer under the strict no-leakage CV protocol.
- `O,R,SR only` is very close to `All-6`, while using only 3 views instead of 4.
- If the goal is best accuracy, `Top-3` wins.
- If the goal is reducing preprocessing cost by removing `S`, `O,R,SR only` is the strongest speed-oriented option tested here.

## Saved CSVs

- All-6:
  - `/home/tako/seelhwan/ffpp_matched4views_full_experiment/results/cv5_mixed_ensemble_strict_all6.csv`
- Top-3:
  - `/home/tako/seelhwan/ffpp_matched4views_full_experiment/results/cv5_mixed_ensemble_strict_top3.csv`
- O,R,SR only:
  - `/home/tako/seelhwan/ffpp_matched4views_full_experiment/results/cv5_mixed_ensemble_strict_orsr.csv`
