# FF++ Dual-Branch Strict CV

This folder contains our full FF++ dual-branch experiment:

- full-frame branch: `DINOv2`
- face branch: `ArcFace`
- evaluation: strict 5-fold no-leakage CV

## Best result

Best tested setting:

- full-frame pairs: `O_R, O_SR, R_SR`
- face pairs: `O_R, O_S, O_SR, R_S, R_SR, S_SR`
- output file: `results/cv5_dual_branch_strict_fullORSR_faceAll6.csv`

Mean test metrics across 5 folds:

- `BACC`: `0.9771`
- `AUC`: `0.9623`
- `PR-AUC`: `0.9748`
- `F1`: `0.9764`

## Included files

- `scripts/extract_timm_4view_sims.py`
  - extracts 4-view full-frame similarities with `DINOv2`
- `scripts/merge_timm_4view_sims.py`
  - merges DINO shard outputs
- `scripts/extract_arcface_face_sims.py`
  - extracts face-branch similarities with `ArcFace`
- `scripts/merge_face_arc_sims.py`
  - merges ArcFace shard outputs
- `scripts/mixed_pair_helpers.py`
  - shared feature engineering helpers
- `scripts/cv5_mixed_ensemble_strict.py`
  - strict no-leakage single-branch evaluation
- `scripts/cv5_dual_branch_ensemble_strict.py`
  - strict no-leakage dual-branch ensemble evaluation

## Result files

- `results/cv5_dual_branch_strict_fullORSR_faceAll6.csv`
  - best dual-branch setting
- `results/cv5_dual_branch_strict_fullTop3_faceAll6.csv`
  - DINO Top-3 + ArcFace All-6
- `results/cv5_dual_branch_strict_fullTop3_faceTop3.csv`
  - DINO Top-3 + ArcFace Top-3

## Protocol

- source dataset: `matched_4views.csv`
- split unit: original video IDs
- outer loop: 5-fold CV on original IDs
- inner loop: validation split from training original IDs only
- fake pair rule: keep a fake pair only when both source original IDs belong to the same subset
- threshold tuning: validation only

## Notes

- This is a stricter evaluation than the earlier fixed-test setup.
- In the tested settings, the face branch usually received a larger ensemble weight than the full-frame branch.
- The best result came from a lighter full-frame pair set (`O,R,SR only`) combined with the richer face-branch pair set (`All-6`).

## Dependencies

Recommended Python packages:

- `torch`
- `timm`
- `pandas`
- `numpy`
- `scikit-learn`
- `Pillow`
- `opencv-python`
- `insightface`
- `onnxruntime-gpu`
