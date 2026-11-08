# Best Dual-Branch Result Summary

Best tested setting:

- full-frame branch: `DINOv2`
- face branch: `ArcFace`
- full-frame pairs: `O_R, O_SR, R_SR`
- face pairs: `O_R, O_S, O_SR, R_S, R_SR, S_SR`
- result file: `cv5_dual_branch_strict_fullORSR_faceAll6.csv`

Mean test metrics:

- `BACC`: `0.9771`
- `AUC`: `0.9623`
- `PR-AUC`: `0.9748`
- `F1`: `0.9764`

Comparison under the same protocol:

- `fullTop3_faceAll6`
  - `BACC 0.9741`
  - `AUC 0.9642`
  - `PR-AUC 0.9764`
  - `F1 0.9734`
- `fullTop3_faceTop3`
  - `BACC 0.9625`
  - `AUC 0.9452`
  - `PR-AUC 0.9638`
  - `F1 0.9608`
- `fullORSR_faceAll6`
  - `BACC 0.9771`
  - `AUC 0.9623`
  - `PR-AUC 0.9748`
  - `F1 0.9764`

Interpretation:

- The best dual-branch result came from the lighter full-frame pair set (`O,R,SR only`) combined with the richer face pair set (`All-6`).
- The face branch tended to receive a larger ensemble weight than the full-frame branch.
