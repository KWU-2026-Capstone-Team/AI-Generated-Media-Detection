# Best EfficientNet-B2 Result Summary

Selected setting:

- experiment: `Top-3`
- pairs: `O_S, R_S, S_SR`
- views required: `O, R, S, SR`
- protocol: strict 5-fold no-leakage CV

Mean test metrics:

- `BACC`: `0.9342`
- `AUC`: `0.8730`
- `PR-AUC`: `0.9147`
- `F1`: `0.9294`

Comparison under the same protocol:

- `All-6`: `BACC 0.9271`, `AUC 0.8601`, `PR-AUC 0.9063`, `F1 0.9212`
- `Top-3`: `BACC 0.9342`, `AUC 0.8730`, `PR-AUC 0.9147`, `F1 0.9294`
- `O,R,SR only`: `BACC 0.9288`, `AUC 0.8634`, `PR-AUC 0.9092`, `F1 0.9226`

Interpretation:

- If the goal is best accuracy with EfficientNet-B2, `Top-3` is the strongest setting we tested.
- If the goal is reducing preprocessing cost by removing `S`, `O,R,SR only` is the strongest lighter alternative.
