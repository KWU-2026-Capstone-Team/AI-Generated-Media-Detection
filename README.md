# AI-Generated-Media-Detection
Detects AI-generated images and videos

data set link : https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images

## Added experiment

- `experiments/ffpp_effb2_top3_strict`
  - FF++ 4-view similarity experiment
  - `EfficientNet-B2` embeddings
  - strict 5-fold no-leakage mixed-ensemble evaluation
  - best tested setting: `Top-3 (O_S, R_S, S_SR)`
