# 모델 가중치

이중브랜치 딥페이크 판별기의 학습된 가중치. **Git LFS로 저장소에 포함됨**
(ConvNeXt fold 각 107MB > GitHub 100MB 단일파일 한도라 일반 git 불가 → LFS 사용).
git-lfs가 설치된 환경에서 `git clone` 하면 자동으로 받아진다.

## 구성 (총 8개, ~663MB)

```
models/
├── spatial/    ConvNeXt-tiny 5-fold 앙상블 (전체 얼굴, 공간 브랜치)
│   └── convnext_fold{1..5}.pt      (각 ~107MB)
└── temporal/   mc3_18 3D-CNN 6-model 앙상블 (입영역 클립, 시공간 브랜치)
    ├── temporal_base_fold{1..3}.pt   표준 지도학습 3-fold  (각 ~23MB, fp16)
    └── temporal_aug_fold{1..3}.pt    과평활-aug(frac=0.30) 3-fold (각 ~23MB, fp16)
    # 추론(infer.py)은 temporal 폴더의 *.pt 전부를 평균 앙상블.
    # aug 모델은 재연(reenactment) 강건성 hedge: in-dist NT 0.895→0.904(앙상블), swap 무손상.
    # 근거: EXPERIMENT_LOG Part 3 (LOMO NT +0.055, aug 강도 스윕 frac=0.30 sweet spot).
```

각 `.pt`는 `torch.save` 딕셔너리:
- spatial: `{state_dict, backbone:"convnext_tiny.fb_in22k_ft_in1k", image_size, ...}`
- temporal: `{state_dict, backbone:"mc3_18", T:16, size:112, mouth:(...)}`

## 확보 방법

**옵션 A — Git LFS (저장소에 포함하려면)**
```bash
git lfs install
git lfs track "models/**/*.pt"
# .gitignore에서 models 제외 줄을 지운 뒤
git add .gitattributes models/ && git commit -m "add model weights via LFS"
```
※ GitHub 무료 LFS 쿼터(1GB 저장/1GB월 대역폭)를 확인할 것. 663MB는 빠듯함.

**옵션 B — 재학습 (권장, 재현 가능)**
`EXPERIMENT_LOG.md`의 실험 14(딥 temporal)·배포 학습 절차 참고. 학습 스크립트:
- 공간: `train_direct_face_classifier.py` (4기법 전부, 5-fold, `--save_fold_models`)
- 시간: `clip_temporal.py` (mc3_18, 입crop 3D-CNN, `--save_models`)

## 성능 (누수0 component split)
- FF++ in-dist AUC **0.940** (4기법 전부 ≥0.89, NeuralTextures 0.55→0.89)
- Celeb-DF v2 cross-dataset AUC **0.80** (공간 브랜치, 완전 미지 데이터셋)
- temporal 앙상블(6모델) in-dist 0.931 (base3 0.927 대비 +0.004, NT +0.009, swap 무손상)
- 미지-재연 일반화(LOMO, swap만 지도→재연 held-out): 과평활 aug가 NT +0.055
- 캘리브레이션: temporal Platt 보정(FF++ fit) → 고정0.5 BACC FF++ +0.081, Celeb융합 +0.050 (AUC 불변). calibration.json
