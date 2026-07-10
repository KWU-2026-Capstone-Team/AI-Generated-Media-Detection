# 모델 가중치

이중브랜치 딥페이크 판별기의 학습된 가중치. **fp16(반정밀도)으로 저장돼 각 파일 <100MB → git에 직접 포함**
(원본 fp32는 ConvNeXt fold 107MB로 GitHub 100MB 한도 초과 → fp16으로 절반 압축). `git clone`만 하면 바로 실행 가능.

`infer.py`가 로드 시 fp32로 업캐스트하므로 추론 정밀도 손실 없음 (딥페이크 점수 차이 <0.001, 검증됨).

## 구성 (총 8개, ~331MB)

```
models/
├── spatial/    ConvNeXt-tiny 5-fold 앙상블 (전체 얼굴, 공간 브랜치)
│   └── convnext_fold{1..5}.pt      (각 ~56MB, fp16)
└── temporal/   mc3_18 3D-CNN 3-fold 앙상블 (입영역 클립, 시공간 브랜치)
    └── temporal_fold{1..3}.pt      (각 ~23MB, fp16)
```

각 `.pt`는 `torch.save` 딕셔너리 (state_dict가 fp16):
- spatial: `{state_dict, backbone:"convnext_tiny.fb_in22k_ft_in1k", image_size, ...}`
- temporal: `{state_dict, backbone:"mc3_18", T:16, size:112, mouth:(...)}`

## 재학습 (재현용)
`EXPERIMENT_LOG.md`의 실험 14(딥 temporal)·배포 학습 절차 참고:
- 공간: `train_direct_face_classifier.py` (4기법 전부, 5-fold, `--save_fold_models`)
- 시간: `clip_temporal.py` (mc3_18, 입crop 3D-CNN, `--save_models`)

## 성능 (누수0 component split)
- FF++ in-dist AUC **0.940** (4기법 전부 ≥0.89, NeuralTextures 0.55→0.89)
- Celeb-DF v2 cross-dataset AUC **0.80** (공간 브랜치, 완전 미지 데이터셋)
