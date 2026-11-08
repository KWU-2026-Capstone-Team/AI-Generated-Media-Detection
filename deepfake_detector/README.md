# 딥페이크 판별기 — 공간 × 시간 이중브랜치

FaceForensics++ 기반 졸업작품. **공간(전체 얼굴)** 브랜치와 **시공간(입영역 클립)** 브랜치가
각기 다른 조작을 담당해, 어떤 단일 모델도 못 잡던 조작까지 커버한다.

## 핵심 성과 (누수 0 · component split · in-dist)

| | Deepfakes | Face2Face | FaceSwap | NeuralTextures | **전체** |
|---|---|---|---|---|---|
| 융합(제품) | 0.975 | 0.945 | 0.950 | **0.891** | **0.940** |

- **4기법 전부 ≥ 0.89, 구멍 없음.**
- **NeuralTextures 벽 돌파:** 외형·잔차·blending·foundation 4패러다임이 전부 ~0.55에서 막혔던 조작을
  시공간 브랜치가 **0.89**로 끌어올림.
- **미지 조작 robust:** 학습 때 안 본 SimSwap을 시공간 브랜치(0.93)가 잡음.

## 폴더 구조

```
deepfake_detector/
├── app.py                # 웹앱 서버 (영상 업로드 → 판정)
├── infer.py              # 추론 파이프라인 / CLI (자립 실행)
├── models/
│   ├── spatial/          # ConvNeXt 5-fold 앙상블 (전체 얼굴)
│   │   └── convnext_fold{1..5}.pt
│   └── temporal/         # 3D-CNN(mc3_18) 6모델 앙상블 (입영역 클립)
│       ├── temporal_base_fold{1..3}.pt   # 표준 지도학습
│       └── temporal_aug_fold{1..3}.pt    # 과평활-aug(재연 강건성 hedge)
├── presentation.html     # 발표·심사용 비주얼 페이지 (브라우저로 열기)
├── EXPERIMENT_LOG.md     # 전체 연구 여정 기록 (반증 6개 + 벤치마크 + 돌파)
└── README.md
```

## 웹앱 — 영상 업로드로 판정 (권장)

```bash
python app.py                 # 기본 http://0.0.0.0:7860
python app.py --port 8000     # 포트 변경
```
브라우저로 `http://localhost:7860` 접속 → 영상을 끌어다 놓으면 얼굴을 검출해
**REAL / FAKE 판정 + 신뢰 점수 + 어느 브랜치가 잡았는지**를 보여준다. (설치 불필요: werkzeug 기반)

## 사용법 (CLI)

```bash
# 1) raw 비디오 (MTCNN 얼굴검출 자동)
python infer.py --video sample.mp4

# 2) 정렬된 얼굴 크롭 이미지 폴더 (연속 프레임)
python infer.py --frames_dir path/to/face_frames/
```

출력 예:
```
╔══════════ 딥페이크 판별 결과 ══════════╗
  입력: sample.mp4  (얼굴 16프레임 · device=cuda)
  ├ 공간 브랜치 (ConvNeXt 전체얼굴):  0.068
  ├ 시공간 브랜치 (3D-CNN 입영역):    0.929
  └ 종합 fake 확률 (mean fusion):     0.498
  ▶ 판정: 🔴 FAKE  (근거: 시공간 | mean≥0.5 or 브랜치≥0.85)
╚═══════════════════════════════════════╝
```

**판정 규칙(robust):** `mean ≥ 0.5` **또는** 어느 한 브랜치 `≥ 0.85` → FAKE.
미지 조작에서 한 브랜치만 강하게 반응하는 경우(위 SimSwap 예)를 놓치지 않는다.

## 의존성

```
torch  torchvision  timm  pillow  numpy
# --video 사용 시 추가:  opencv-python  facenet-pytorch
```

- 모델 가중치는 `models/`에 포함(공간 ~557MB, 시간 ~138MB). 백본 사전학습 재다운로드 불필요.
- GPU 있으면 자동 사용, 없으면 CPU로 동작.

## 동작 원리 (요약)

```
video → 얼굴검출 → ┌ 공간: 전체 얼굴 → ConvNeXt      → 외형 아티팩트
                    └ 시간: 입 클립(16f) → 3D-CNN      → 텍스처 flicker(동역학)
                → robust fusion → fake 확률
```

- **왜 이 구조인가:** 누수 없는 4패러다임 벤치마크가 NeuralTextures를 "모든 공간 신호가 막히는 벽"으로
  규명했고, 그 원인("공간적으로 조용, 시간적으로 시끄러움")을 진단해 직교축(시간)으로 뚫었다.
  자세한 여정은 `EXPERIMENT_LOG.md` 참고.
