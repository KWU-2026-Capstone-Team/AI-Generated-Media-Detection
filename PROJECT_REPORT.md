# 딥페이크 판별 프로젝트 종합 리포트
## 공간 × 시간 이중브랜치 탐지기 — 진단에서 제품까지

> **한 줄 요약.** FaceForensics++로 학습한 딥페이크 판별기를 만들되, "무엇이 딥페이크를 배신하는가"를 누수 없는 실험으로 규명했다. 네 가지 탐지 패러다임이 모두 무너진 조작(NeuralTextures)이 **공간적으로는 조용하지만 시간적으로는 시끄럽다**는 것을 측정으로 밝히고, 그 진단이 지목한 직교축(시간)으로 벽을 뚫어 **공간+시간 이중브랜치 판별기**를 완성했다. 누수0 in-dist AUC **0.940**(4기법 구멍 없음), 미지 데이터셋 Celeb-DF cross-dataset **0.80**.

---

## 초록

이 프로젝트의 가치는 최종 성능뿐 아니라 **정직한 실패 지도**에 있다. 극적인 가설을 여덟 번 세웠고 여덟 번 데이터에 부딪혀 정정했다. 각 반증이 다음 실험을 더 정확하게 만들었고, "FaceSwap을 콕 집어 0.52"라는 극적 서사를 "조작 계열마다 최적 탐지 축이 다르다"는 견고한 구조로 바꿨다. 최종 산출물은 ① 작동하는 이중브랜치 판별기(코드+웹앱+가중치), ② 누수0 4패러다임 벤치마크, ③ 조작 계열별 상보성의 정량적 규명, ④ 재현 가능한 전 과정 기록, ⑤ 재연=과평활 방향의 규명과 미지-재연 일반화 aug(§11), ⑥ 충실도 검증된 XAI(§12)이다.

---

## 1. 서론

### 1.1 배경과 출발점
초기 목표는 "재조작(challenge-response) 잔차 |X − T(X)|로 딥페이크를 판별한다"였다. 이 가설은 검증 과정에서 여러 문제가 드러났고(추론 시점 InSwapper 지문 혼입, 신원 거리 confound), 이를 정직하게 해체하며 **"무엇이 딥페이크를 배신하는가"**라는 더 근본적 질문으로 전환했다.

### 1.2 이 리포트가 다루는 것
- 진단: 저수준·외형·blending·foundation 네 패러다임의 누수0 비교와 그 과정의 반증 연쇄
- 돌파: NeuralTextures 벽의 규명과 시간 축을 통한 해결
- 시스템: 공간+시간 이중브랜치 판별기와 성능
- 일반화: 미지 데이터셋(Celeb-DF) cross-dataset 검증
- 게이트: 상보성을 라우팅으로 착취할 수 있는가에 대한 사전 등록 실험

---

## 2. 데이터와 방법론

### 2.1 데이터셋
- **FaceForensics++ (FF++)** — original 500 + 4 조작(Deepfakes, Face2Face, FaceSwap, NeuralTextures) × 500 = 총 2,500 비디오. c23 압축, MTCNN 정렬 얼굴 크롭(224).
- **Celeb-DF v2** — real 590 + Celeb-synthesis(fake) 5,639. **학습에 전혀 쓰지 않은 미지 데이터셋**(다른 인물·조작·출처)으로 cross-dataset 일반화 측정에 사용.

### 2.2 방법론적 토대: Component Split (핵심)
FF++ fake A_B(A=대상, B=소스 신원)에서, 같은 사람이 train과 test에 동시에 있으면 모델이 "조작 흔적" 대신 "이 사람 얼굴"을 외워 지표가 부풀려진다(신원 누수).

- **이전 strict split의 결함:** fake를 "A·B가 둘 다 같은 fold일 때만" 포함 → 신원을 개별 배정하면 대부분 흩어져 **fake의 86%가 버려짐**(500개 중 68개 생존, fold당 8~22개). 지표 신뢰 불가.
- **Component split:** fake A_B를 간선 A–B로 보고 union-find로 연결요소 생성(285개, 대부분 크기 2), 요소를 통째로 fold에 배정 → **신원 누수 0 + fake 100% 보존**을 동시 달성.

이 방법론이 이 리포트 모든 수치의 신뢰 근거다. 이후 모든 실험은 component split을 사용한다.

---

## 3. 진단: 반증의 연쇄

극적 가설을 세우고 가장 싼 falsifying 실험으로 깨는 방식(falsify-first)으로 진행했다. 반증된 가설도 1급 결과로 기록한다.

| # | 가설 | 결과 | 핵심 |
|---|---|---|---|
| A | FaceSwap은 고주파 부호가 반대다 | 반증 | 4기법 전부 고주파 감소. FaceSwap은 변화량이 최소(DF의 1/7) |
| B | 흔적은 합성 경계 seam에 숨어있다 | 반증 | FaceSwap seam은 DF의 1/10. 경계에도 최소 |
| C | 압축이 저수준 흔적을 파괴한다 | 반증 | Δ 붕괴는 분산 스케일링 착시. 효과크기·AUC는 보존 |
| D | FaceSwap=0.52, 유일한 사각지대 | 과장 | 잔차 입력 핸디캡. RGB 모델은 0.60, NT도 0.55 |
| E | 저수준 일반의 한계인가? | 공정검증 | SBI(blending 특화)로 반증 시도 → SBI도 FS 0.64 |
| F | NT 벽 = 그냥 약한 백본 탓? | 공정검증 | CLIP L/14·LN-tune으로도 NT 0.56 요지부동 = 진짜 벽 |

### 3.1 방법론적 교훈 (재사용 가능)
1. **평균차이(Δ) ≠ 판별력.** 분산이 함께 변하면 오도한다 → Cohen's d / AUC로 검증.
2. **작은 n의 상관은 outlier 하나가 만든다.** seam↔난이도 "단조 관계"는 DF 한 점의 착시였다.
3. **"그 부류가 안 된다"는 그 부류의 SOTA로 공정 반증하라.** 비특화 구현의 한계일 수 있다.

---

## 4. 4패러다임 벤치마크

"가짜의 정체가 어디 있나"에 대한 근본적으로 다른 네 가설을, 같은 누수0 · LOMO(미지 기법) 프로토콜에서 정면 비교했다.

**LOMO cross-manipulation AUC (± std, 5-fold):**

| 패러다임 (모델) | Deepfakes | Face2Face | FaceSwap | NeuralTextures | in-dist |
|---|---|---|---|---|---|
| 외형 (ConvNeXt) | 0.824 | 0.652 | 0.582 | 0.549 | **0.929** |
| 잔차 (SRM+Bayar) | 0.782 | 0.652 | 0.524 | 0.572 | 0.849 |
| blending (SBI) | 0.798 | 0.534 | 0.644 | 0.480 | — |
| foundation (CLIP L/14) | **0.923** | 0.673 | 0.774 | 0.556 | 0.832 |

**핵심 발견:**
- **NeuralTextures는 4패러다임 공통의 벽** — 전부 ≤0.57. 백본 스케일(B/16→L/14)도 미세조정(LN-tune 0.562)도 NT만 못 움직인다 → **모델 용량이 아니라 정보 수준의 벽.**
- **FaceSwap "사각지대"는 약한 백본 탓이었다** — CLIP L/14가 0.774. 강한 foundation이면 잘 잡는다.
- **백본 이원화:** in-dist는 ConvNeXt(0.929) 승, cross-manip은 CLIP L/14 승.

---

## 5. 돌파구: 시간 축

### 5.1 진단
시도한 네 축은 전부 **단일 프레임 공간 신호**였다. NeuralTextures는 입 텍스처를 프레임마다 신경망이 재생성한다 → 공간적으론 조용해도 **시간적으론 flicker**. 진단이 지목한 직교축 = 시간.

### 5.2 검증과 대조군
얼굴 크롭 연속프레임(원본 비디오 불필요)에서 **픽셀 시간특징 8개** + LogReg:

| method | 공간(CLIP L/14) | 시간(픽셀 8feat) |
|---|---|---|
| Deepfakes | 0.923 | 0.729 |
| Face2Face | 0.673 | 0.707 |
| **FaceSwap** | **0.774** | **0.513** (시간축 무신호) |
| **NeuralTextures** | 0.556 | **0.777** (시간축 강신호) |

**NT 0.556 → 0.777**(손특징 8개 + 로지스틱만으로). **FaceSwap 대조군(0.513)이 결정적:** temporal이 "가짜 티"를 잡는 거였으면 FaceSwap도 높아야 하는데 우연 수준 → temporal은 **진짜 얼굴 동역학 조작**을 잡는다(재연 계열↑, 정적 paste↓). **공간(FaceSwap)×시간(NeuralTextures) 직교 상보성 확정.**

문헌 확인: NT 최난이도 = 문헌 동의. NT 최선책 = temporal/mouth(LipForensics). **진단이 문헌 SOTA 방향과 독립적으로 일치.**

### 5.3 딥 시간 브랜치
손특징은 ~0.78에서 포화(다영역 27feat+GBM으로도 0.756). **mc3_18(입crop 16프레임 3D-CNN, Kinetics 사전학습)** 엔드투엔드 학습 → in-dist per-method:

| method | 공간 | 손 temporal | 딥 temporal |
|---|---|---|---|
| Deepfakes | 0.926 | 0.734 | 0.919 |
| Face2Face | 0.786 | 0.713 | 0.890 |
| FaceSwap | 0.878 | 0.516 | 0.869 |
| **NeuralTextures** | 0.707 | 0.784 | **0.873** |

NT를 0.71/0.78 → **0.873**으로 끌어올렸고, **약한 기법이 하나도 없다**(전부 0.87~0.92).

---

## 6. 최종 시스템

### 6.1 아키텍처

    video/frames → 얼굴검출(MTCNN) → [공간] 전체얼굴 → ConvNeXt  (외형 아티팩트)
                                    [시간] 입클립(16f) → 3D-CNN  (텍스처 flicker)
                                  → robust fusion → fake 확률

### 6.2 성능 (in-dist, 누수0, ConvNeXt 공간 + 딥 temporal)

| method | 공간 | 시간 | 융합 |
|---|---|---|---|
| Deepfakes | 0.975 | 0.919 | 0.975 |
| Face2Face | 0.939 | 0.892 | 0.945 |
| FaceSwap | 0.956 | 0.870 | 0.950 |
| **NeuralTextures** | 0.825 | 0.873 | **0.891** |
| **전체 AUC** | 0.924 | 0.888 | **0.940** |

**4기법 전부 ≥0.89, 구멍 없음.** NT(4패러다임 벽, 0.55) → 최종 제품 0.891. stacking ≈ mean(둘 다 0.940) → 게이트 불필요, 단순 평균으로 충분.

### 6.3 작동 데모 (infer.py)

| 입력 | 공간 | 시간 | 판정 |
|---|---|---|---|
| original (real) | 0.014 | 0.007 | REAL |
| Deepfakes | 0.918 | 0.181 | FAKE (공간이 잡음) |
| FaceSwap | 0.978 | 0.880 | FAKE |
| NeuralTextures | 0.829 | 0.898 | FAKE (시간이 잡음) |
| **SimSwap (미지 조작)** | 0.068 | 0.929 | FAKE (시간이 미지 조작 포착) |

산출물: infer.py(CLI), app.py(영상 업로드 웹앱, werkzeug), 5+3 fold 앙상블 가중치. GitHub 팀 저장소에 배포.

---

## 7. 일반화 검증 (cross-dataset)

FF++로 학습한 판별기를 미지의 Celeb-DF v2에 적용(n=2,086).

| 브랜치 | AUC | 3-branch 융합 | AUC |
|---|---|---|---|
| **공간 (ConvNeXt)** | **0.800** | 공간+시간 | 0.785 |
| 시공간 (3D-CNN) | 0.634 | 공간+CLIP | 0.766 |
| CLIP L/14 | 0.709 | 3-branch 평균 | 0.771 |

### 7.1 두 가지 정직한 교정 (일화 ≠ 벤치마크)
- **G. "temporal이 미지 조작에 더 강함" → 반증.** Celeb-DF는 swap 계열이라 시간적으로 안정 → temporal 0.634 « spatial 0.800. (SimSwap 한 케이스는 운 좋은 일화였다.)
- **H. "CLIP이 cross-dataset도 올림" → 반증.** CLIP 0.709 < ConvNeXt 0.800. **cross-manip ≠ cross-dataset:** frozen CLIP은 cross-manip 최강이지만, 미지 데이터셋엔 **끝까지 fine-tune한 ConvNeXt가 더 잘 전이**된다.

**결론:** 제품은 FF++에 갇히지 않고 미지 데이터셋에 **AUC 0.80**으로 일반화(과적합 아님). 정직하게 견고하되 SOTA(0.85~0.93)엔 못 미친다. 브랜치 분해가 모든 발견(swap=공간, 재연=시간)과 일관.

---

## 8. 게이트 실험 (사전 등록)

**질문:** 상보성이 실재한다면, 입력 조건부 라우터(게이트)가 고정 fusion을 이기는가? in-dist에선 두 브랜치가 모두 강해 게이트가 할 일이 없으므로 **LOMO에서** 검증. 게이트는 조작 라벨을 절대 보지 못하며, 누수0 프로토콜(held_out≠M & fold≠f로 학습 → held_out=M & fold=f로 평가)로 학습.

| held-out | spatial | temporal | mean | stacking | gate | oracle |
|---|---|---|---|---|---|---|
| Deepfakes | 0.897 | 0.723 | 0.896 | 0.910 | 0.911 | 0.897 |
| Face2Face | 0.679 | 0.703 | 0.744 | 0.721 | 0.735 | 0.708 |
| FaceSwap | 0.744 | 0.510 | 0.643 | 0.636 | 0.650 | 0.744 |
| NeuralTextures | 0.588 | 0.773 | 0.726 | 0.637 | 0.668 | 0.773 |
| **평균** | 0.727 | 0.677 | **0.752** | 0.726 | **0.741** | **0.781** |

**판정: 실패(사전 등록한 실패 케이스).** 게이트(0.741)가 단순 mean fusion(0.752)보다 나쁘고, 더 나은 브랜치에 근접 못 한다(FS 0.650«0.744, NT 0.668«0.773). 게이트 weight가 라우팅을 거꾸로/무의미하게 냈다(FaceSwap→0.45, NeuralTextures→0.69, 필요한 방향과 반대).

**정밀한 결론(oracle 0.781이 못박음):** 완벽 라우팅은 실제로 이득이 있고(oracle » mean) fusion 코드도 무결하다. 즉 **상보성은 원리적으로 착취 가능하지만, 관찰 가능한 feature로는 미지 조작을 라우팅하는 정보를 얻을 수 없다.** → mean fusion이 최선의 단순 선택.

---

## 9. 종합 결과

- **방법론:** component split — 신원 누수 0, fake 100% 보존 (이전 프로토콜 지표 부풀림 규명)
- **벤치마크:** 4패러다임(외형·잔차·blending·foundation) × 누수0 × 오차막대
- **핵심 발견:** 조작 계열마다 최적 축이 다르다 — swap=공간, 재연=시간. NT는 정보수준 벽
- **돌파:** 시간 축이 NT를 0.55 → 0.89로 뚫음 (진단이 설계를 정당화)
- **제품:** 공간+시간 이중브랜치, in-dist 0.940, 구멍 없음, 실행 가능한 판별기+웹앱
- **일반화:** Celeb-DF cross-dataset 0.80 (과적합 아님)
- **게이트:** 상보성 실재하나 라우팅은 학습 불가 — mean fusion이 최선 (사전등록 실패=1급 결과)

반증된 가설 8개(A~F + G·H)와 게이트 실패가 이 프로젝트의 지적 골격이다.

---

## 10. 기여 · 한계 · 향후

### 10.1 기여 (정직한 평가)
개별 부품(ConvNeXt·3D-CNN·CLIP·temporal)은 알려진 기술이고, "NT가 어렵다 / temporal이 잡는다"도 문헌이 안다. **이 작업의 가치는 완성도 + 누수0 벤치마크 + 정직한 실패 지도 + 진단→설계 논리**, 그리고 그 전부가 재현 가능한 코드로 남았다는 데 있다.

### 10.2 한계
- cross-dataset 0.80은 견고하나 SOTA(0.85~0.93)엔 못 미침.
- 시간 브랜치는 재연(NT) 전문가 — swap 중심 데이터(Celeb-DF)엔 약함.
- 게이트가 안 되므로 조작 계열별 최적화는 고정 fusion에 의존.
- Celeb-DF 평가는 공식 518 test split이 아닌 전체/샘플(리스트 부재).

### 10.3 향후 방향 (독창성 후보)
1. **Temporal Self-Blend** — SBI(공간 self-blend)가 못 잡는 재연을, 그 정체를 real 영상에 합성 주입한 "시간적 pseudo-fake"로 학습. (→ **§11에서 수행**: 정체는 flicker가 아니라 **과평활**이었고, 일반 개념은 선행연구(STRD·VB)임을 확인. 생존 각도는 "기존과 반대 방향".)
2. **시간-주파수 서명** — 영역별 시간 신호의 주파수 스펙트럼. 자연 모션=저주파, 재연=부자연 고주파. 공간 주파수는 포화, 시간 주파수는 덜 탐구 + 해석 가능.
3. **조건부 재조작 응답** — 원래 challenge-response를 공간×시간 프로파일 이동의 관점으로 부활.

---

## 11. 확장: 재연 = 과평활 방향과 강건성 aug

§10.3에서 후보로 남긴 "Temporal Self-Blend"를 실제로 수행하고 정직하게 정정했다.

### 11.1 정체 규명 — flicker가 아니라 과평활
NeuralTextures의 시간 서명은 떨림(flicker)이 아니라 **텍스처 시간 과평활**(프레임 간 미세변동 상실)이었다. 첫 flicker 가설은 반증(NT 0.558)됐고, 특징 분석으로 방향을 과평활로 정정하니 통과(NT 0.700). 광학흐름 모션보정 후에도 텍스처 분리력이 남아(d_resid AUC 0.762 > 모션 0.721) **모션 무관**임을 확인했다.

### 11.2 방향이 핵심 — 기존 self-blend와 정반대
| 합성 방향 | NeuralTextures(재연) AUC |
|---|---|
| 과평활(우리) | 0.70 |
| 교란 = STRD | 0.22 (반전) |
| warp-drift = VB | 0.31 (반전) |

기존 temporal self-blend(STRD arXiv:2207.10402, VB 2408.17065)는 시간 **교란/드리프트** 방향인데, 재연을 잡으려면 **정반대(과평활)**가 필요하다.

### 11.3 일반화 — 미지 재연으로 전이 (LOMO)
swap만 지도학습하고 재연을 held-out(leave-reenactment-out):

| 방식 | 재연 held-out AUC |
|---|---|
| A 지도(swap만) | 0.706 |
| B 지도 + 과평활 aug | 0.733 (NT +0.052) |
| C real-only 과평활 | 0.644 (단 NT 0.701 > A) |

이득이 과평활 지배 기법(NT)에만 몰려(F2F +0.001) 메커니즘 근거가 됐다.

### 11.4 배포 — 앙상블 hedge + 캘리브레이션
- 직접 교체(aug만): in-dist −0.010(FaceSwap 붕괴) → **기각**.
- **6모델 앙상블**(표준3 + 과평활-aug3): in-dist 무손실(+0.004, NT +0.009), Celeb-DF cross 시공간 AUC +0.091.
- 캘리브레이션(FF++ Platt fit, Celeb-DF 독립검증): 고정 0.5 BACC FF++ +0.081 / Celeb +0.050, AUC 불변.

**정직한 위치.** 일반적 "temporal self-blend" 개념은 선행연구. 생존한 좁은 각도는 "재연 = 과평활, 기존과 반대 방향"(모션 통제)이며, aug는 in-dist 부스터가 아니라 **일반화 도구**임을 반증을 통해 정정했다. 절대 이득은 완만(+0.004~0.055).

---

## 12. XAI: 설명가능성과 충실도 검증

"어디가 fake인가"를 시각화하되, 히트맵은 '모델 주목 영역'이지 물증이 아니므로 **충실도를 측정**했다.

### 12.1 시공간 Grad-CAM 충실도 (falsify-first)
정답 = FF++ 풀프레임 |fake − original|(픽셀정렬, 배경 diff≈0.6). 크롭 diff는 프레임별 독립 검출로 정렬이 깨져(입 8.5 < 나머지 24.1) 풀프레임으로 교정했다(반증→수정).

| | pointing-game | CAM-vs-mask AUC |
|---|---|---|
| 학습 6모델 | 0.516 | 0.706 |
| 랜덤가중치(Adebayo sanity) | 0.214 | 0.591 |
| 우연 | 0.200 | 0.500 |

학습 CAM이 실제 조작영역을 **우연의 2.6배**로 국소화하고, 랜덤모델은 우연 수준(Adebayo sanity 통과) = 충실도가 **학습에서 비롯**. "그럴듯하지만 무의미"(Adebayo et al. 2018)를 반증했다.

### 12.2 웹 통합과 정직한 한계
`/api/predict`가 판정 + 공간·시공간 Grad-CAM 히트맵(base64) + 브랜치 반응을 반환한다. 두 가지를 정직하게 표기: (1) 히트맵은 주목영역(충실도 병기), 물증 아님. (2) 브랜치 기반 계열판별(swap/재연)은 per-sample 신뢰 못 함(공간분기가 재연에도 강하게 반응) → family를 '단정'에서 '서술 + 가설'로 낮췄다. pointing 0.52라 절반은 빗나간다.

---

## 부록 A. 실험 인덱스

| 실험 | 내용 | 스크립트 |
|---|---|---|
| 진단 1·2 | 고주파 부호 / 경계 seam | hf_sign_probe.py, hf_boundary_probe.py |
| 진단 3 | residual SRM LOMO | train_remanip_srm.py |
| 진단 4~7 | 압축 붕괴 / 검증 / 페어분해 / 효과크기 | degrade_collapse.py, verify_degrade.py, paired_breakdown.py, effect_size.py |
| 진단 8 | APP/SRM LOMO per-method | train_direct_face_classifier.py |
| 진단 9 | SBI 공정 baseline | train_sbi_baseline.py |
| CLIP 배터리 | frozen B/16·L/14 + LN-tune | clip_probe_lomo.py, clip_lntune.py |
| 시간 | 픽셀/enriched/딥 temporal | temporal_probe.py, temporal_enriched.py, clip_temporal.py |
| 융합 | in-dist / 개선(ConvNeXt+stacking) | fusion_probe.py, improved_fusion.py |
| cross-dataset | Celeb-DF 2·3 branch | eval_celebdf.py, eval_celebdf_3branch.py |
| 게이트 | OOF 생성 + 사전등록 실험 | gate_oof_generate.py, gate_experiment.py |
| §11 과평활 방향 | 정체 규명·방향 대조·confound | temporal_selfblend_probe.py, direction_compare.py, confound_check.py |
| §11 aug 검증 | LOMO·강도 스윕·배포 재학습 | lomo_verify.py, sweep_aug.py, deploy_retrain.py, deploy_finalize.py |
| §11 캘리브레이션 | Platt 보정(FF++ fit, Celeb 검증) | calibrate_temporal.py |
| §12 XAI | Grad-CAM·충실도(pointing) | gradcam_demo.py, temporal_faithfulness.py, xai.py |
| cross-dataset 준비 | 범용 얼굴 추출기 | extract_faces_generic.py |
| 제품 | 추론/웹앱(XAI 통합) | infer.py, app.py |

## 부록 B. 관련 문서
- EXPERIMENT_LOG_2026-07-09.md — 시간순 상세 로그(반증 과정 포함)
- deepfake_detector/ — 배포 가능한 판별기(코드+가중치+발표페이지). GitHub seelhwan 브랜치.

---

## 부록 C. 방법론 상세 (모델 · 원리 · 수식)

### C.1 전처리
프레임별 얼굴을 MTCNN[9]으로 검출·정렬해 224×224 크롭. **공간 브랜치**는 전체 얼굴, **시간 브랜치**는 입영역 ROI(세로 [0.50, 0.96]·가로 [0.16, 0.84] 비율)를 112×112로 리샘플하고 클립 길이 T=16. 정규화는 공간=ImageNet, 시간=Kinetics 통계.

### C.2 공간 브랜치 — ConvNeXt-tiny [1]
ConvNeXt 블록의 핵심 연산(입력 x, 채널 C):
```
z = DWConv7×7(x)                                   # depthwise: 채널별 큰 커널 공간 혼합
z = LayerNorm(z)
z = Linear(C→4C); z = GELU(z); z = Linear(4C→C)     # inverted bottleneck (Transformer MLP)
x_out = x + DropPath(z)                              # residual
```
- **Patchify stem**: 4×4 stride-4 conv로 비겹침 패치 임베딩(ViT patch embedding 모방).
- **depthwise 7×7**은 self-attention의 "공간 위치 혼합"을 큰 커널로 근사 → 넓은 수용영역을 저비용으로.
- 계층 깊이 (3,3,9,3), 채널 (96,192,384,768). ImageNet-22k 사전학습 → 1k 미세조정[10].
- 분류 head: h = Linear(LayerNorm(GAP(feat))), p_spatial = σ(h). **5-fold 앙상블** 평균.

**왜 swap에 맞나:** 얼굴 교체는 blending 경계·색/질감 불일치를 **단일 프레임 공간 패턴**으로 남긴다. 강한 in22k 백본의 넓은 수용영역이 이를 포착한다.

### C.3 시간 브랜치 — mc3_18 (Mixed 3D Convolution) [2]
3D 컨볼루션(입력 X ∈ ℝ^{C×T×H×W}):
```
Y(c',t,i,j) = Σ_c Σ_{τ,u,v} W(c',c,τ,u,v) · X(c, t+τ, i+u, j+v) + b
```
시간 인덱스 τ가 합에 포함 → 프레임 간 동역학을 **직접** 학습.

**MC3의 3D→2D 전환(핵심):** ResNet-18 골격에서 **초기** residual 그룹은 3D conv(시간 커널 k_t=3), **후기** 그룹은 2D conv(k_t=1 → 시간 혼합 없음)로 둔다.
- 근거[2]: 모션 정보는 저수준(초기 층)에서 가장 유용하고, 깊어질수록 정적 의미표현으로 충분하다. 순수 3D(R3D)는 파라미터·연산이 크고 과적합 위험이 커 — MC3는 **초기만 3D로 모션을 잡고 이후 2D로 효율화**한다(R2+1D는 3D를 공간2D+시간1D로 분해하는 또 다른 절충).
- 딥페이크 해석: 재연의 이상은 "입 텍스처가 프레임마다 어떻게 변하나"라는 **저수준 동역학** → 초기 3D 층이 이를 포착. 입력을 입 ROI로 제한해 신호를 증폭(전체 얼굴 3D-CNN은 방향 효과가 희석됨을 실측).
- Kinetics-400 사전학습[3]. head: p_temporal = σ(Linear(GAP_{t,h,w}(feat))). **6-모델 앙상블**(§C.5).

### C.4 과평활 pseudo-fake (real-only) [ours; cf. 4,5,6]
입 ROI 프레임열 {x_t}에 시간 이동평균 x̄_t = (x_{t-1}+x_t+x_{t+1})/3 를 정의하고, 혼합비 α ∼ U(0.55, 0.8)로
```
x'_t = (1−α)·x_t + α·x̄_t        (ROI 내, t=1..T)
```
프레임 간 미세변동을 감쇠(과평활)한 "시간적 가짜". 원본을 음성(0), x'을 양성(1)으로 학습해 **"재연=과평활"** 귀납편향을 주입한다. 기존 self-blend(SBI[4]는 공간, STRD[5]·VB[6]는 시간 **교란/드리프트**)와 방향이 정반대 — 이것이 규명·검증된 좁은 각도.

### C.5 앙상블 + Platt 캘리브레이션 [7,8]
앙상블: p_temporal = (1/M) Σ_m σ(f_m(clip)), M=6 (표준 3 + 과평활-aug 3).
Platt scaling[7]: 로짓 ℓ = log(p/(1−p))에 대해 보정확률
```
q = σ(a·ℓ + b),   (a,b) = argmin −Σ_i [ y_i log q_i + (1−y_i) log(1−q_i) ]   (NLL)
```
FF++ leak-free 검증점에서 fit. 단조변환이라 **AUC(순위) 불변**, 의사결정 임계 0.5만 재정렬한다. Celeb-DF로 독립검증(전이 확인). a<1이면 온도 스케일링[8]과 동형(과신뢰 완화). 실측 a=0.714, b=−0.907.

### C.6 융합·판정 (robust verdict)
```
p_fused = (p_spatial + p_temporal)/2
ŷ = FAKE  ⟺  p_fused ≥ 0.5   ∨   max(p_spatial, p_temporal) ≥ 0.85
```
OR-0.85는 단일 축만 강하게 반응하는 미지 조작(swap만·재연만)에 대한 재현율 보강.

### C.7 Component split (union-find)
fake video_id "A_B"를 간선 (A,B)로 보고 union-find로 연결성분을 형성, KFold를 성분 인덱스에 적용해 fold를 나눈다 → 모든 신원 n에 대해 fold(n)이 유일 → train/test 신원 교집합 = ∅. fake 전량 보존(strict split은 86% 폐기).

### C.8 Grad-CAM 수식 [11] + 충실도
공간(2D, 특징맵 A^k ∈ ℝ^{H×W}, 점수 y):
```
α_k = (1/Z) Σ_i Σ_j ∂y/∂A^k_{ij}          # 채널 중요도 = 그래디언트 GAP
L = ReLU( Σ_k α_k A^k )                    # 양의 기여만 → 업샘플
```
시간(3D, A^k ∈ ℝ^{T'×H'×W'}): α_k = (1/Z) Σ_{t,i,j} ∂y/∂A^k_{tij}, L(t,i,j)=ReLU(Σ_k α_k A^k(t,i,j)) → (T,112,112) 삼선형 업샘플로 **프레임별** 히트맵.
충실도: 정답 G = 𝟙[|fake−orig| ≥ 80퍼센타일]. pointing-game = (1/N) Σ 𝟙[argmax L ∈ G], 우연 ≈ |G|/전체 = 0.20. Adebayo sanity[12]: 랜덤가중치 모델 pointing이 우연 수준이면 충실도가 **학습에서 비롯**됨을 보장(실측 학습 0.516 vs 랜덤 0.214).

### C.9 학습 설정
손실 BCEWithLogits, 옵티마이저 AdamW(lr 3e-4, wd 1e-4), 스케줄 OneCycleLR(pct_start 0.15), AMP(fp16), 클래스균형 WeightedRandomSampler. 가중치 저장 fp16(GitHub 100MB 대응).

---

## 부록 D. 참고문헌
- [1] Z. Liu et al., "A ConvNet for the 2020s," **CVPR 2022**. (ConvNeXt)
- [2] D. Tran et al., "A Closer Look at Spatiotemporal Convolutions for Action Recognition," **CVPR 2018**. (MC3·R2+1D)
- [3] W. Kay et al., "The Kinetics Human Action Video Dataset," arXiv:1705.06950, 2017.
- [4] K. Shiohara, T. Yamasaki, "Detecting Deepfakes with Self-Blended Images," **CVPR 2022**. (SBI)
- [5] "Self-supervised Temporal Regularity Disruption for deepfake video detection," arXiv:2207.10402. (STRD)
- [6] "Video-Level Blending for temporal forgery," arXiv:2408.17065. (VB)
- [7] J. Platt, "Probabilistic Outputs for Support Vector Machines," 1999. (Platt scaling)
- [8] C. Guo et al., "On Calibration of Modern Neural Networks," **ICML 2017**. (temperature scaling)
- [9] K. Zhang et al., "Joint Face Detection and Alignment using MTCNN," IEEE SPL 2016.
- [10] J. Deng et al., "ImageNet," CVPR 2009 (+ ImageNet-21k/22k 사전학습).
- [11] R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," **ICCV 2017**.
- [12] J. Adebayo et al., "Sanity Checks for Saliency Maps," **NeurIPS 2018**.
- [13] A. Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," **ICCV 2019**.

---

*작성: 딥페이크 판별 프로젝트 · FaceForensics++ / Celeb-DF v2 · 공간×시간 이중브랜치*
*원칙: 반증된 가설도 1급 결과. 누수0 프로토콜. 일화가 아니라 벤치마크.*
