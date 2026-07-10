#!/usr/bin/env python3
"""cross-dataset 평가 — FF++로 학습한 판별기를 Celeb-DF v2에 적용.
이미 추출된 celebdf_faces(정렬 얼굴 크롭) 사용. label: real=0, fake=1(=P(fake)).
공식 518 test split 아님(리스트 없음) → Celeb-DF v2 전체/샘플. 진짜 미지 데이터셋 일반화 측정."""
import argparse, glob, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from infer import load_models, predict

FACES = "/home/tako/seelhwan/00_datasets/celebdf_v2/celebdf_faces"

def video_faces(folder, T=16):
    ps = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))[:T]
    out = []
    for p in ps:
        try: out.append(Image.open(p).convert("RGB"))
        except Exception: pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_real", type=int, default=0, help="0=전체")
    ap.add_argument("--max_fake", type=int, default=1200, help="0=전체")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv, temp = load_models(dev)
    print(f"[models] 공간 {len(conv)} + 시간 {len(temp)}  device={dev.type}", flush=True)

    reals = sorted(glob.glob(f"{FACES}/real/*"))
    fakes = sorted(glob.glob(f"{FACES}/fake/*"))
    rng = random.Random(args.seed); rng.shuffle(reals); rng.shuffle(fakes)
    if args.max_real: reals = reals[:args.max_real]
    if args.max_fake: fakes = fakes[:args.max_fake]
    items = [(f, 0) for f in reals] + [(f, 1) for f in fakes]
    print(f"[data] real {len(reals)} + fake {len(fakes)} = {len(items)} 비디오", flush=True)

    sp_s, tp_s, fu_s, ys = [], [], [], []
    for i, (folder, lab) in enumerate(items):
        faces = video_faces(folder)
        if len(faces) < 4: continue
        with torch.no_grad():
            sp, tp, fu = predict(faces, conv, temp, dev)
        sp_s.append(sp); tp_s.append(tp); fu_s.append(fu); ys.append(lab)
        if i % 200 == 0: print(f"  {i}/{len(items)}  (fused_auc so far={roc_auc_score(ys,fu_s):.3f})" if len(set(ys))>1 else f"  {i}/{len(items)}", flush=True)

    ys = np.array(ys); sp_s = np.array(sp_s); tp_s = np.array(tp_s); fu_s = np.array(fu_s)
    print(f"\n===== Celeb-DF v2 cross-dataset (FF++ 학습, n={len(ys)}) =====")
    for name, s in [("공간(ConvNeXt)", sp_s), ("시공간(3D-CNN)", tp_s), ("융합(fusion)", fu_s)]:
        auc = roc_auc_score(ys, s)
        bacc = balanced_accuracy_score(ys, (s >= 0.5).astype(int))
        print(f"  {name:18s} AUC={auc:.3f}  BACC@0.5={bacc:.3f}")
    # robust 규칙(mean≥.5 or 브랜치≥.85)의 정확도
    conf = np.maximum(sp_s, tp_s); pred = ((fu_s >= 0.5) | (conf >= 0.85)).astype(int)
    print(f"  robust 규칙 BACC={balanced_accuracy_score(ys, pred):.3f}")
    print(f"\n  참고: FF++ in-dist 0.940 → Celeb-DF는 완전 미지 데이터셋(다른 신원·조작·출처)")

if __name__ == "__main__":
    main()
