#!/usr/bin/env python3
"""Celeb-DF cross-dataset 3-브랜치 — 공간(ConvNeXt) + 시공간(3D-CNN) + CLIP L/14.
CLIP 헤드는 FF++ 캐시 feature(frozen CLIP L/14)에 LogReg 학습 → Celeb-DF에 적용.
CLIP이 cross-dataset을 올리나 확인."""
import argparse, glob, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/tako/seelhwan/ffpp_pair_branch_context_experiment/scripts")
import numpy as np, torch, timm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from infer import load_models, predict, mouth  # 공간+시간 재사용
from clip_probe_lomo import annotate_components, sample_frames_per_video

FFPP = "/home/tako/seelhwan/ffpp_pair_branch_context_experiment"
CLIP_CACHE = f"{FFPP}/results/clip_probe_L14/feats_vit_large_patch14_clip_224.openai_20f.npy"
FF_CSV = f"{FFPP}/cache/face_x_context.csv"
FACES = "/home/tako/seelhwan/00_datasets/celebdf_v2/celebdf_faces"
CLIP_MODEL = "vit_large_patch14_clip_224.openai"

def train_clip_head():
    import pandas as pd
    feats = np.load(CLIP_CACHE)
    df = sample_frames_per_video(annotate_components(pd.read_csv(FF_CSV, dtype={"video_id": str, "frame": str})), 20, 42).reset_index(drop=True)
    df["idx"] = np.arange(len(df))
    X, y = [], []
    for (m, v), g in df.groupby(["method", "video_id"]):
        X.append(feats[g["idx"].values].mean(0)); y.append(int(g["label"].iloc[0]))
    X = np.array(X); y = np.array(y)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    # FF++ in-dist sanity (train=test, 낙관적이지만 헤드 정상 여부 확인)
    p = clf.predict_proba(sc.transform(X))[:, 1]
    print(f"[clip-head] FF++ 학습 {len(y)}비디오, train-fit AUC={roc_auc_score(y, p):.3f} (헤드 정상)", flush=True)
    return sc, clf

def video_faces(folder, T=16):
    ps = sorted(glob.glob(os.path.join(folder, "*.jpg")))[:T]
    out = []
    for p in ps:
        try: out.append(Image.open(p).convert("RGB"))
        except Exception: pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_real", type=int, default=0); ap.add_argument("--max_fake", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv, temp = load_models(dev)
    sc, clf = train_clip_head()
    clipm = timm.create_model(CLIP_MODEL, pretrained=True, num_classes=0).to(dev).eval()
    cfg = timm.data.resolve_data_config({}, model=clipm); ctf = timm.data.create_transform(**cfg)
    print(f"[models] ConvNeXt {len(conv)} + temporal {len(temp)} + CLIP L/14  device={dev.type}", flush=True)

    reals = sorted(glob.glob(f"{FACES}/real/*")); fakes = sorted(glob.glob(f"{FACES}/fake/*"))
    rng = random.Random(args.seed); rng.shuffle(reals); rng.shuffle(fakes)
    if args.max_real: reals = reals[:args.max_real]
    if args.max_fake: fakes = fakes[:args.max_fake]
    items = [(f, 0) for f in reals] + [(f, 1) for f in fakes]
    print(f"[data] real {len(reals)} + fake {len(fakes)} = {len(items)}", flush=True)

    SP, TP, CL, Y = [], [], [], []
    for i, (folder, lab) in enumerate(items):
        faces = video_faces(folder)
        if len(faces) < 4: continue
        with torch.no_grad():
            sp, tp, _ = predict(faces, conv, temp, dev)
            x = torch.stack([ctf(f) for f in faces]).to(dev)
            feat = clipm(x).float().mean(0).cpu().numpy()
            cl = float(clf.predict_proba(sc.transform(feat[None]))[0, 1])
        SP.append(sp); TP.append(tp); CL.append(cl); Y.append(lab)
        if i % 300 == 0 and len(set(Y)) > 1:
            print(f"  {i}/{len(items)}  clip_auc={roc_auc_score(Y, CL):.3f}", flush=True)

    Y = np.array(Y); SP = np.array(SP); TP = np.array(TP); CL = np.array(CL)
    def auc(s): return roc_auc_score(Y, s)
    print(f"\n===== Celeb-DF v2 cross-dataset · 3-branch (n={len(Y)}) =====")
    print(f"  [단일] 공간 ConvNeXt   AUC={auc(SP):.3f}")
    print(f"  [단일] 시공간 3D-CNN    AUC={auc(TP):.3f}")
    print(f"  [단일] CLIP L/14       AUC={auc(CL):.3f}")
    print(f"  ─ 융합 ─")
    print(f"  공간+시간 (기존 제품)   AUC={auc((SP+TP)/2):.3f}")
    print(f"  공간+CLIP              AUC={auc((SP+CL)/2):.3f}")
    print(f"  CLIP+시간              AUC={auc((CL+TP)/2):.3f}")
    print(f"  3-branch 평균          AUC={auc((SP+TP+CL)/3):.3f}")
    print(f"  공간+CLIP+시간(2:2:1)  AUC={auc((2*SP+2*CL+TP)/5):.3f}")
    print(f"\n  참고: 기존 공간+시간 0.785 → CLIP 추가 효과 확인")

if __name__ == "__main__":
    main()
