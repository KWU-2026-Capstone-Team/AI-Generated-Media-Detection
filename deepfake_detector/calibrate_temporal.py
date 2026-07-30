#!/usr/bin/env python3
"""신규 6모델 temporal 앙상블 점수 캘리브레이션.
문제: 앙상블로 AUC↑ 지만 점수분포 이동 → 고정 0.5 BACC↓ (Celeb 융합 0.719→0.549).
해결: temporal 점수를 FF++(학습분포)에서 leak-free로 Platt 보정(logit 스케일+바이어스),
      Celeb-DF(미지)에서 독립 검증. Celeb에 맞추지 않음(반칙 방지). spatial 불변.
출력: models/temporal/calibration.json {a,b,thr}  (infer.py가 로드해 적용)
"""
import glob, json, os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer
from infer import make_temporal, FaceClassifier, mouth
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from collections import defaultdict

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_MEAN, K_STD, T = infer.K_MEAN, infer.K_STD, infer.T
IMG_MEAN, IMG_STD = infer.IMG_MEAN, infer.IMG_STD
ttf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize(K_MEAN, K_STD)])
ctf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)])
FF_CSV = "/home/tako/seelhwan/ffpp_pair_branch_context_experiment/cache/face_x_context.csv"
CELEB = "/home/tako/seelhwan/00_datasets/celebdf_v2/celebdf_faces"
BACKUP = sorted(glob.glob("models/temporal_backup_*"))[0]

def load_temporal(paths):
    ms = []
    for p in paths:
        ck = torch.load(p, map_location=dev); m = make_temporal().to(dev)
        m.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["state_dict"].items()}); m.eval(); ms.append(m)
    return ms

@torch.no_grad()
def temp_score(models, faces):
    sel = faces[:T] if len(faces) >= T else faces + [faces[-1]] * (T - len(faces))
    clip = torch.stack([ttf(mouth(f)) for f in sel], 1).unsqueeze(0).to(dev)
    return float(np.mean([torch.sigmoid(m(clip)).item() for m in models]))

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

# ---------- 1) FF++ leak-free 캘리브레이션 fit ----------
df = pd.read_csv(FF_CSV, dtype={"video_id": str, "frame": str})
# component split (학습과 동일 seed 42) → 각 fold의 test 비디오를 그 fold 모델로만 채점(leak-free)
par = {}
def find(x):
    par.setdefault(x, x); r = x
    while par[r] != r: r = par[r]
    while par[x] != r: par[x], x = r, par[x]
    return r
def uni(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: par[ra] = rb
for v in df.loc[df.label.astype(int) == 1, "video_id"].unique(): a, b = v.split("_", 1); uni(a, b)
for o in df.loc[df.label.astype(int) == 0, "video_id"].unique(): find(o)
comps_d = defaultdict(list)
for n in list(par.keys()): comps_d[find(n)].append(n)
comps = [sorted(c) for c in comps_d.values()]; node2comp = {n: ci for ci, c in enumerate(comps) for n in c}
comp_fold = np.zeros(len(comps), int)
kf = KFold(5, shuffle=True, random_state=42)
for fi, (_, te) in enumerate(kf.split(np.arange(len(comps))), 1): comp_fold[list(te)] = fi

# fold별 신규 모델(base_fi+aug_fi) 로드 (fi=1..3만 학습에 test로 쓰였음 → leak-free)
fold_models = {fi: load_temporal([f"models/temporal/temporal_base_fold{fi}.pt", f"models/temporal/temporal_aug_fold{fi}.pt"]) for fi in (1, 2, 3)}
print(f"[fit] FF++ leak-free 채점 (fold1-3 test), 신규 temporal", flush=True)
sc_ff, y_ff = [], []
groups = list(df.groupby(["method", "video_id"], sort=False))
import random as _r; _r.Random(0).shuffle(groups)
per_fold_cap = {1: 0, 2: 0, 3: 0}
for (m, v), g in groups:
    node = v.split("_")[0] if "_" in v else v
    if node not in node2comp: continue
    fo = comp_fold[node2comp[node]]
    if fo not in (1, 2, 3): continue
    if per_fold_cap[fo] >= 500: continue
    ps = g.sort_values("frame")["face_X_path"].tolist()[:T]
    faces = [Image.open(p).convert("RGB") for p in ps if os.path.exists(p)]
    if len(faces) < 4: continue
    sc_ff.append(temp_score(fold_models[fo], faces)); y_ff.append(0 if m == "original" else 1)
    per_fold_cap[fo] += 1
sc_ff = np.array(sc_ff); y_ff = np.array(y_ff)
print(f"[fit] n={len(y_ff)} (real {int((y_ff==0).sum())}, fake {int((y_ff==1).sum())})", flush=True)

# Platt: logit(score) → label
clf = LogisticRegression(C=1e6, solver="lbfgs").fit(logit(sc_ff).reshape(-1, 1), y_ff)
a = float(clf.coef_[0, 0]); b = float(clf.intercept_[0])
def calib(p): return 1 / (1 + np.exp(-(a * logit(p) + b)))
# 최적 임계값(원점수 기준, 참고용)
ths = np.linspace(0.1, 0.9, 81); baccs = [balanced_accuracy_score(y_ff, (sc_ff >= t).astype(int)) for t in ths]
thr_star = float(ths[int(np.argmax(baccs))])
print(f"[fit] Platt a={a:.3f} b={b:.3f}  (원점수 최적임계 thr*={thr_star:.3f})", flush=True)
print(f"[fit] FF++ temporal BACC@0.5 {balanced_accuracy_score(y_ff,(sc_ff>=0.5)):.3f} → 보정후 {balanced_accuracy_score(y_ff,(calib(sc_ff)>=0.5)):.3f}  (AUC {roc_auc_score(y_ff,sc_ff):.3f} 불변)", flush=True)

# ---------- 2) Celeb-DF 독립 검증 (fit에 안 씀) ----------
conv = []
for p in sorted(glob.glob("models/spatial/*.pt")):
    ck = torch.load(p, map_location=dev); mm = FaceClassifier(ck.get("backbone", "convnext_tiny.fb_in22k_ft_in1k")).to(dev)
    mm.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["state_dict"].items()}); mm.eval(); conv.append(mm)
tnew = load_temporal(sorted(glob.glob("models/temporal/*.pt")))
reals = sorted(glob.glob(f"{CELEB}/real/*")); fakes = sorted(glob.glob(f"{CELEB}/fake/*"))
_r.Random(0).shuffle(reals); _r.Random(0).shuffle(fakes); fakes = fakes[:1200]
items = [(f, 0) for f in reals] + [(f, 1) for f in fakes]
sp_s, tp_s, ys = [], [], []
for folder, lab in items:
    ps = sorted(glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.png"))[:T]
    faces = [Image.open(p).convert("RGB") for p in ps]
    if len(faces) < 4: continue
    with torch.no_grad():
        xs = torch.stack([ctf(f) for f in faces]).to(dev)
        sp = float(np.mean([torch.sigmoid(mm(xs)).mean().item() for mm in conv]))
    tp_s.append(temp_score(tnew, faces)); sp_s.append(sp); ys.append(lab)
ys = np.array(ys); sp_s = np.array(sp_s); tp_s = np.array(tp_s); tp_c = calib(tp_s)
def rep(name, s): print(f"  {name:26s} AUC={roc_auc_score(ys,s):.3f}  BACC@.5={balanced_accuracy_score(ys,(s>=0.5)):.3f}")
print(f"\n===== Celeb-DF 독립검증 (n={len(ys)}) — 보정 전/후 =====", flush=True)
rep("시공간 보정전", tp_s); rep("시공간 보정후", tp_c)
rep("융합 보정전 (sp+tp)/2", (sp_s + tp_s) / 2); rep("융합 보정후 (sp+tp_c)/2", (sp_s + tp_c) / 2)

json.dump({"method": "platt_on_logit", "a": a, "b": b, "thr_star_raw": thr_star,
           "note": "calibrated = sigmoid(a*logit(temporal_score)+b); infer.py가 temporal 점수에 적용"},
          open("models/temporal/calibration.json", "w"), indent=2)
print("\n[save] models/temporal/calibration.json")
