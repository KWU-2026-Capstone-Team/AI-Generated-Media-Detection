#!/usr/bin/env python3
"""공간 브랜치(ConvNeXt) Grad-CAM 프로토타입 — '얼굴 어디를 보고 fake라 했나' 시각화.
모델이 주목한 영역(설명)이지 '여기가 가짜라는 증거'는 아님(정직 표기). 4기법 샘플 히트맵 격자 생성."""
import sys, os, glob, numpy as np, torch, torch.nn.functional as F, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer
from infer import FaceClassifier
from PIL import Image
from torchvision import transforms

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(infer.IMG_MEAN, infer.IMG_STD)])

MODELS = []   # 배포와 동일한 5-fold 앙상블 (점수·CAM 모두 평균)
for p in sorted(glob.glob("models/spatial/*.pt")):
    ck = torch.load(p, map_location=dev)
    mm = FaceClassifier(ck.get("backbone", "convnext_tiny.fb_in22k_ft_in1k")).to(dev)
    mm.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["state_dict"].items()}); mm.eval()
    a, g = {}, {}
    L = mm.backbone.stages[-1]
    L.register_forward_hook(lambda mod, i, o, a=a: a.__setitem__("v", o))
    L.register_full_backward_hook(lambda mod, gi, go, g=g: g.__setitem__("v", go[0]))
    MODELS.append((mm, a, g))

def gradcam(pil):
    x = ctf(pil).unsqueeze(0).to(dev); cams, scores = [], []
    for mm, a, g in MODELS:
        mm.zero_grad()
        logit = mm.head(mm.backbone(x)); scores.append(float(torch.sigmoid(logit)))
        logit.backward()
        A, G = a["v"], g["v"]
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams.append(cam.detach().cpu().numpy())
    return np.mean(cams, axis=0), float(np.mean(scores))

def overlay(pil, cam):
    face = np.array(pil.resize((224, 224)))[:, :, ::-1]        # RGB→BGR
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return (0.55 * face + 0.45 * heat).astype(np.uint8)

df = pd.read_csv("/home/tako/seelhwan/ffpp_pair_branch_context_experiment/cache/face_x_context.csv", dtype={"video_id": str, "frame": str})
rows = []
for meth in ["original", "FaceSwap", "Deepfakes", "Face2Face", "NeuralTextures"]:
    g = df[df.method == meth]; vid = g["video_id"].iloc[0]
    p = g[g.video_id == vid].sort_values("frame")["face_X_path"].iloc[0]
    if not os.path.exists(p): continue
    pil = Image.open(p).convert("RGB"); cam, sc = gradcam(pil)
    orig = np.array(pil.resize((224, 224)))[:, :, ::-1]
    ov = overlay(pil, cam)
    strip = np.concatenate([orig, ov], axis=1)
    cv2.putText(strip, f"{meth} p(fake)={sc:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    rows.append(strip)
    print(f"{meth:16s} p(fake)={sc:.3f}", flush=True)
grid = np.concatenate(rows, axis=0)
out = "/tmp/claude-1005/-home-tako-seelhwan/0249eb81-2723-48a9-9085-49144bba4df9/scratchpad/gradcam_grid.png"
os.makedirs(os.path.dirname(out), exist_ok=True); cv2.imwrite(out, grid)
print(f"[save] {out}  (좌=원본, 우=Grad-CAM 오버레이)")
