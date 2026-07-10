#!/usr/bin/env python3
"""딥페이크 판별기 — 공간(ConvNeXt 전체얼굴) + 시공간(3D-CNN 입영역) 이중브랜치.

누수0 in-dist AUC 0.940, FF++ 4기법 전부 ≥0.89 (NeuralTextures 벽 0.55→0.89),
학습 때 안 본 조작(SimSwap)까지 잡음. 자립 실행(다른 스크립트 의존 없음).

사용:
  python infer.py --video   <경로>          # raw 비디오 (MTCNN 얼굴검출)
  python infer.py --frames_dir <폴더>        # 정렬된 얼굴 크롭 이미지 폴더
의존: torch, torchvision, timm, pillow, numpy  (+ --video 시 opencv-python, facenet-pytorch)
"""
import argparse, glob, os
import numpy as np, torch, torch.nn as nn, timm
import torchvision.models.video as vmodels
from torchvision import transforms
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SPATIAL_DIR  = os.path.join(HERE, "models", "spatial")
TEMPORAL_DIR = os.path.join(HERE, "models", "temporal")
IMG_MEAN, IMG_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # ImageNet (공간)
K_MEAN, K_STD     = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)  # Kinetics (시간)
MOUTH = (0.50, 0.96, 0.16, 0.84)   # 얼굴 크롭에서 입+턱 영역 (상,하,좌,우 비율)
T = 16                             # 클립 프레임 수

# ---- 모델 정의 (학습 스크립트와 동일 구조) ----
class FaceClassifier(nn.Module):
    def __init__(self, backbone="convnext_tiny.fb_in22k_ft_in1k", dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        feat = int(self.backbone.num_features)
        self.head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(dropout), nn.Linear(feat, 1))
    def forward(self, x): return self.head(self.backbone(x))

def make_temporal():
    m = vmodels.mc3_18(weights=None); m.fc = nn.Linear(m.fc.in_features, 1); return m

def load_models(dev):
    conv = []
    for p in sorted(glob.glob(os.path.join(SPATIAL_DIR, "*.pt"))):
        ck = torch.load(p, map_location=dev)
        m = FaceClassifier(ck.get("backbone", "convnext_tiny.fb_in22k_ft_in1k")).to(dev)
        m.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["state_dict"].items()}); m.eval(); conv.append(m)
    temp = []
    for p in sorted(glob.glob(os.path.join(TEMPORAL_DIR, "*.pt"))):
        ck = torch.load(p, map_location=dev)
        m = make_temporal().to(dev); m.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["state_dict"].items()}); m.eval(); temp.append(m)
    if not conv or not temp:
        raise SystemExit(f"[에러] 모델을 못 찾음. spatial={len(conv)} temporal={len(temp)} (models/ 폴더 확인)")
    return conv, temp

# ---- 얼굴 입력 ----
def faces_from_dir(d):
    ps = sorted(glob.glob(os.path.join(d, "*.jpg"))) + sorted(glob.glob(os.path.join(d, "*.png")))
    return [Image.open(p).convert("RGB") for p in ps[:T]]

def faces_from_video(path):
    import cv2
    from facenet_pytorch import MTCNN
    mt = MTCNN(keep_all=False, device="cpu")
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // 32 if total else 1); faces = []; fi = 0
    while len(faces) < T:
        ret, fr = cap.read()
        if not ret: break
        if fi % step == 0:
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            boxes, _ = mt.detect(im)
            if boxes is not None and len(boxes):
                x1, y1, x2, y2 = boxes[0]; pad = 0.1 * (y2 - y1)
                faces.append(im.crop((max(0, x1), max(0, y1 - pad), min(im.width, x2), min(im.height, y2 + pad))).resize((224, 224)))
        fi += 1
    cap.release(); return faces

def mouth(im):
    w, h = im.size; a, b, c, d = MOUTH
    return im.crop((int(c * w), int(a * h), int(d * w), int(b * h)))

# ---- 예측 ----
def predict(faces, conv, temp, dev):
    ctf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)])
    ttf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize(K_MEAN, K_STD)])
    xs = torch.stack([ctf(f) for f in faces]).to(dev)
    with torch.no_grad(): sp = float(np.mean([torch.sigmoid(m(xs)).mean().item() for m in conv]))
    sel = faces[:T] if len(faces) >= T else faces + [faces[-1]] * (T - len(faces))
    clip = torch.stack([ttf(mouth(f)) for f in sel], 1).unsqueeze(0).to(dev)
    with torch.no_grad(): tp = float(np.mean([torch.sigmoid(m(clip)).item() for m in temp]))
    return sp, tp, (sp + tp) / 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir"); ap.add_argument("--video")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=0.85, help="단일 브랜치 고신뢰 임계값(미지 조작 robust)")
    args = ap.parse_args()
    if not (args.frames_dir or args.video): raise SystemExit("--video 또는 --frames_dir 필요")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = args.frames_dir or args.video
    faces = faces_from_dir(args.frames_dir) if args.frames_dir else faces_from_video(args.video)
    if len(faces) < 4: raise SystemExit(f"[에러] 얼굴 프레임 부족({len(faces)})")
    conv, temp = load_models(dev)
    sp, tp, fused = predict(faces, conv, temp, dev)
    conf = max(sp, tp)
    is_fake = (fused >= args.thr) or (conf >= args.hi)
    reason = "mean" if fused >= args.thr else ("공간" if sp >= args.hi else "시공간") if is_fake else "-"
    print("\n╔══════════ 딥페이크 판별 결과 ══════════╗")
    print(f"  입력: {os.path.basename(src.rstrip('/'))}  (얼굴 {len(faces)}프레임 · device={dev.type})")
    print(f"  ├ 공간 브랜치 (ConvNeXt 전체얼굴):  {sp:.3f}")
    print(f"  ├ 시공간 브랜치 (3D-CNN 입영역):    {tp:.3f}")
    print(f"  └ 종합 fake 확률 (mean fusion):     {fused:.3f}")
    print(f"  ▶ 판정: {'🔴 FAKE' if is_fake else '🟢 REAL'}  (근거: {reason} | mean≥{args.thr} or 브랜치≥{args.hi})")
    print("╚═══════════════════════════════════════╝")

if __name__ == "__main__":
    main()
