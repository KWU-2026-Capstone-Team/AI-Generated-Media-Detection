#!/usr/bin/env python3
"""시공간 Grad-CAM + 충실도(faithfulness) 검증.
정답: FF++ 풀프레임 |fake - original| (픽셀정렬됨) = 진짜 조작영역 마스크.
같은 얼굴박스로 얼굴크롭·마스크크롭 → 시공간 Grad-CAM(mc3_18 layer4, 6모델 평균)이
실제 조작영역을 국소화하나 pointing-game·AUC로 측정. 재연(F2F,NT) 대상(시간 신호 존재).
베이스라인: (a) 우연(=GT 영역 비율) (b) 랜덤가중치 mc3_18 (Adebayo sanity — 학습이 충실도의 원천인지)."""
import sys, os, glob, random, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer
from infer import make_temporal, mouth, MOUTH, T

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FR = "/home/tako/seelhwan/00_datasets/FF++/ffpp_data/frames_c23"
ttf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize(infer.K_MEAN, infer.K_STD)])
from facenet_pytorch import MTCNN
mt = MTCNN(keep_all=False, device=dev if dev.type == "cuda" else "cpu")

def facebox(pil):
    b, _ = mt.detect(pil)
    if b is None or not len(b): return None
    x1, y1, x2, y2 = b[0]; pad = 0.1 * (y2 - y1)
    return (max(0, x1), max(0, y1 - pad), min(pil.width, x2), min(pil.height, y2 + pad))

def gradcam3d(model, act, grad, clip):
    model.zero_grad()
    logit = model(clip); logit.backward()
    A, G = act["v"], grad["v"]                                # (1,C,t,h,w)
    w = G.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((w * A).sum(1, keepdim=True))                # (1,1,t,h,w)
    cam = F.interpolate(cam, size=(T, 112, 112), mode="trilinear", align_corners=False)[0, 0]  # (T,112,112)
    return cam.detach().cpu().numpy(), float(torch.sigmoid(logit))

def hook_model(m):
    act, grad = {}, {}
    L = m.layer4
    L.register_forward_hook(lambda mod, i, o: act.__setitem__("v", o))
    L.register_full_backward_hook(lambda mod, gi, go: grad.__setitem__("v", go[0]))
    return act, grad

# 배포 6모델 + 랜덤가중치 1개(sanity)
tmodels = []
for p in sorted(glob.glob("models/temporal/*.pt")):
    ck = torch.load(p, map_location=dev); m = make_temporal().to(dev)
    m.load_state_dict({k: (v.float() if torch.is_floating_point(v) else v) for k, v in ck["state_dict"].items()}); m.eval()
    tmodels.append((m, *hook_model(m)))
rnd = make_temporal().to(dev); rnd.eval(); rnd_hk = hook_model(rnd)   # 학습 안 된 랜덤 모델

def mouth_mask(mask_face_112):
    a, b, c, d = MOUTH; H = W = 112
    sub = mask_face_112[int(a * H):int(b * H), int(c * W):int(d * W)]
    return np.asarray(Image.fromarray(sub).resize((112, 112)))

def eval_video(fake_dir, orig_dir):
    frames = sorted(glob.glob(f"{fake_dir}/*.jpg"))
    faces, mouths, mmasks = [], [], []
    for fp in frames:
        fn = os.path.basename(fp); op = f"{orig_dir}/{fn}"
        if not os.path.exists(op): continue
        ff = Image.open(fp).convert("RGB"); of = Image.open(op).convert("RGB")
        if ff.size != of.size: continue
        box = facebox(ff)
        if box is None: continue
        face = ff.crop(box).resize((224, 224))
        # 마스크: 풀프레임 diff → 같은 박스 크롭 → 224 → 얼굴/입
        d = np.abs(np.asarray(ff.convert("L"), np.float32) - np.asarray(of.convert("L"), np.float32))
        mcrop = np.asarray(Image.fromarray(d.astype(np.uint8)).crop(box).resize((224, 224)), np.float32)
        mface112 = np.asarray(Image.fromarray(mcrop.astype(np.uint8)).resize((112, 112)), np.float32)
        faces.append(face); mouths.append(mouth(face)); mmasks.append(mouth_mask(mface112))
        if len(faces) >= T: break
    if len(faces) < 8: return None
    while len(mouths) < T: mouths.append(mouths[-1]); mmasks.append(mmasks[-1])
    clip = torch.stack([ttf(mm) for mm in mouths[:T]], 1).unsqueeze(0).to(dev)
    gt = np.stack(mmasks[:T])                                  # (T,112,112) 조작강도
    return clip, gt

def pointing_and_auc(cam, gt):
    """프레임별: cam peak이 GT 상위20% 영역에 드나(pointing) + cam이 GT를 랭킹하나(AUC)."""
    pg, aucs = [], []
    for t in range(cam.shape[0]):
        g = gt[t]
        if g.max() < 3: continue                               # 조작 거의 없는 프레임 skip
        thr = np.percentile(g, 80); gb = (g >= thr).astype(int)
        yx = np.unravel_index(np.argmax(cam[t]), cam[t].shape)
        pg.append(int(gb[yx] == 1))
        if gb.sum() and (gb == 0).sum(): aucs.append(roc_auc_score(gb.ravel(), cam[t].ravel()))
    return pg, aucs

def main():
    random.seed(0)
    vids = []
    for meth in ["Face2Face", "NeuralTextures"]:
        ds = sorted(glob.glob(f"{FR}/manipulated_sequences/{meth}/*"))
        random.shuffle(ds)
        for fd in ds[:12]:
            tgt = os.path.basename(fd).split("_")[0]; od = f"{FR}/original/{tgt}"
            if os.path.isdir(od): vids.append((meth, fd, od))
    print(f"[data] 재연 비디오 {len(vids)}개 (F2F+NT)", flush=True)
    res = {"trained": ([], []), "random": ([], []), "chance": []}
    for i, (meth, fd, od) in enumerate(vids):
        out = eval_video(fd, od)
        if out is None: continue
        clip, gt = out
        # 학습 6모델 CAM 평균
        cams = []
        for m, a, g in tmodels: c, _ = gradcam3d(m, a, g, clip); cams.append(c)
        cam_tr = np.mean(cams, 0)
        cam_rd, _ = gradcam3d(rnd, rnd_hk[0], rnd_hk[1], clip)
        pg, au = pointing_and_auc(cam_tr, gt); res["trained"][0].extend(pg); res["trained"][1].extend(au)
        pgr, aur = pointing_and_auc(cam_rd, gt); res["random"][0].extend(pgr); res["random"][1].extend(aur)
        # chance = GT 영역 비율(상위20%) = 0.20 이론값, 실측
        for t in range(gt.shape[0]):
            if gt[t].max() < 3: continue
            thr = np.percentile(gt[t], 80); res["chance"].append((gt[t] >= thr).mean())
        if (i + 1) % 6 == 0: print(f"  {i+1}/{len(vids)}", flush=True)
    def rep(name, pg, au):
        print(f"  {name:16s} pointing-game={np.mean(pg):.3f} (n={len(pg)})  CAM-vs-mask AUC={np.mean(au):.3f}")
    print(f"\n===== 시공간 Grad-CAM 충실도 (재연, 정답=풀프레임 조작마스크) =====")
    rep("학습 6모델", *res["trained"])
    rep("랜덤가중치(sanity)", *res["random"])
    print(f"  {'우연(chance)':16s} pointing-game={np.mean(res['chance']):.3f}  (GT 상위20% 영역비율)  AUC=0.500")
    print("\n판정: 학습모델 pointing-game≫우연 & AUC≫0.5 & 랜덤가중치≈우연 이면")
    print("      → Grad-CAM이 실제 조작영역을 충실히 국소화(학습에서 비롯) = 웹에 '측정된 근거'로 표기 가능")

if __name__ == "__main__":
    main()
