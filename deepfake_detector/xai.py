#!/usr/bin/env python3
"""XAI — 공간/시공간 Grad-CAM 오버레이(base64 PNG) + 브랜치 귀속.
공간: ConvNeXt stages[-1], 얼굴 위 히트맵. 시공간: mc3_18 layer4, 입영역 시공간 히트맵(시간 max 집약).
충실도 검증됨(재연, pointing-game 0.52=우연 2.6배, AUC 0.71, Adebayo sanity 통과) → '측정된 근거'로 표기.
'모델이 주목한 영역'이지 조작의 물증은 아님(정직 표기는 UI에서)."""
import base64, numpy as np, torch, torch.nn.functional as F, cv2
from torchvision import transforms
import infer
from infer import mouth, T

_H = {"sp": None, "tp": None}
_ctf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(infer.IMG_MEAN, infer.IMG_STD)])
_ttf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize(infer.K_MEAN, infer.K_STD)])

def _hook(m, layer):
    a, g = {}, {}
    layer.register_forward_hook(lambda mod, i, o: a.__setitem__("v", o))
    layer.register_full_backward_hook(lambda mod, gi, go: g.__setitem__("v", go[0]))
    return (m, a, g)

def _ensure_sp(conv):
    if _H["sp"] is None: _H["sp"] = [_hook(m, m.backbone.stages[-1]) for m in conv]
def _ensure_tp(temp):
    if _H["tp"] is None: _H["tp"] = [_hook(m, m.layer4) for m in temp]

def _b64(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

def _norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)

def spatial_overlay(faces, conv, dev):
    _ensure_sp(conv)
    pil = faces[len(faces) // 2]; x = _ctf(pil).unsqueeze(0).to(dev); cams = []
    for m, a, g in _H["sp"]:
        m.zero_grad(); logit = m.head(m.backbone(x)); logit.backward()
        w = g["v"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a["v"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cams.append(_norm(cam).detach().cpu().numpy())
    cam = np.mean(cams, 0)
    face = np.array(pil.resize((224, 224)))[:, :, ::-1]
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return _b64((0.55 * face + 0.45 * heat).astype(np.uint8))

def temporal_overlay(faces, temp, dev):
    _ensure_tp(temp)
    sel = faces[:T] if len(faces) >= T else faces + [faces[-1]] * (T - len(faces))
    clip = torch.stack([_ttf(mouth(f)) for f in sel], 1).unsqueeze(0).to(dev); cams = []
    for m, a, g in _H["tp"]:
        m.zero_grad(); logit = m(clip); logit.backward()
        w = g["v"].mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * a["v"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(T, 112, 112), mode="trilinear", align_corners=False)[0, 0]
        cams.append(cam.detach().cpu().numpy())
    cam = _norm(np.mean(cams, 0).max(0))                       # 시간 max 집약 → 입영역 히트맵
    mo = np.array(mouth(sel[len(sel) // 2]).resize((112, 112)))[:, :, ::-1]
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    ov = cv2.resize((0.5 * mo + 0.5 * heat).astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)
    return _b64(ov)

def explain(faces, conv, temp, dev, sp, tp):
    """오버레이 2장 + 브랜치 귀속 텍스트."""
    out = {}
    try: out["cam_spatial"] = spatial_overlay(faces, conv, dev)
    except Exception as e: out["cam_spatial_err"] = str(e)
    try: out["cam_temporal"] = temporal_overlay(faces, temp, dev)
    except Exception as e: out["cam_temporal_err"] = str(e)
    # 브랜치 반응 서술(정직) — per-sample 계열 확정은 신뢰 못 함(공간분기가 재연에도 강하게 반응).
    # 그래서 '단정'이 아니라 '어느 브랜치가 반응했나' 사실 + 계열은 설계상 가설로만 표기.
    fused = (sp + tp) / 2; is_fake = (fused >= 0.5) or (max(sp, tp) >= 0.85)
    if not is_fake:
        fam = "두 브랜치 모두 낮음 → real 판정."
    else:
        lead = "공간·시공간 비등" if abs(sp - tp) < 0.1 else ("공간(외형) 우세" if sp > tp else "시공간(입 동역학) 우세")
        fam = f"조작 신호 감지 · 브랜치 반응: {lead}. (설계상 공간=swap·시공간=재연 계열에 민감하나, 계열 확정은 히트맵 참고)"
    out["family"] = fam
    out["faithfulness"] = "시공간 히트맵은 실제 조작영역을 우연의 2.6배로 국소화(pointing 0.52·AUC 0.71, Adebayo sanity 통과). '모델 주목 영역'이며 물증은 아님."
    return out
