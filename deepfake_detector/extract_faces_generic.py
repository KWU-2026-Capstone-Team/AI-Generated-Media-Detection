#!/usr/bin/env python3
"""범용 얼굴 추출기 — 원본 비디오 트리 → 정렬 얼굴 크롭 (celebdf_faces 호환 레이아웃).
KoDF(학습 다양성) / FakeAVCeleb(held-out 평가) 둘 다에 사용. infer.faces_from_video 와 동일 검출 규약.

출력: {out_root}/{class}/{video_id}/frame{i:02d}.jpg    (class ∈ {real,fake})
학습/평가 로더는 이 레이아웃을 그대로 읽음 (eval_celebdf.py 와 동일).

라벨/필터 규칙(--fake_kw / --real_kw)은 경로 부분문자열로 지정 — 데이터셋 폴더구조 보고 조정.
  예) KoDF 재연만:  --fake_kw fom,atfhp,wav2lip,audio  --real_kw real,original
  예) FakeAVCeleb:  --fake_kw wav2lip,faceswap,fsgan   --real_kw real  (Wav2Lip만 원하면 --fake_kw wav2lip)
대용량 대비: --max_per_class 로 샘플, --delete_after 로 추출 후 원본 삭제(스트림 추출)."""
import argparse, glob, os, sys, random, re

def find_videos(root):
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts): out.append(os.path.join(dp, fn))
    return sorted(out)

def classify(path, fake_kw, real_kw):
    p = path.lower()
    if any(k in p for k in fake_kw): return "fake"
    if any(k in p for k in real_kw): return "real"
    return None  # 규칙 밖 → 건너뜀

def vid_id(path, root):
    rel = os.path.relpath(path, root)
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(rel)[0]).strip("_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True); ap.add_argument("--out_root", required=True)
    ap.add_argument("--fake_kw", default="", help="쉼표구분 부분문자열 → fake")
    ap.add_argument("--real_kw", default="real,original,orig", help="쉼표구분 → real")
    ap.add_argument("--T", type=int, default=16); ap.add_argument("--sample_stride_of", type=int, default=32,
                    help="비디오 전체를 대략 이 개수로 균등샘플 후 얼굴 검출")
    ap.add_argument("--max_per_class", type=int, default=0, help="0=전체")
    ap.add_argument("--size", type=int, default=224); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delete_after", action="store_true", help="추출 성공 시 원본 비디오 삭제(디스크 절약)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    fake_kw = [k for k in args.fake_kw.lower().split(",") if k]
    real_kw = [k for k in args.real_kw.lower().split(",") if k]

    import cv2
    from PIL import Image
    from facenet_pytorch import MTCNN
    mt = MTCNN(keep_all=False, device=args.device)

    vids = find_videos(args.video_root)
    buckets = {"real": [], "fake": []}
    for v in vids:
        c = classify(v, fake_kw, real_kw)
        if c: buckets[c].append(v)
    rng = random.Random(args.seed)
    for c in buckets:
        rng.shuffle(buckets[c])
        if args.max_per_class: buckets[c] = buckets[c][:args.max_per_class]
    print(f"[scan] {args.video_root}: real {len(buckets['real'])} + fake {len(buckets['fake'])} 비디오 (필터적용)", flush=True)

    done = skip = 0
    for c in ("real", "fake"):
        for vi, vpath in enumerate(buckets[c]):
            vid = vid_id(vpath, args.video_root)
            odir = os.path.join(args.out_root, c, vid)
            if os.path.isdir(odir) and len(glob.glob(f"{odir}/*.jpg")) >= 4:
                skip += 1; continue
            cap = cv2.VideoCapture(vpath); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(1, (total // args.sample_stride_of) if total else 1)
            os.makedirs(odir, exist_ok=True); saved = 0; fi = 0
            while saved < args.T:
                ret, fr = cap.read()
                if not ret: break
                if fi % step == 0:
                    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    boxes, _ = mt.detect(im)
                    if boxes is not None and len(boxes):
                        x1, y1, x2, y2 = boxes[0]; pad = 0.1 * (y2 - y1)
                        crop = im.crop((max(0, x1), max(0, y1 - pad), min(im.width, x2), min(im.height, y2 + pad))).resize((args.size, args.size))
                        crop.save(f"{odir}/frame{saved:02d}.jpg", quality=95); saved += 1
                fi += 1
            cap.release()
            if saved >= 4:
                done += 1
                if args.delete_after:
                    try: os.remove(vpath)
                    except OSError: pass
            if (vi + 1) % 50 == 0: print(f"  [{c}] {vi+1}/{len(buckets[c])}  (추출{done} 스킵{skip})", flush=True)
    print(f"[done] 추출 {done}, 스킵 {skip}  → {args.out_root}", flush=True)

if __name__ == "__main__":
    main()
