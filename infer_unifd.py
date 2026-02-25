from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torchvision.transforms as T
from PIL import Image

# =========================
# 경로 설정
# =========================
THIS_DIR = Path(__file__).resolve().parent
UFD_ROOT = THIS_DIR / "uniFD" / "UniversalFakeDetect"
sys.path.insert(0, str(UFD_ROOT))

from models import get_model  # noqa: E402


def unwrap_logits(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for v in out.values():
            if torch.is_tensor(v):
                return v
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return unwrap_logits(out[0])
    raise ValueError(f"Unknown output type: {type(out)}")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="weights/unifd_best.pth")
    ap.add_argument("--test_dir", type=str, default="test/test_pictures")
    ap.add_argument("--out_txt", type=str, default="test/result.txt")
    args = ap.parse_args()

    device = get_device()
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["model_name"]
    class_to_idx = ckpt.get("class_to_idx", {"fake": 0, "real": 1})
    img_size = ckpt.get("img_size", 224)

    model = get_model(model_name)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device).float()
    model.eval()

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose(
        [
            T.Resize(img_size + 32),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(clip_mean, clip_std),
        ]
    )

    test_dir = Path(args.test_dir)
    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted([p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No image files found in: {test_dir}")

    lines: list[str] = []

    with torch.no_grad():
        for i, p in enumerate(img_paths, start=1):
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                lines.append(f"{i}. error ({p.name})")
                continue

            x = transform(img).unsqueeze(0).to(device)

            logits = unwrap_logits(model(x))
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            if logits.shape[1] != 1:
                raise ValueError(f"Expected logits shape [B,1], got {tuple(logits.shape)}")

            p_real = torch.sigmoid(logits)[0, 0].item()
            p_fake = 1.0 - p_real

            # class_to_idx에 맞춰 pred 결정
            probs = {}
            for cls, idx in class_to_idx.items():
                if idx == 1:
                    probs[cls] = p_real
                elif idx == 0:
                    probs[cls] = p_fake

            pred_cls = max(probs, key=lambda k: probs[k])
            lines.append(f"{i}. {pred_cls}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {out_path} ({len(lines)} files)")


if __name__ == "__main__":
    main()