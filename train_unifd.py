from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
UFD_ROOT = THIS_DIR / "uniFD" / "UniversalFakeDetect"
sys.path.insert(0, str(UFD_ROOT))

from models import get_model  # noqa: E402


class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception:
            print(f"Skipping corrupted image: {path}")
            return None


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)


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


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CLIP:RN50")
    parser.add_argument("--train_dir", default="data/train")
    parser.add_argument("--val_dir", default="data/val")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out", default="weights/unifd_best.pth")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    device = get_device(force_cpu=args.force_cpu)
    print("Device:", device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    train_tf = T.Compose([
        T.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(clip_mean, clip_std),
    ])

    val_tf = T.Compose([
        T.Resize(args.img_size + 32),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(clip_mean, clip_std),
    ])

    train_ds = SafeImageFolder(args.train_dir, transform=train_tf)
    val_ds = SafeImageFolder(args.val_dir, transform=val_tf)
    print("class_to_idx:", train_ds.class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=safe_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=safe_collate,
        pin_memory=(device.type == "cuda"),
    )

    model = get_model(args.model).to(device).float()

    if hasattr(model, "model"):
        for p in model.model.parameters():
            p.requires_grad = False

    if not hasattr(model, "fc"):
        raise AttributeError("Expected model to have attribute 'fc'.")
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for imgs, labels in train_loader:
            if imgs.numel() == 0:
                continue

            imgs = imgs.to(device).float().contiguous()
            labels = labels.to(device).long().contiguous()
            labels_f = labels.float().unsqueeze(1).contiguous()

            optimizer.zero_grad(set_to_none=True)

            logits = unwrap_logits(model(imgs)).float()
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            logits = logits.clamp(-20, 20)

            loss = F.binary_cross_entropy_with_logits(logits, labels_f)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss={loss.item()} logits_minmax=({logits.min().item()},{logits.max().item()})"
                )

            loss.backward()
            optimizer.step()

            tr_loss += float(loss.detach().cpu().item()) * imgs.size(0)

            preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).long()
            tr_correct += int((preds == labels).sum().item())
            tr_total += int(imgs.size(0))

        tr_loss /= max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)

        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                if imgs.numel() == 0:
                    continue

                imgs = imgs.to(device).float().contiguous()
                labels = labels.to(device).long().contiguous()
                labels_f = labels.float().unsqueeze(1).contiguous()

                logits = unwrap_logits(model(imgs)).float()
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                logits = logits.clamp(-20, 20)

                loss = F.binary_cross_entropy_with_logits(logits, labels_f)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite val loss={loss.item()} logits_minmax=({logits.min().item()},{logits.max().item()})"
                    )

                va_loss += float(loss.detach().cpu().item()) * imgs.size(0)

                preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).long()
                va_correct += int((preds == labels).sum().item())
                va_total += int(imgs.size(0))

        va_loss /= max(1, va_total)
        va_acc = va_correct / max(1, va_total)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "img_size": args.img_size,
                },
                out_path,
            )
            print("Saved best ->", out_path)

    print("Done. Best Val Acc:", best_val_acc)


if __name__ == "__main__":
    main()