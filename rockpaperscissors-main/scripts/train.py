"""Fine-tune one pretrained timm model on the RPS split; select by val macro-F1, report test metrics.

Usage: python train.py <timm_model_name> [--epochs N] [--lr LR]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["paper", "rock", "scissors"]

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--patience", type=int, default=5)
p.add_argument("--img", type=int, default=224)
p.add_argument("--nobg", action="store_true", help="train on data_nobg/ (see scripts/nobg.py)")
args = p.parse_args()
name = args.model + ("_nobg" if args.nobg else "")

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ---- data: decode everything once into uint8 tensors (~730 MB) ----
rows = list(csv.DictReader((ROOT / "splits/split.csv").open()))
splits = {}
for s in ("train", "val", "test"):
    sel = [r for r in rows if r["split"] == s]
    imgs = torch.stack([T.functional.pil_to_tensor(Image.open(ROOT / (r["path"].replace("data/", "data_nobg/", 1) if args.nobg else r["path"])).convert("RGB")) for r in sel])
    labels = torch.tensor([CLASSES.index(r["label"]) for r in sel])
    splits[s] = (imgs, labels)

model = timm.create_model(args.model, pretrained=True, num_classes=len(CLASSES))
cfg = timm.data.resolve_data_config({}, model=model)
norm = T.Normalize(cfg["mean"], cfg["std"])

# Strong augmentation to bridge the gap to messy webcam frames
train_tf = T.Compose([
    T.RandomResizedCrop(args.img, scale=(0.5, 1.0), ratio=(0.8, 1.25), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomRotation(25, fill=128 if args.nobg else 0),  # 128 = nobg.FILL
    T.ColorJitter(0.4, 0.4, 0.4, 0.08),
    T.RandomGrayscale(0.05),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
    T.ToDtype(torch.float32, scale=True),
    norm,
])
eval_tf = T.Compose([T.Resize(args.img, antialias=True), T.ToDtype(torch.float32, scale=True), norm])


class TensorDS(Dataset):
    def __init__(self, imgs, labels, tf):
        self.imgs, self.labels, self.tf = imgs, labels, tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), self.labels[i]


loaders = {
    s: DataLoader(TensorDS(*splits[s], train_tf if s == "train" else eval_tf),
                  batch_size=args.bs, shuffle=(s == "train"), num_workers=0)
    for s in splits
}

model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
steps = args.epochs * len(loaders["train"])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)


@torch.no_grad()
def predict(loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


out_dir = ROOT / "models"
out_dir.mkdir(exist_ok=True)
ckpt = out_dir / f"{name}.pt"
best_f1, best_ep, bad, history = -1.0, 0, 0, []
t0 = time.time()
for ep in range(1, args.epochs + 1):
    model.train()
    tot = 0.0
    for x, y in loaders["train"]:
        x, y = x.to(device), y.to(device)
        loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item() * len(y)
    yv, pv = predict(loaders["val"])
    vf1 = f1_score(yv, pv, average="macro")
    history.append(dict(epoch=ep, loss=tot / len(loaders["train"].dataset), val_f1=vf1))
    print(f"[{name}] ep {ep:2d} loss {history[-1]['loss']:.4f} val_f1 {vf1:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if vf1 > best_f1:
        best_f1, best_ep, bad = vf1, ep, 0
        torch.save(model.state_dict(), ckpt)
    else:
        bad += 1
        if bad >= args.patience:
            print("early stop", flush=True)
            break
train_time = time.time() - t0

model.load_state_dict(torch.load(ckpt, map_location=device))
yt, pt = predict(loaders["test"])


def latency(dev):
    m = model.to(dev).eval()
    x = torch.randn(1, 3, args.img, args.img, device=dev)
    with torch.no_grad():
        for _ in range(10):
            m(x)
        if dev.type == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            m(x)
        if dev.type == "mps":
            torch.mps.synchronize()
    return (time.perf_counter() - t) / 50 * 1000


res = dict(
    model=name,
    params_m=sum(p.numel() for p in model.parameters()) / 1e6,
    best_epoch=best_ep,
    val_f1=best_f1,
    test_f1=f1_score(yt, pt, average="macro"),
    test_acc=accuracy_score(yt, pt),
    per_class_f1=dict(zip(CLASSES, f1_score(yt, pt, average=None).tolist())),
    confusion=confusion_matrix(yt, pt).tolist(),
    train_time_s=train_time,
    latency_ms_mps=latency(torch.device("mps")) if device.type == "mps" else None,
    latency_ms_cpu=latency(torch.device("cpu")),
    history=history,
)
print(classification_report(yt, pt, target_names=CLASSES, digits=4))
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))
