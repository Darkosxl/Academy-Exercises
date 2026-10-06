"""Audit the rock/paper/scissors dataset: integrity, sizes, duplicates, and a random sample grid."""
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "reports")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

rows = []
corrupt = []
for cls_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
    for f in sorted(cls_dir.iterdir()):
        try:
            raw = f.read_bytes()
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                im.load()
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
                rows.append(dict(
                    path=f, cls=cls_dir.name, size=im.size, mode=im.mode, bytes=len(raw),
                    md5=hashlib.md5(raw).hexdigest(), phash=imagehash.phash(im.convert("RGB")),
                    mean=rgb.mean(), std=rgb.std(),
                ))
        except Exception as e:  # noqa: BLE001
            corrupt.append((f, repr(e)))

print(f"Total readable images: {len(rows)}  corrupt: {len(corrupt)}")
for c in corrupt:
    print("  CORRUPT", c)
print("Per class:", dict(Counter(r["cls"] for r in rows)))
print("Sizes:", Counter(r["size"] for r in rows).most_common(10))
print("Modes:", dict(Counter(r["mode"] for r in rows)))
b = np.array([r["bytes"] for r in rows]) / 1024
print(f"File size KB: min {b.min():.0f} median {np.median(b):.0f} max {b.max():.0f}")

for cls in sorted({r["cls"] for r in rows}):
    m = np.array([r["mean"] for r in rows if r["cls"] == cls])
    s = np.array([r["std"] for r in rows if r["cls"] == cls])
    print(f"  {cls:9s} brightness mean {m.mean():.1f} (+/-{m.std():.1f})  contrast {s.mean():.1f}")

# Flag near-blank / extreme images
odd = [r for r in rows if r["std"] < 10 or r["mean"] < 20 or r["mean"] > 240]
print(f"Near-blank / extreme-exposure images: {len(odd)}", [str(r['path']) for r in odd[:10]])

# Exact duplicates
by_md5 = defaultdict(list)
for r in rows:
    by_md5[r["md5"]].append(r)
exact = [g for g in by_md5.values() if len(g) > 1]
print(f"Exact-duplicate groups: {len(exact)}")
for g in exact[:10]:
    print("  ", [f"{r['cls']}/{r['path'].name}" for r in g])

# Near duplicates (perceptual hash distance <= 4); flag cross-class ones as label conflicts
near, cross = 0, []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        if rows[i]["phash"] - rows[j]["phash"] <= 4:
            near += 1
            if rows[i]["cls"] != rows[j]["cls"]:
                cross.append((rows[i]["path"], rows[j]["path"]))
print(f"Near-duplicate pairs (pHash<=4): {near}; cross-class: {len(cross)}")
for a, c in cross[:10]:
    print("  CROSS", a, c)

# 27 random images: 9 per class, labelled
random.seed(SEED)
sample = []
for cls in sorted({r["cls"] for r in rows}):
    sample += random.sample([r for r in rows if r["cls"] == cls], 9)
random.shuffle(sample)
fig, axes = plt.subplots(3, 9, figsize=(27, 10))
for ax, r in zip(axes.flat, sample):
    ax.imshow(Image.open(r["path"]).convert("RGB"))
    ax.set_title(f"{r['cls']}\n{r['path'].name}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "sample_27.png", dpi=60)
print("Saved", OUT / "sample_27.png")
print("Sample:", [f"{r['cls']}/{r['path'].name}" for r in sample])
