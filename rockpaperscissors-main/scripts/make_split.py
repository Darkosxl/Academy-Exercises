"""Group-aware stratified 70/15/15 split. Near-duplicate frames (pHash distance <= 4, same class)
share a group so they never straddle train/val/test. Stratified by class x wall (smooth vs textured),
so each split has the same background mix per class."""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "splits/split.csv")
SEED = 42
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}

files = sorted(DATA.glob("*/*.png"))
hashes = [imagehash.phash(Image.open(f)) for f in files]


def wall(f):
    """Two shoot backgrounds: a smooth pale wall and a textured grey wall. Median Laplacian
    magnitude in the top corners is bimodal (~2 vs >8); 5 sits in the empty gap."""
    g = np.asarray(Image.open(f).convert("L")).astype(np.float32)
    hp = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    tex = np.median(np.concatenate([hp[:50, :50].ravel(), hp[:50, -50:].ravel()]))
    return "smooth" if tex < 5 else "textured"


walls = [wall(f) for f in files]

parent = list(range(len(files)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for i in range(len(files)):
    for j in range(i + 1, len(files)):
        if files[i].parent == files[j].parent and hashes[i] - hashes[j] <= 4:
            parent[find(i)] = find(j)

rows = []
rng = random.Random(SEED)
groups = defaultdict(list)
for i, f in enumerate(files):
    groups[find(i)].append(i)
strata = defaultdict(list)  # (class, majority wall of group) -> group ids
for gid, members in groups.items():
    ws = [walls[i] for i in members]
    strata[(files[gid].parent.name, max(set(ws), key=ws.count))].append(gid)
for key in sorted(strata):
    gids = strata[key]
    rng.shuffle(gids)
    total = sum(len(groups[g]) for g in gids)
    counts = {k: 0 for k in FRACS}
    for gid in gids:
        # assign group to the split furthest below its target share
        split = min(FRACS, key=lambda k: counts[k] / total - FRACS[k])
        counts[split] += len(groups[gid])
        rows += [(str(files[i]), key[0], walls[i], split, gid) for i in groups[gid]]
    print(key, counts)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "label", "wall", "split", "group"])
    w.writerows(rows)
print("wrote", OUT, len(rows))
