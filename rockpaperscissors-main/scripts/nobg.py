"""Replace the background with flat grey using rembg, so the classifier sees only the hand.

  python scripts/nobg.py        # data/ -> data_nobg/ (same layout); run before train.py --nobg
"""
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent.parent
FILL = 128  # grey level painted over the background; train.py and predict.py must agree on it
_session = new_session("u2net")  # ~33 ms/crop on GPU; "u2netp" is ~20 ms but weaker on clutter


def strip_bg(rgb):
    """HxWx3 uint8 RGB -> same image with everything rembg calls background set to FILL."""
    mask = np.asarray(remove(rgb, session=_session, only_mask=True))
    return np.where(mask[..., None] > 127, rgb, FILL).astype(np.uint8)


if __name__ == "__main__":
    files = sorted((ROOT / "data").glob("*/*.png"))
    empty = 0
    for i, f in enumerate(files, 1):
        out = ROOT / "data_nobg" / f.relative_to(ROOT / "data")
        out.parent.mkdir(parents=True, exist_ok=True)
        img = strip_bg(np.asarray(Image.open(f).convert("RGB")))
        empty += (img == FILL).all()
        Image.fromarray(img).save(out)
        if i % 200 == 0 or i == len(files):
            print(f"nobg {i}/{len(files)}  empty masks so far: {empty}", flush=True)
