# rockpaperscissors

Small image classifier for rock / paper / scissors hand gestures, compared across 8 pretrained
models and selected by macro-F1. The webcam demo finds hands with MediaPipe and draws a labelled bounding box around each.

## Setup
```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchvision timm scikit-learn pillow pandas matplotlib imagehash opencv-python mediapipe "rembg[gpu]"
```
Download the [Kaggle dataset](https://www.kaggle.com/datasets/alexandredj/rock-paper-scissors-dataset)
and unzip it so you have `data/paper`, `data/rock`, `data/scissors`.

## Pipeline
| Step | Command |
|---|---|
| Audit data (integrity, duplicates, 27-image sample grid) | `python scripts/audit_data.py data reports` |
| Train/val/test split (70/15/15) | `python scripts/make_split.py data splits/split.csv` |
| Train all models | `./scripts/run_all.sh` |
| Leaderboard | `python scripts/summarize.py` |
| Webcam / image demo | `python scripts/predict.py [--model NAME] [images...]` |
| Strip backgrounds with rembg (`data/` → `data_nobg/`) | `python scripts/nobg.py` |
| Train on background-stripped images (saves `NAME_nobg.pt`) | `python scripts/train.py NAME --nobg` |
| Webcam demo with background removal | `python scripts/predict.py --nobg [--model NAME]` |

## Data notes
- 2,717 images, 300×300 RGB, balanced (907 paper / 907 rock / 903 scissors). No corrupt files or exact duplicates.
- Many near-duplicate video frames → the split keeps each near-duplicate group within one split (no leakage).
- Two backgrounds (smooth pale wall: exactly 400/class; textured grey wall) → split stratified by class × wall.
- All shots are a single centred hand on a plain wall, so expect a drop on real webcam frames.
