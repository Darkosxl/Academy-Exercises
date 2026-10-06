"""Try a trained classifier on the webcam (default) or on image files.

  python scripts/predict.py                         # webcam, mobilenetv3_small_100
  python scripts/predict.py --model resnet18        # another trained model
  python scripts/predict.py --nobg                  # background removed with rembg first
  python scripts/predict.py img1.jpg img2.png       # image files

The model was trained on square, hand-centred shots against a plain wall, so the webcam view
finds each hand with MediaPipe, crops a padded square around it and classifies that crop.
Press q to quit, s to save the current raw frame to logs/captures/.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import timm
import torch
from PIL import Image
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["paper", "rock", "scissors"]

p = argparse.ArgumentParser()
p.add_argument("images", nargs="*")
p.add_argument("--model", default="mobilenetv3_small_100")
p.add_argument("--camera", type=int, default=0)
p.add_argument("--nobg", action="store_true", help="strip background with rembg, use the *_nobg checkpoint")
args = p.parse_args()
name = args.model + ("_nobg" if args.nobg else "")
if args.nobg:
    from nobg import strip_bg

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = timm.create_model(args.model, pretrained=False, num_classes=len(CLASSES))
model.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu"))
model.to(device).eval()
cfg = timm.data.resolve_data_config({}, model=model)
tf = T.Compose([T.ToImage(), T.Resize(224, antialias=True), T.CenterCrop(224),
                T.ToDtype(torch.float32, scale=True), T.Normalize(cfg["mean"], cfg["std"])])


@torch.no_grad()
def classify(rgb):
    probs = model(tf(rgb).unsqueeze(0).to(device)).softmax(1)[0].cpu().numpy()
    return probs


if args.images:
    for path in args.images:
        img = np.asarray(Image.open(path).convert("RGB"))
        probs = classify(strip_bg(img) if args.nobg else img)
        ranked = ", ".join(f"{CLASSES[i]} {probs[i]:.2f}" for i in probs.argsort()[::-1])
        print(f"{path}: {CLASSES[probs.argmax()].upper()}  ({ranked})")
    raise SystemExit

# ---- webcam: MediaPipe finds the hands, the classifier labels each crop ----
PAD = 1.6  # box side = PAD x landmark extent; training shots show wrist + margin, tune if labels flicker
MIN_CONF = 0.7  # below this the box turns orange
task = ROOT / "models" / "hand_landmarker.task"
if not task.exists():
    from urllib.request import urlretrieve
    urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                "float16/1/hand_landmarker.task", task)
vision = mp.tasks.vision
hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),
    running_mode=vision.RunningMode.VIDEO, num_hands=2))


def hand_boxes(rgb, ts_ms):
    """Square pixel boxes (x0, y0, x1, y1), clipped to the frame, one per detected hand."""
    h, w = rgb.shape[:2]
    res = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
    boxes = []
    for lm in res.hand_landmarks:
        xs, ys = [q.x * w for q in lm], [q.y * h for q in lm]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = PAD * max(max(xs) - min(xs), max(ys) - min(ys)) / 2
        box = max(0, int(cx - half)), max(0, int(cy - half)), min(w, int(cx + half)), min(h, int(cy + half))
        if box[2] - box[0] > 20 and box[3] - box[1] > 20:  # hand mostly off-frame
            boxes.append(box)
    return boxes


cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. On macOS, allow camera access for your terminal app "
                     "in System Settings > Privacy & Security > Camera.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # camera falls back to its nearest mode if unsupported
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("rock-paper-scissors", cv2.WINDOW_NORMAL)  # resizable, image scales with the window
fps, last = 0.0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror, feels natural
    raw = frame.copy()  # undrawn, for the s-key capture
    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes = hand_boxes(rgb, int(time.perf_counter() * 1000))
    for x0, y0, x1, y1 in boxes:
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        if args.nobg:
            crop = strip_bg(crop)
            frame[y0:y1, x0:x1] = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)  # show what the model sees
        probs = classify(crop)
        k = int(probs.argmax())
        color = (0, 200, 0) if probs[k] > MIN_CONF else (0, 200, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, f"{CLASSES[k].upper()} {probs[k]:.2f}", (x0, max(30, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if not boxes:
        cv2.putText(frame, "no hand", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / (now - last)
    last = now
    cv2.putText(frame, f"{fps:.0f} fps  [{name}]  s=save  q=quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("rock-paper-scissors", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):  # save the raw frame so a misread can be replayed offline
        (ROOT / "logs" / "captures").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ROOT / "logs" / "captures" / f"{time.time():.2f}.png"), raw)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
