"""Live test: is the person talking Cem Berke, or someone else?

    python live_test.py              # webcam window, who is speaking at the bottom (q quits)
    python live_test.py --no-camera  # microphone only, prints one line per second
    python live_test.py talk.wav     # run on a recording instead
    python live_test.py --check      # score it against the validation conversations

The fine-tuned pyannote model finds who is speaking when (anonymous speakers).
Each speaker's voice is then compared to my voiceprint to decide "Cem Berke" or "someone else".
"""
import csv
import functools
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image, ImageDraw, ImageFont
from pyannote.audio import Inference, Model
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

NAME = "Cem Berke"  # shown when the voice matches my voiceprint
SR = 16000
WINDOW_S = 4        # how much audio the model looks at for each decision
STEP_S = 1          # how often it decides
MIN_SPEECH_S = 0.5  # a speaker needs this much speech in the window to be judged
MIC_GAIN = 1.0      # raise if the mic is quiet and speech is being missed
THRESHOLD = None    # voice similarity needed to count as me; None = use the calibrated one
DATA = Path("data")
VOICEPRINT = Path("models/voiceprint.npz")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation = Inference(Model.from_pretrained("models/segmentation_tr.ckpt"), window="whole", device=device)
embedder = PretrainedSpeakerEmbedding("pyannote/wespeaker-voxceleb-resnet34-LM", device=device)


def embed(audio, mask=None):
    """Voiceprint of a clip as a unit vector. `mask` says which frames belong to the speaker."""
    weights = None if mask is None else torch.from_numpy(mask)[None]
    e = embedder(torch.from_numpy(audio)[None, None], weights)[0]
    return e / np.linalg.norm(e)


def load_voiceprint():
    """My average voiceprint, plus the similarity threshold that separates me from others."""
    if not VOICEPRINT.exists():
        mine = {"train": [], "val": []}
        for r in csv.DictReader(open(DATA / "me_manifest.csv")):
            mine[r["split"]].append(embed(sf.read(DATA / "me" / r["file"], dtype="float32")[0]))
        voiceprint = np.mean(mine["train"], axis=0)
        voiceprint /= np.linalg.norm(voiceprint)

        # calibrate on clips the voiceprint was not built from
        me_scores = np.array([e @ voiceprint for e in mine["val"]])
        other_clips = sorted((DATA / "others").glob("*.wav"))[::20]
        other_scores = np.array([embed(sf.read(f, dtype="float32")[0]) @ voiceprint for f in other_clips])
        threshold = (me_scores.mean() + other_scores.mean()) / 2
        print(f"voiceprint built | similarity to me: {me_scores.mean():.2f}, to others: {other_scores.mean():.2f}, "
              f"threshold {threshold:.2f} | me kept {np.mean(me_scores > threshold):.0%}, "
              f"others rejected {np.mean(other_scores <= threshold):.0%}")
        np.savez(VOICEPRINT, voiceprint=voiceprint, threshold=threshold)
    saved = np.load(VOICEPRINT)
    return saved["voiceprint"], float(saved["threshold"] if THRESHOLD is None else THRESHOLD)


def who_is_talking(window, voiceprint, threshold):
    """Who spoke in the last STEP_S of the window: a set containing NAME and/or 'someone else'."""
    activity = segmentation({"waveform": torch.from_numpy(window)[None], "sample_rate": SR})
    activity = np.asarray(getattr(activity, "data", activity), dtype=np.float32)  # (frames, speakers), 0 or 1
    frames_per_s = len(activity) / (len(window) / SR)
    recent = activity[-int(STEP_S * frames_per_s):]

    labels = set()
    for speaker in range(activity.shape[1]):
        if recent[:, speaker].mean() < 0.3 or activity[:, speaker].sum() < MIN_SPEECH_S * frames_per_s:
            continue
        similarity = embed(window, activity[:, speaker]) @ voiceprint
        labels.add(NAME if similarity > threshold else "someone else")
    return labels


def microphone():
    """Yield STEP_S blocks of microphone audio, via ffmpeg (already installed, no extra library)."""
    ffmpeg = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "pulse", "-i", "default",
                               "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], stdout=subprocess.PIPE)
    while block := ffmpeg.stdout.read(SR * STEP_S * 4):
        yield np.frombuffer(block, dtype=np.float32) * MIC_GAIN


def recording(path):
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SR and audio.ndim == 1, "need 16 kHz mono (see Audio_Data_Cleaning.ipynb for the ffmpeg command)"
    for i in range(0, len(audio) - SR * STEP_S + 1, SR * STEP_S):
        yield audio[i:i + SR * STEP_S]


def listen(blocks, voiceprint, threshold):
    """Yield (seconds, labels) once per STEP_S, judging the most recent WINDOW_S of audio."""
    window = np.zeros(WINDOW_S * SR, dtype=np.float32)
    for n, block in enumerate(blocks, 1):
        window = np.concatenate([window[len(block):], block])
        yield n * STEP_S, who_is_talking(window, voiceprint, threshold)


def check(voiceprint, threshold):
    """Score every second of the validation conversations against their who-spoke-when labels."""
    hits = {NAME: [], "someone else": []}
    for rttm in sorted((DATA / "conversations" / "rttm").glob("val_*.rttm")):
        turns = [(float(p[3]), float(p[3]) + float(p[4]), NAME if p[7] == "me" else "someone else")
                 for p in map(str.split, rttm.read_text().splitlines())]
        wav = DATA / "conversations" / "wav" / f"{rttm.stem}.wav"
        for t, labels in listen(recording(wav), voiceprint, threshold):
            # only seconds where one known speaker talks the whole second, so the right answer is unambiguous
            truth = {who for start, end, who in turns if start < t and end > t - STEP_S}
            solo = [who for start, end, who in turns if start <= t - STEP_S and end >= t]
            if len(truth) == 1 and solo:
                hits[solo[0]].append(labels == truth)
    for who, h in hits.items():
        print(f"{who:>13}: right {np.mean(h):.0%} of {len(h)} seconds")
    assert np.mean(hits[NAME]) > 0.5 and np.mean(hits["someone else"]) > 0.5, "worse than a coin flip"


GREEN, ORANGE, GREY, RED = (60, 220, 130), (255, 170, 60), (150, 150, 150), (255, 80, 80)  # RGB


@functools.lru_cache
def font(style, size):
    """A system font through fontconfig; Pillow's built-in one if that fails."""
    try:
        found = subprocess.run(["fc-match", "-f", "%{file}", f"Noto Sans:{style}"], capture_output=True, text=True)
        return ImageFont.truetype(found.stdout, size)
    except OSError:
        return ImageFont.load_default(size)


def draw_status(frame, labels, error=None):
    """Frosted-glass pill at the bottom centre: a status dot, who it is, and what they are doing."""
    if error:
        dot, title, rest = RED, "Audio stopped", error[:40]
    elif labels is None:
        dot, title, rest = GREY, "", "Listening…"
    elif not labels:
        dot, title, rest = GREY, "", "Nobody speaking"
    else:
        names = sorted(labels, key=lambda label: label != NAME)  # my name first
        dot, title, rest = (GREEN if NAME in labels else ORANGE), " + ".join(names), "speaking"
        title = title[0].upper() + title[1:]

    h, w = frame.shape[:2]
    size = h // 24
    bold, regular = font("bold", size), font("regular", size)
    r, pad = size // 3, size
    title_w = int(bold.getlength(title + " ")) if title else 0
    pw, ph = pad + 2 * r + pad * 3 // 4 + title_w + int(regular.getlength(rest)) + pad, int(size * 2.2)
    x0, y0 = (w - pw) // 2, h - ph - h // 18

    # the pill is the camera image behind it, blurred and darkened
    behind = frame[y0:y0 + ph, x0:x0 + pw]
    glass = (cv2.GaussianBlur(behind, (0, 0), size / 2) * 0.4).astype(np.uint8)
    pill = Image.fromarray(cv2.cvtColor(glass, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pill, "RGBA")
    cx, cy = pad + r, ph // 2
    if labels and not error:  # soft ring that pulses while someone speaks
        glow = r * (1.9 + 0.5 * np.sin(time.time() * 5))
        draw.ellipse((cx - glow, cy - glow, cx + glow, cy + glow), fill=(*dot, 70))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=dot)
    tx = cx + r + pad * 3 // 4
    draw.text((tx, cy), title, font=bold, fill=(255, 255, 255), anchor="lm")
    draw.text((tx + title_w, cy), rest, font=regular, fill=(205, 205, 205), anchor="lm")

    # rounded corners, drawn 4x larger then shrunk so the edge is smooth
    mask = Image.new("L", (pw * 4, ph * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw * 4 - 1, ph * 4 - 1), radius=ph * 2, fill=255)
    mask = mask.resize((pw, ph), Image.LANCZOS)
    out = Image.composite(pill, Image.fromarray(cv2.cvtColor(behind, cv2.COLOR_BGR2RGB)), mask)
    frame[y0:y0 + ph, x0:x0 + pw] = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def webcam(voiceprint, threshold, camera=0):
    """Webcam window with the speaker label. Audio runs in its own thread so the video stays smooth."""
    state = {"labels": None, "error": None}

    def audio():
        try:
            for _, labels in listen(microphone(), voiceprint, threshold):
                state["labels"] = labels
            state["error"] = "microphone stream ended"
        except Exception as e:  # show it in the window instead of freezing on the last label
            state["error"] = str(e)

    threading.Thread(target=audio, daemon=True).start()
    cap = cv2.VideoCapture(camera)
    assert cap.isOpened(), f"cannot open camera {camera}"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # camera falls back to its nearest mode if unsupported
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("who is speaking", cv2.WINDOW_NORMAL)  # resizable, image scales with the window
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, feels natural
        if frame.shape[0] < 720:  # low-res cameras: enlarge first so the overlay text stays sharp
            frame = cv2.resize(frame, None, fx=720 / frame.shape[0], fy=720 / frame.shape[0])
        draw_status(frame, state["labels"], state["error"])
        cv2.imshow("who is speaking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    voiceprint, threshold = load_voiceprint()
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--check":
        check(voiceprint, threshold)
    elif arg is None:
        webcam(voiceprint, threshold)
    else:
        arg = None if arg == "--no-camera" else arg
        print("listening, Ctrl+C to stop" if arg is None else f"reading {arg}")
        try:
            for t, labels in listen(microphone() if arg is None else recording(arg), voiceprint, threshold):
                print(f"{t:4d}s  {' + '.join(sorted(labels)).upper() or '...'}", flush=True)
        except KeyboardInterrupt:
            pass
