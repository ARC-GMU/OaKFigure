"""Hand signs for Rock Paper Scissors.

The pose model already knows where each player's wrists are, so we cut a small
box around the raised hand and give just that to MediaPipe's gesture recognizer.
That keeps the hand big enough to read even though the camera sees the whole room.
Runs on its own thread so the screen never waits for it.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from figures import L_EL, L_SH, L_WR, R_EL, R_SH, R_WR
from pose import Person

ROCK, PAPER, SCISSORS = "Rock", "Paper", "Scissors"

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
             "gesture_recognizer/float16/latest/gesture_recognizer.task")
MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

# MediaPipe's built-in gestures; a fist with the thumb out still counts as Rock
GESTURES = {"Closed_Fist": ROCK, "Thumb_Up": ROCK, "Thumb_Down": ROCK,
            "Open_Palm": PAPER, "Victory": SCISSORS}

MIN_HAND_PX = 60  # a hand smaller than this (in camera pixels) is too far away to read


def hand_box(p: Person) -> tuple[tuple[float, float, float, float], float] | None:
    """Square box (x0, y0, x1, y1) around the player's higher hand, and the hand's estimated size in pixels."""
    k, v = p.keypoints, p.visible
    if not (v[L_SH] and v[R_SH]):
        return None
    hand = 0.55 * float(np.linalg.norm(k[L_SH] - k[R_SH]))  # a hand is about half a shoulder width
    wrists = [(w, e) for w, e in ((L_WR, L_EL), (R_WR, R_EL)) if v[w]]
    if not wrists:
        return None
    w, e = min(wrists, key=lambda t: k[t[0], 1])  # the raised (higher) hand
    d = k[w] - k[e] if v[e] else np.array([0.0, -1.0])
    n = float(np.linalg.norm(d))
    d = d / n if n > 1 else np.array([0.0, -1.0])
    c = k[w] + d * 0.55 * hand  # the hand sits past the wrist, along the forearm
    half = 1.25 * hand
    return (c[0] - half, c[1] - half, c[0] + half, c[1] + half), hand


def too_far(p: Person) -> bool:
    hb = hand_box(p)
    return hb is not None and hb[1] < MIN_HAND_PX


def count_fingers(lm) -> str | None:
    """Backup when MediaPipe has no named gesture: count the straight fingers."""
    pts = np.array([(q.x, q.y) for q in lm])
    wrist = pts[0]
    straight = [np.linalg.norm(pts[tip] - wrist) > 1.15 * np.linalg.norm(pts[pip] - wrist)
                for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    index, middle, ring, pinky = straight
    if not any(straight):
        return ROCK
    if index and middle and not ring and not pinky:
        return SCISSORS
    if sum(straight) >= 4 or (index and middle and ring):
        return PAPER
    return None


class HandSigns:
    """Background recognizer. submit() the newest pose result, read latest() whenever."""

    def __init__(self):
        self._job = None
        self._out: dict[int, str | None] = {}
        self._version = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.ready = False
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frames, people: list[Person]) -> None:
        boxes = {}
        for p in people:
            hb = hand_box(p)
            if hb is not None and hb[1] >= MIN_HAND_PX:
                boxes[p.identity] = hb[0]
        with self._lock:
            self._job = (frames.rgb, boxes)
        self._wake.set()

    def latest(self) -> tuple[int, dict[int, str | None]]:
        with self._lock:
            return self._version, dict(self._out)

    def _load(self):
        os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe is chatty on startup
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not MODEL_PATH.exists():
            print(f"[hands] downloading {MODEL_PATH.name} (~8 MB)…")
            tmp = MODEL_PATH.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.rename(MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4)
        return mp, vision.GestureRecognizer.create_from_options(options)

    def _read(self, mp, rec, rgb: np.ndarray) -> str | None:
        res = rec.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            return None
        if res.gestures and res.gestures[0]:
            g = res.gestures[0][0]
            if g.category_name in GESTURES and g.score > 0.45:
                return GESTURES[g.category_name]
        return count_fingers(res.hand_landmarks[0])

    def _loop(self) -> None:
        try:
            mp, rec = self._load()
            self.ready = True
            print("[hands] hand-sign recognizer ready")
            while True:
                self._wake.wait()
                self._wake.clear()
                with self._lock:
                    job, self._job = self._job, None
                if job is None:
                    continue
                frame, boxes = job
                h, w = frame.shape[:2]
                out = {}
                for ident, (x0, y0, x1, y1) in boxes.items():
                    x0, y0 = max(0, int(x0)), max(0, int(y0))
                    x1, y1 = min(w, int(x1)), min(h, int(y1))
                    if x1 - x0 < 16 or y1 - y0 < 16:
                        continue
                    crop = np.ascontiguousarray(frame[y0:y1, x0:x1, ::-1])  # BGR -> RGB
                    out[ident] = self._read(mp, rec, crop)
                with self._lock:
                    self._out = out
                    self._version += 1
        except Exception as exc:
            self.error = exc
            print(f"[hands] hand signs unavailable ({exc!r}); using body poses instead")
            time.sleep(0)
