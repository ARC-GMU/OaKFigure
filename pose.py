"""Multi-person pose estimation + tracking on a background thread.

Uses an Ultralytics YOLO pose model (17 COCO keypoints) with ByteTrack for
per-frame tracker IDs, then IdentityManager for colours that stay consistent.
"""

from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np

from camera import FrameSource, Frames

# PyTorch prints a long warning about old GPUs (e.g. GTX 1050) it can't use; we fall back to CPU anyway
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.cuda")
from identity import IdentityManager


@dataclass
class Person:
    identity: int
    color: tuple[int, int, int]
    keypoints: np.ndarray  # (17, 2) pixel coords in the camera frame
    visible: np.ndarray  # (17,) bool
    box: np.ndarray  # (4,) x0, y0, x1, y1
    distance_mm: float | None = None  # from the depth camera, if available


@dataclass
class PoseResult:
    people: list[Person]
    frame_size: tuple[int, int]  # (width, height)
    frames: Frames  # the camera frames this result was computed from
    infer_fps: float


def person_distance(depth: np.ndarray, kps: np.ndarray, visible: np.ndarray, box: np.ndarray) -> float | None:
    """Median depth (mm) sampled around the shoulders and hips, falling back to the box centre."""
    h, w = depth.shape
    pts = [kps[i] for i in (5, 6, 11, 12) if visible[i]]
    if not pts:
        pts = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)]
    samples = []
    for x, y in pts:
        x, y = int(x), int(y)
        patch = depth[max(0, y - 6):min(h, y + 7), max(0, x - 6):min(w, x + 7)]
        patch = patch[(patch > 150) & (patch < 15000)]
        if patch.size:
            samples.append(np.median(patch))
    return float(np.median(samples)) if samples else None


def pick_device(requested: str | None) -> str:
    """Use the GPU only if this PyTorch build actually supports it (e.g. old Pascal cards may not be)."""
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
                return "cuda:0"
            print(f"[pose] GPU sm_{major}{minor} not supported by this PyTorch build, using CPU")
    except Exception as exc:
        print(f"[pose] CUDA check failed ({exc}), using CPU")
    return "cpu"


class Smoother:
    """Per-identity keypoint smoothing; briefly holds joints that drop out."""

    def __init__(self, alpha: float = 0.55, hold_frames: int = 6):
        self.alpha = alpha
        self.hold_frames = hold_frames
        self.state: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # id -> (kps, frames_missing)

    def __call__(self, ident: int, kps: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if ident not in self.state:
            missing = np.where(visible, 0, self.hold_frames + 1)
            self.state[ident] = (kps.copy(), missing)
            return kps, visible
        prev, missing = self.state[ident]
        out = prev.copy()
        jump = np.linalg.norm(kps - prev, axis=1)
        # move faster on big motion (less lag), smoother on small jitter
        a = np.clip(self.alpha + jump / 200.0, self.alpha, 1.0)[:, None]
        fresh = visible & (missing > self.hold_frames)  # joint reappeared: snap to it
        out[visible] = (1 - a[visible]) * prev[visible] + a[visible] * kps[visible]
        out[fresh] = kps[fresh]
        missing = np.where(visible, 0, missing + 1)
        self.state[ident] = (out, missing)
        return out.copy(), missing <= self.hold_frames

    def prune(self, alive: set[int]) -> None:
        for k in list(self.state):
            if k not in alive:
                del self.state[k]


class PoseWorker:
    def __init__(self, source: FrameSource, model: str = "yolo11n-pose.pt", imgsz: int = 640,
                 conf: float = 0.4, kp_conf: float = 0.35, device: str | None = None):
        from ultralytics import YOLO

        self.source = source
        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf
        self.kp_conf = kp_conf
        self.device = pick_device(device)
        self.identities = IdentityManager()
        self.smoother = Smoother()
        self._result: PoseResult | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None
        print(f"[pose] model={model} device={self.device} imgsz={imgsz}")

    def start(self) -> "PoseWorker":
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    def latest(self) -> PoseResult | None:
        with self._lock:
            return self._result

    def _loop(self) -> None:
        last_id = -1
        fps = 0.0
        try:
            while self._running:
                fid, frames = self.source.latest()
                if frames is None or fid == last_id:
                    time.sleep(0.003)
                    continue
                last_id = fid
                t0 = time.perf_counter()
                result = self._process(frames)
                dt = time.perf_counter() - t0
                fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6)) if fps else 1.0 / max(dt, 1e-6)
                result.infer_fps = fps
                with self._lock:
                    self._result = result
        except Exception as exc:
            self.error = exc
            self._running = False

    def _process(self, frames: Frames) -> PoseResult:
        frame = frames.rgb
        h, w = frame.shape[:2]
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz,
                             conf=self.conf, device=self.device, verbose=False)[0]

        tracks = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            xy = r.keypoints.xy.cpu().numpy()
            kconf = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                     else np.ones(xy.shape[:2], np.float32))
            tracks = [(int(t), xy[i], kconf[i], boxes[i]) for i, t in enumerate(ids)]

        mapping = self.identities.update(frame, [t[:3] for t in tracks])

        people = []
        for tid, kps, kconf, box in tracks:
            ident = mapping.get(tid)
            if ident is None:  # still being identified (a fraction of a second)
                continue
            visible = (kconf >= self.kp_conf) & (kps[:, 0] > 0) & (kps[:, 1] > 0)
            kps_s, vis_s = self.smoother(ident.index, kps, visible)
            dist = person_distance(frames.depth, kps, visible, box) if frames.depth is not None else None
            people.append(Person(ident.index, ident.color, kps_s, vis_s, box, dist))
        self.smoother.prune({p.identity for p in people} | set(
            i.index for i in self.identities.identities if i.track_id is not None))

        return PoseResult(people, (w, h), frames, 0.0)
