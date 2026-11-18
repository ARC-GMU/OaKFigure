"""Stable person identities and pastel colours.

The tracker (ByteTrack) already keeps an ID while a person stays in view or is
briefly hidden. When someone walks out of frame and comes back, the tracker
hands out a *new* ID. To keep that person's colour, we store a small
appearance signature (colour histograms of torso and legs) for every identity
and match new tracks against identities that are currently missing.

A person's colour is tied to their identity, never reused for anyone else.
"""

from __future__ import annotations

import colorsys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# Hand-picked pastels, chosen to be easy to tell apart. Blue comes first.
PASTELS: list[tuple[int, int, int]] = [
    (137, 180, 250),  # blue
    (250, 160, 170),  # pink
    (166, 227, 161),  # mint green
    (249, 226, 145),  # butter yellow
    (203, 166, 247),  # lavender
    (250, 179, 135),  # peach
    (137, 220, 235),  # sky / teal
    (245, 194, 231),  # rose
    (190, 215, 140),  # pistachio
    (180, 190, 254),  # periwinkle
]


def pastel_color(index: int) -> tuple[int, int, int]:
    """Colour for the n-th identity: the fixed palette first, then golden-ratio hues."""
    if index < len(PASTELS):
        return PASTELS[index]
    hue = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.38, 0.97)
    return int(r * 255), int(g * 255), int(b * 255)


# COCO keypoint indices used for appearance regions
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14


def _region_hist(hsv: np.ndarray, poly: np.ndarray) -> np.ndarray | None:
    h, w = hsv.shape[:2]
    poly = np.clip(poly, [0, 0], [w - 1, h - 1]).astype(np.int32)
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    if (x1 - x0) < 6 or (y1 - y0) < 6:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    total = hist.sum()
    if total < 50:  # too few pixels to trust
        return None
    return (hist / total).astype(np.float32)


def appearance(frame_bgr: np.ndarray, kps: np.ndarray, conf: np.ndarray, min_conf: float = 0.5):
    """Signature = (torso histogram, legs histogram); either may be None."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    def ok(*idx):
        return all(conf[i] >= min_conf for i in idx)

    torso = legs = None
    if ok(L_SH, R_SH, L_HIP, R_HIP):
        # shrink the torso quad toward its centre to avoid background pixels
        quad = kps[[L_SH, R_SH, R_HIP, L_HIP]]
        quad = quad.mean(axis=0) + 0.75 * (quad - quad.mean(axis=0))
        torso = _region_hist(hsv, quad)
    if ok(L_HIP, R_HIP, L_KNEE, R_KNEE):
        quad = kps[[L_HIP, R_HIP, R_KNEE, L_KNEE]]
        quad = quad.mean(axis=0) + 0.8 * (quad - quad.mean(axis=0))
        legs = _region_hist(hsv, quad)
    return torso, legs


def _blend(old, new, a):
    if new is None:
        return old
    if old is None:
        return new
    return (1 - a) * old + a * new


def _distance(sig_a, sig_b) -> float | None:
    """Bhattacharyya distance averaged over regions both signatures have (0=same, 1=different)."""
    ds, ws = [], []
    for a, b, weight in ((sig_a[0], sig_b[0], 0.65), (sig_a[1], sig_b[1], 0.35)):
        if a is not None and b is not None:
            ds.append(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
            ws.append(weight)
    if not ds:
        return None
    return float(np.average(ds, weights=ws))


@dataclass
class Identity:
    index: int
    color: tuple[int, int, int]
    torso: np.ndarray | None = None
    legs: np.ndarray | None = None
    samples: int = 0
    last_seen: float = field(default_factory=time.monotonic)
    track_id: int | None = None

    @property
    def signature(self):
        return self.torso, self.legs


class IdentityManager:
    """Maps tracker IDs -> persistent identities (and therefore colours)."""

    def __init__(self, match_threshold: float = 0.42, min_samples: int = 3, max_identities: int = 64):
        self.match_threshold = match_threshold
        self.min_samples = min_samples
        self.max_identities = max_identities
        self.identities: list[Identity] = []
        self.track_to_identity: dict[int, Identity] = {}
        # brand-new tracks wait a few frames before being committed, so we can
        # gather an appearance signature and try to re-identify them first
        self._pending: dict[int, list] = {}
        self._next_index = 0  # never reused, so colours are never handed out twice

    def _new_identity(self) -> Identity:
        ident = Identity(index=self._next_index, color=pastel_color(self._next_index))
        self._next_index += 1
        self.identities.append(ident)
        if len(self.identities) > self.max_identities:
            # forget the identity unseen for the longest (its colour stays retired)
            stale = min((i for i in self.identities if i.track_id is None),
                        key=lambda i: i.last_seen, default=None)
            if stale is not None:
                self.identities.remove(stale)
        return ident

    def update(self, frame: np.ndarray, tracks: list[tuple[int, np.ndarray, np.ndarray]]) -> dict[int, Identity]:
        """tracks: [(track_id, keypoints[17,2], conf[17])]. Returns {track_id: Identity}."""
        now = time.monotonic()
        active_ids = {t[0] for t in tracks}

        # release identities whose track has vanished
        for tid in list(self.track_to_identity):
            if tid not in active_ids:
                self.track_to_identity.pop(tid).track_id = None
        for tid in list(self._pending):
            if tid not in active_ids:
                del self._pending[tid]

        sigs = {tid: appearance(frame, kps, conf) for tid, kps, conf in tracks}

        # known tracks: refresh their signature
        for tid, _, _ in tracks:
            ident = self.track_to_identity.get(tid)
            if ident is None:
                continue
            torso, legs = sigs[tid]
            a = 0.3 if ident.samples < 10 else 0.05  # learn fast at first, then slowly
            ident.torso = _blend(ident.torso, torso, a)
            ident.legs = _blend(ident.legs, legs, a)
            if torso is not None or legs is not None:
                ident.samples += 1
            ident.last_seen = now

        # new tracks: accumulate a signature, then match to a missing identity
        candidates = []
        for tid, _, _ in tracks:
            if tid in self.track_to_identity:
                continue
            pend = self._pending.setdefault(tid, [None, None, 0, now])
            torso, legs = sigs[tid]
            pend[0] = _blend(pend[0], torso, 0.4)
            pend[1] = _blend(pend[1], legs, 0.4)
            pend[2] += 1
            enough = pend[2] >= self.min_samples or (now - pend[3]) > 0.6
            if enough:
                candidates.append(tid)

        missing = [i for i in self.identities if i.track_id is None and i.samples > 0]
        pairs = []
        for tid in candidates:
            pend = self._pending[tid]
            for ident in missing:
                d = _distance((pend[0], pend[1]), ident.signature)
                if d is not None and d < self.match_threshold:
                    pairs.append((d, tid, ident))

        # greedy, best match first, each identity used once
        for d, tid, ident in sorted(pairs, key=lambda p: p[0]):
            if tid in self.track_to_identity or ident.track_id is not None:
                continue
            self._assign(tid, ident, now)

        for tid in candidates:
            if tid not in self.track_to_identity:
                ident = self._new_identity()
                ident.torso, ident.legs = self._pending[tid][0], self._pending[tid][1]
                ident.samples = 1 if (ident.torso is not None or ident.legs is not None) else 0
                self._assign(tid, ident, now)

        return dict(self.track_to_identity)

    def _assign(self, tid: int, ident: Identity, now: float) -> None:
        ident.track_id = tid
        ident.last_seen = now
        self.track_to_identity[tid] = ident
        self._pending.pop(tid, None)
