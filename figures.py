"""Drawing helpers: anti-aliased stick figures and frame-to-screen fitting."""

from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.gfxdraw

from pose import Person

# COCO keypoints
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
LIMBS = [
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_HIP, R_HIP), (L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN),
]


class Fit:
    """Letterboxes a camera frame of size (fw, fh) into a screen rect, optionally mirrored."""

    def __init__(self, frame_size, rect: pygame.Rect, mirror: bool):
        self.fw, self.fh = frame_size
        self.mirror = mirror
        self.scale = min(rect.w / self.fw, rect.h / self.fh)
        self.w, self.h = int(self.fw * self.scale), int(self.fh * self.scale)
        self.ox = rect.x + (rect.w - self.w) // 2
        self.oy = rect.y + (rect.h - self.h) // 2

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.ox, self.oy, self.w, self.h)

    def points(self, xy: np.ndarray) -> np.ndarray:
        """Frame pixel coords (N, 2) -> screen coords (N, 2)."""
        x = (self.fw - xy[:, 0]) if self.mirror else xy[:, 0]
        return np.stack([x * self.scale + self.ox, xy[:, 1] * self.scale + self.oy], axis=1)

    def point(self, x: float, y: float) -> tuple[int, int]:
        p = self.points(np.array([[x, y]], float))[0]
        return int(p[0]), int(p[1])

    def blit(self, screen: pygame.Surface, img_rgb: np.ndarray, smooth: bool = True) -> None:
        """Draw an RGB (or gray) numpy image of the frame's size, fitted to the rect."""
        surf = to_surface(img_rgb, self.mirror)
        scaler = pygame.transform.smoothscale if smooth else pygame.transform.scale
        screen.blit(scaler(surf, (self.w, self.h)), (self.ox, self.oy))


def to_surface(img: np.ndarray, mirror: bool = False) -> pygame.Surface:
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if mirror:
        img = img[:, ::-1]
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return pygame.image.frombuffer(img.tobytes(), (img.shape[1], img.shape[0]), "RGB")


def thick_line(surf, color, a, b, width):
    """Anti-aliased thick line with round caps."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    r = width / 2
    if length > 0.5:
        nx, ny = -dy / length * r, dx / length * r
        poly = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
        pygame.gfxdraw.aapolygon(surf, poly, color)
        pygame.gfxdraw.filled_polygon(surf, poly, color)
    for x, y in (a, b):
        pygame.gfxdraw.aacircle(surf, int(x), int(y), int(r), color)
        pygame.gfxdraw.filled_circle(surf, int(x), int(y), int(r), color)


def ring(surf, color, center, radius, width):
    """Anti-aliased circle outline."""
    cx, cy, radius = int(center[0]), int(center[1]), int(radius)
    for k in range(max(1, int(width))):
        pygame.gfxdraw.aacircle(surf, cx, cy, radius - k, color)
    pygame.draw.circle(surf, color, (cx, cy), radius, max(1, int(width)))


def draw_figure(surf, p: Person, fit: Fit, label: str | None = None, font=None,
                color=None, width_scale: float = 1.0):
    """Draw one stick figure. Returns the screen point just above the head (for labels), or None."""
    pts = fit.points(p.keypoints)
    vis = p.visible
    if vis.sum() < 4:
        return None

    # size the stroke from the person's on-screen height
    ys = pts[vis, 1]
    height = max(ys.max() - ys.min(), 40)
    width = float(np.clip(height * 0.03, 3, 16)) * width_scale
    color = color or p.color

    def mid(i, j):
        if vis[i] and vis[j]:
            return (pts[i] + pts[j]) / 2
        if vis[i]:
            return pts[i]
        if vis[j]:
            return pts[j]
        return None

    neck = mid(L_SH, R_SH)
    hip = mid(L_HIP, R_HIP)

    # head: centre on the face points, size from ears / shoulders / eyes
    face = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if vis[i]]
    head_c = head_r = None
    if face:
        head_c = pts[face].mean(axis=0)
        if vis[L_EAR] and vis[R_EAR]:
            head_r = np.linalg.norm(pts[L_EAR] - pts[R_EAR]) * 0.65
        elif vis[L_SH] and vis[R_SH]:
            head_r = np.linalg.norm(pts[L_SH] - pts[R_SH]) * 0.32
        elif vis[L_EYE] and vis[R_EYE]:
            head_r = np.linalg.norm(pts[L_EYE] - pts[R_EYE]) * 1.6
        else:
            head_r = width * 3
        head_r = max(head_r, width * 2.2)

    for i, j in LIMBS:
        if vis[i] and vis[j]:
            thick_line(surf, color, pts[i], pts[j], width)
    if neck is not None and hip is not None:
        thick_line(surf, color, neck, hip, width)
    if neck is not None and head_c is not None:
        d = neck - head_c  # neck runs from the shoulders to the bottom of the head
        n = np.linalg.norm(d)
        if n > head_r:
            thick_line(surf, color, neck, head_c + d / n * head_r, width)
    if head_c is not None:
        ring(surf, color, head_c, head_r, width)

    top = head_c - (0, head_r + width + 4) if head_c is not None else pts[vis].min(axis=0)
    top = (int(top[0]), int(top[1]))  # plain ints: pygame rejects numpy floats
    if label and font is not None:
        text = font.render(label, True, color)
        surf.blit(text, text.get_rect(midbottom=top))
    return top
