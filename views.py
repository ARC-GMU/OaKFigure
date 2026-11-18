"""The different ways to look at what the OAK-D Pro sees.

Each view has a kid-friendly title and one-sentence explanation shown on screen.
To add a view, subclass View, implement render(), and add it to make_views().
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pygame

from camera import Frames
from figures import Fit, draw_figure, to_surface
from pose import Person, PoseResult

BG = (30, 30, 46)
TEXT = (205, 214, 244)
SUBTLE = (147, 153, 178)
FADE_SECONDS = 0.35  # figures fade out briefly after a person leaves


@dataclass
class Ctx:
    screen: pygame.Surface
    area: pygame.Rect
    frames: Frames | None  # newest camera frames
    result: PoseResult | None  # newest pose result (has its own matching frames)
    mirror: bool
    units: str
    font: callable  # font(size, bold=False) -> pygame.font.Font
    now: float = field(default_factory=time.monotonic)
    fading: list = field(default_factory=list)  # [(Person, alpha 0..1)] people who just left


def fmt_distance(mm: float, units: str) -> str:
    if units == "m":
        return f"{mm / 1000:.1f} m"
    feet = mm / 304.8
    return f"{round(feet * 2) / 2:g} ft"


def center_text(ctx: Ctx, lines: list[str]) -> None:
    y = ctx.area.centery - len(lines) * 26
    for i, line in enumerate(lines):
        f = ctx.font(max(22, ctx.area.h // 22), bold=(i == 0))
        img = f.render(line, True, TEXT if i == 0 else SUBTLE)
        ctx.screen.blit(img, img.get_rect(midtop=(ctx.area.centerx, y)))
        y += img.get_height() + 10


def draw_people(ctx: Ctx, fit: Fit, label=None, color=None, width_scale=1.0) -> dict[int, tuple]:
    """Draw everyone (fading ones first). Returns {identity: point above head}."""
    tops = {}
    for p, alpha in ctx.fading:
        layer = pygame.Surface(ctx.screen.get_size(), pygame.SRCALPHA)
        draw_figure(layer, p, fit, color=color(p) if color else None, width_scale=width_scale)
        layer.set_alpha(int(255 * alpha))
        ctx.screen.blit(layer, (0, 0))
    for p in sorted(ctx.result.people, key=lambda q: q.identity):
        top = draw_figure(ctx.screen, p, fit, label=label(p) if label else None,
                          font=ctx.font(max(18, ctx.area.h // 34), bold=True),
                          color=color(p) if color else None, width_scale=width_scale)
        if top is not None:
            tops[p.identity] = top
    return tops


def colorize_depth(depth: np.ndarray, near: float = 300, far: float = 5000) -> np.ndarray:
    """uint16 mm -> RGB rainbow, red = close, blue = far, black = unknown."""
    d = depth.astype(np.float32)
    t = np.clip((far - d) / (far - near), 0, 1)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
    img[depth == 0] = 0
    return img


class View:
    name = "View"
    short = "View"
    blurb = ""
    hint = ""  # extra keys for this view
    needs_depth = False
    needs_mono = False  # the two black-and-white stereo cameras
    needs_pose = False

    def __init__(self):
        self._memo_key = None
        self._memo_val = None

    def memo(self, frames, extra, fn):
        """Recompute only when the camera frames (or extra settings) change."""
        key = self._memo_key
        if key is None or key[0] is not frames or key[1] != extra:
            self._memo_val = fn()
            self._memo_key = (frames, extra)
        return self._memo_val

    def available(self, source) -> bool:
        return (not self.needs_depth or source.has_depth) and (not self.needs_mono or source.has_ir)

    def handle_event(self, ev, ctx: Ctx) -> None:
        pass

    def render(self, ctx: Ctx) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------
class StickFigures(View):
    name = "Stick Figures"
    short = "Stick"
    blurb = "The computer finds your head, arms and legs and turns you into a stick figure!"
    needs_pose = True

    def render(self, ctx):
        if ctx.result is None:
            return center_text(ctx, ["Looking for people…"])
        fit = Fit(ctx.result.frame_size, ctx.area, ctx.mirror)
        draw_people(ctx, fit)


class SkeletonVision(View):
    name = "Skeleton Vision"
    short = "Skeleton"
    blurb = "The real camera picture with your stick-figure skeleton drawn on top."
    needs_pose = True

    def render(self, ctx):
        if ctx.result is None:
            return center_text(ctx, ["Looking for people…"])
        fit = Fit(ctx.result.frame_size, ctx.area, ctx.mirror)
        frames = ctx.result.frames
        surf = self.memo(frames, (fit.w, fit.h, ctx.mirror), lambda: pygame.transform.smoothscale(
            to_surface((frames.rgb[:, :, ::-1] * 0.55).astype(np.uint8), ctx.mirror), (fit.w, fit.h)))
        ctx.screen.blit(surf, (fit.ox, fit.oy))
        draw_people(ctx, fit, width_scale=1.2)


class DepthRainbow(View):
    name = "Depth Rainbow"
    short = "Depth"
    blurb = "Red means close, blue means far. The camera uses two eyes to see distance, just like you!"
    hint = "Up/Down keys change the range"
    needs_depth = True

    def __init__(self):
        super().__init__()
        self.far = 5000

    def handle_event(self, ev, ctx):
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_UP, pygame.K_DOWN):
            self.far = int(np.clip(self.far + (500 if ev.key == pygame.K_UP else -500), 1500, 10000))

    def render(self, ctx):
        f = ctx.frames
        if f is None or f.depth is None:
            return center_text(ctx, ["Warming up the depth camera…"])
        fit = Fit(f.depth.shape[::-1], ctx.area, ctx.mirror)
        surf = self.memo(f, (fit.w, fit.h, ctx.mirror, self.far), lambda: pygame.transform.smoothscale(
            to_surface(colorize_depth(f.depth, far=self.far), ctx.mirror), (fit.w, fit.h)))
        ctx.screen.blit(surf, (fit.ox, fit.oy))
        self._legend(ctx)
        self._crosshair(ctx, f, fit)

    def _legend(self, ctx):
        a = ctx.area
        w, h = min(420, a.w // 3), 18
        x, y = a.right - w - 24, a.bottom - 52
        ramp = colorize_depth(np.linspace(300, self.far, w)[None, :].astype(np.uint16), far=self.far)
        bar = pygame.transform.scale(to_surface(ramp), (w, h))
        ctx.screen.blit(bar, (x, y))
        font = ctx.font(18, bold=True)
        for text, pos, anchor in (("close", (x, y + h + 4), "topleft"),
                                  (fmt_distance(self.far, ctx.units) + "  far", (x + w, y + h + 4), "topright")):
            img = font.render(text, True, TEXT)
            ctx.screen.blit(img, img.get_rect(**{anchor: pos}))

    def _crosshair(self, ctx, f, fit):
        h, w = f.depth.shape
        patch = f.depth[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]
        patch = patch[patch > 0]
        c = fit.point(w / 2, h / 2)
        pygame.draw.circle(ctx.screen, (255, 255, 255), c, 14, 3)
        pygame.draw.circle(ctx.screen, (0, 0, 0), c, 17, 1)
        text = fmt_distance(float(np.median(patch)), ctx.units) if patch.size else "?"
        img = ctx.font(max(22, ctx.area.h // 28), bold=True).render(text, True, (255, 255, 255))
        box = img.get_rect(midleft=(c[0] + 26, c[1])).inflate(16, 8)
        pygame.draw.rect(ctx.screen, (0, 0, 0), box)
        ctx.screen.blit(img, img.get_rect(center=box.center))


class HowFar(View):
    name = "How Far Away?"
    short = "How Far"
    blurb = "The camera measures how far away each person is standing. Step closer or back up!"
    needs_depth = True
    needs_pose = True

    def __init__(self):
        super().__init__()
        self.smoothed: dict[int, float] = {}

    def render(self, ctx):
        if ctx.result is None:
            return center_text(ctx, ["Looking for people…"])
        frames = ctx.result.frames
        fit = Fit(ctx.result.frame_size, ctx.area, ctx.mirror)
        if frames.depth is not None:  # faint depth picture in the background
            surf = self.memo(frames, (fit.w, fit.h, ctx.mirror), lambda: pygame.transform.smoothscale(
                to_surface((colorize_depth(frames.depth) * 0.22).astype(np.uint8), ctx.mirror), (fit.w, fit.h)))
            ctx.screen.blit(surf, (fit.ox, fit.oy))

        for p in ctx.result.people:
            if p.distance_mm is not None:
                old = self.smoothed.get(p.identity, p.distance_mm)
                self.smoothed[p.identity] = 0.8 * old + 0.2 * p.distance_mm
        tops = draw_people(ctx, fit, width_scale=1.1)

        font = ctx.font(max(24, ctx.area.h // 22), bold=True)
        for p in ctx.result.people:
            if p.identity not in tops or p.identity not in self.smoothed:
                continue
            img = font.render(fmt_distance(self.smoothed[p.identity], ctx.units), True, BG)
            box = img.get_rect(midbottom=tops[p.identity]).inflate(24, 10)
            pygame.draw.rect(ctx.screen, p.color, box)
            ctx.screen.blit(img, img.get_rect(center=box.center))


class PointCloud(View):
    name = "3D Dots"
    short = "3D"
    blurb = "The camera turns the room into thousands of 3D dots. Drag with the mouse to spin it!"
    hint = "T for rainbow colours"
    needs_depth = True

    def __init__(self, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.yaw = self.pitch = 0.0
        self.user_yaw = self.user_pitch = 0.0
        self.last_touch = -100.0
        self.rainbow = False
        self._grid = None

    def handle_event(self, ev, ctx):
        if ev.type == pygame.MOUSEMOTION and ev.buttons[0]:
            self.user_yaw += ev.rel[0] * 0.008
            self.user_pitch = float(np.clip(self.user_pitch + ev.rel[1] * 0.006, -1.0, 1.0))
            self.last_touch = time.monotonic()
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_t:
            self.rainbow = not self.rainbow

    def _points(self, f: Frames, mirror: bool):
        s = self.stride
        d = f.depth[::s, ::s].astype(np.float32)
        if self._grid is None or self._grid[0].shape != d.shape:
            v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
            self._grid = (u * s, v * s)
        u, v = self._grid
        fx, fy, cx, cy = f.intrinsics
        valid = (d > 200) & (d < 6000)
        z = d[valid] / 1000.0
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        if mirror:
            x = -x
        if self.rainbow:
            cols = colorize_depth(f.depth[::s, ::s])[valid]
        else:
            cols = f.rgb[::s, ::s][valid][:, ::-1]
        return x, y, z, cols

    def render(self, ctx):
        f = ctx.frames
        if f is None or f.depth is None or f.intrinsics is None:
            return center_text(ctx, ["Warming up the depth camera…"])
        x, y, z, cols = self.memo(f, (ctx.mirror, self.rainbow), lambda: self._points(f, ctx.mirror))

        # gently sway on its own when nobody is dragging
        idle = ctx.now - self.last_touch > 4
        if idle:
            self.user_yaw *= 0.97
            self.user_pitch *= 0.97
        auto = 0.55 * math.sin(ctx.now * 0.45) if idle else 0.0
        self.yaw += ((self.user_yaw + auto) - self.yaw) * 0.15
        self.pitch += ((self.user_pitch + 0.12) - self.pitch) * 0.15

        rw, rh = ctx.area.w // 2, ctx.area.h // 2
        img = np.empty((rh, rw, 3), np.uint8)
        img[:] = BG
        if z.size:
            pivot = float(np.clip(np.median(z), 1.0, 3.5))
            zc = z - pivot
            cyaw, syaw = math.cos(self.yaw), math.sin(self.yaw)
            cp, sp = math.cos(self.pitch), math.sin(self.pitch)
            x1 = x * cyaw + zc * syaw
            z1 = -x * syaw + zc * cyaw
            y2 = y * cp - z1 * sp
            z2 = y * sp + z1 * cp + pivot + 0.6
            focal = rw * 0.7
            ok = z2 > 0.2
            px = (rw / 2 + x1[ok] * focal / z2[ok]).astype(np.int32)
            py = (rh / 2 + y2[ok] * focal / z2[ok]).astype(np.int32)
            c, zz = cols[ok], z2[ok]
            order = np.argsort(-zz)  # far first, near dots painted last
            px, py, c = px[order], py[order], c[order]
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                qx, qy = px + dx, py + dy
                inside = (qx >= 0) & (qx < rw) & (qy >= 0) & (qy < rh)
                img[qy[inside], qx[inside]] = c[inside]
        surf = pygame.transform.smoothscale(to_surface(img), ctx.area.size)
        ctx.screen.blit(surf, ctx.area.topleft)


class TwoEyes(View):
    name = "Two Eyes"
    short = "Two Eyes"
    blurb = "The camera has two eyes, like you. Each eye sees a slightly different picture. Try closing one eye!"
    hint = "W to flip between the eyes"
    needs_mono = True

    def __init__(self):
        super().__init__()
        self.wiggle = False

    def handle_event(self, ev, ctx):
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_w:
            self.wiggle = not self.wiggle

    def render(self, ctx):
        f = ctx.frames
        if f is None or f.left is None or f.right is None:
            return center_text(ctx, ["Warming up the stereo cameras…"])
        a = ctx.area
        eyes = [("Left eye", f.left), ("Right eye", f.right)]
        if ctx.mirror:
            eyes.reverse()  # the camera's left eye is on your right when it faces you
        font = ctx.font(max(22, a.h // 26), bold=True)
        top, bottom = a.y + font.get_height() + 24, a.bottom - 20  # room for the labels
        if self.wiggle:
            label, img = eyes[int(ctx.now * 3) % 2]
            panels = [(label, img, pygame.Rect(a.x + 40, top, a.w - 80, bottom - top))]
        else:
            half = a.w // 2
            panels = [(lab, img, pygame.Rect(a.x + i * half + 15, top, half - 30, bottom - top))
                      for i, (lab, img) in enumerate(eyes)]
        for label, img, rect in panels:
            fit = Fit(img.shape[::-1], rect, ctx.mirror)
            fit.blit(ctx.screen, img)
            t = font.render(label, True, TEXT)
            ctx.screen.blit(t, t.get_rect(midbottom=(fit.rect.centerx, fit.rect.y - 8)))


def make_views() -> list[View]:
    return [StickFigures(), SkeletonVision(), DepthRainbow(), HowFar(), PointCloud(), TwoEyes()]
