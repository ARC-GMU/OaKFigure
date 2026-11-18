"""The different ways to look at what the OAK-D Pro sees.

Each view has a kid-friendly title and one-sentence explanation shown on screen.
To add a view, subclass View, implement render(), and add it to make_views().
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import pygame

from camera import Frames
from figures import (L_AN, L_EYE, L_SH, L_WR, NOSE, R_AN, R_EYE, R_SH, R_WR, Fit, draw_figure, ring,
                     thick_line, to_surface)
from hands import PAPER, ROCK, SCISSORS, HandSigns, hand_box, too_far
from pixelart import draw_face, draw_hand
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


def dim_camera(view: View, ctx: Ctx, fit: Fit, level: float) -> None:
    """Draw the pose result's colour frame, darkened, as a background."""
    frames = ctx.result.frames
    surf = view.memo(frames, (fit.w, fit.h, ctx.mirror, level), lambda: pygame.transform.smoothscale(
        to_surface((frames.rgb[:, :, ::-1] * level).astype(np.uint8), ctx.mirror), (fit.w, fit.h)))
    ctx.screen.blit(surf, (fit.ox, fit.oy))


def label_box(ctx: Ctx, text: str, pos, bg, fg=BG, size: int | None = None) -> pygame.Rect:
    """A filled label whose bottom centre sits at pos."""
    img = ctx.font(size or max(24, ctx.area.h // 22), bold=True).render(text, True, fg)
    box = img.get_rect(midbottom=pos).inflate(24, 10)
    pygame.draw.rect(ctx.screen, bg, box, border_radius=8)
    ctx.screen.blit(img, img.get_rect(center=box.center))
    return box


# --------------------------------------------------------------------------
# Rock Paper Scissors: hand signs (MediaPipe) or whole-body poses, best of 3 against the computer
THROWS = (ROCK, PAPER, SCISSORS)
BEATS = {ROCK: SCISSORS, PAPER: ROCK, SCISSORS: PAPER}
WHY = {ROCK: "Rock smashes Scissors", PAPER: "Paper covers Rock", SCISSORS: "Scissors cut Paper"}
HOW = {"hands": {ROCK: "make a fist", PAPER: "open hand", SCISSORS: "two fingers"},
       "body": {ROCK: "hands on head", PAPER: "arms out wide", SCISSORS: "arms up in a V"}}
WIN, TIE, LOSE = (166, 227, 161), (249, 226, 175), (243, 139, 168)
CLOSER = (250, 179, 135)
WINS_NEEDED = 2  # best of 3

# little stick-figure icons: arm (shoulder, elbow, wrist) for the left side; the right side is mirrored
_ICON_ARMS = {
    ROCK: ((-0.17, -0.5), (-0.36, -0.78), (-0.07, -0.92)),
    PAPER: ((-0.17, -0.5), (-0.45, -0.5), (-0.72, -0.5)),
    SCISSORS: ((-0.17, -0.5), (-0.3, -0.8), (-0.45, -1.08)),
}


def classify_throw(p: Person) -> str | None:
    """Which body pose is this person making? None if it isn't clearly one of the three."""
    k, v = p.keypoints, p.visible
    if not (v[L_SH] and v[R_SH] and v[L_WR] and v[R_WR]):
        return None
    sw = max(float(np.linalg.norm(k[L_SH] - k[R_SH])), 10.0)  # shoulder width, the unit of size
    sh_y = (k[L_SH, 1] + k[R_SH, 1]) / 2
    head_y = k[NOSE, 1] if v[NOSE] else sh_y - 0.6 * sw
    wy = k[[L_WR, R_WR], 1]
    spread = abs(k[L_WR, 0] - k[R_WR, 0])
    if spread > 2.3 * sw and ((wy > sh_y - 0.6 * sw) & (wy < sh_y + 0.9 * sw)).all():
        return PAPER
    if spread > 1.6 * sw and (wy < head_y - 0.2 * sw).all():
        return SCISSORS
    if spread < 1.4 * sw and (wy < sh_y - 0.4 * sw).all():
        return ROCK
    return None


def draw_body_icon(surf, throw: str, center, size: float, color) -> None:
    cx, cy = center[0], center[1] + 0.2 * size
    P = lambda x, y: (cx + x * size, cy + y * size)  # noqa: E731
    w = max(3.0, size * 0.09)
    thick_line(surf, color, P(-0.17, -0.5), P(0.17, -0.5), w)
    thick_line(surf, color, P(0, -0.5), P(0, 0), w)
    for sx in (-1, 1):
        thick_line(surf, color, P(0, 0), P(0.2 * sx, 0.62), w)
        sh, el, wr = ((x * -sx, y) for x, y in _ICON_ARMS[throw])
        thick_line(surf, color, P(*sh), P(*el), w)
        thick_line(surf, color, P(*el), P(*wr), w)
    ring(surf, color, P(0, -0.72), 0.13 * size, w)


class RockPaperScissors(View):
    name = "Rock Paper Scissors"
    short = "RPS"
    BLURBS = {
        "hands": "Play the computer with hand signs! Fist = Rock, open hand = Paper, two fingers = Scissors. "
                 "Win 2 rounds to win the game!",
        "body": "Play the computer with your whole body! Hands on head = Rock, arms out wide = Paper, "
                "arms up in a V = Scissors. Win 2 rounds to win!",
    }
    hint = "B switches hands/body · N new game"
    needs_pose = True

    BEAT = 0.75  # seconds per countdown word
    COUNT = ("Rock…", "Paper…", "Scissors…")
    SHOOT = {"hands": 1.0, "body": 0.8}  # seconds to catch everyone's throw after "Shoot!"
    REVEAL = 4.5
    MATCH_OVER = 6.0

    def __init__(self):
        super().__init__()
        self.rng = random.Random()
        self.mode = "hands"
        self.blurb = self.BLURBS[self.mode]
        self.hands: HandSigns | None = None
        self.hands_seen = -1
        self.new_game()

    # -- game state -----------------------------------------------------------
    def new_game(self):
        self.phase, self.t0 = "idle", 0.0
        self.computer: str | None = None
        self.votes: dict[int, Counter] = {}
        self.results: dict[int, str | None] = {}
        self.match: dict[int, dict] = {}  # identity -> {"you": wins, "cpu": wins}
        self.round = 0
        self.live: dict[int, str | None] = {}
        self.far: set[int] = set()
        self.last_result = None
        self.last_render = -100.0
        self.people_since: float | None = None
        self.empty_since: float | None = None

    def _set_mode(self, mode):
        self.mode = mode
        self.blurb = self.BLURBS[mode]
        self.new_game()

    def handle_event(self, ev, ctx):
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
            self.new_game()
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_b:
            self._set_mode("body" if self.mode == "hands" else "hands")

    def _go(self, phase, now):
        self.phase, self.t0 = phase, now

    @staticmethod
    def _decided(m) -> bool:
        return m["you"] >= WINS_NEEDED or m["cpu"] >= WINS_NEEDED

    def _step(self, ctx, people):
        now = ctx.now
        if now - self.last_render > 1.0:  # came back to this view: start fresh
            self.new_game()
            self.last_render = now
        self.last_render = now
        present = {p.identity for p in people}
        self.people_since = (self.people_since or now) if people else None
        self.empty_since = None if people else (self.empty_since or now)
        close = present - self.far
        t = now - self.t0
        if self.phase == "idle":
            if self.people_since is not None and now - self.people_since > 1.0:
                self.match, self.round = {}, 0
                self._go("ready", now)
        elif (self.empty_since is not None and now - self.empty_since > 3.0
              and self.phase not in ("reveal", "match_over")):
            self._go("idle", now)
        elif self.phase == "ready" and t > 1.5 and close:  # wait until someone is close enough
            self.round += 1
            self._go("count", now)
        elif self.phase == "count" and t > self.BEAT * len(self.COUNT):
            self.computer = self.rng.choice(THROWS)
            self.votes = {}
            self._go("shoot", now)
        elif self.phase == "shoot" and t > self.SHOOT[self.mode]:
            self._score()
            self._go("reveal", now)
        elif self.phase == "reveal" and t > self.REVEAL:
            playing = [m for i, m in self.match.items() if i in present]
            if self.match and all(self._decided(m) for m in playing):
                self._go("match_over", now)
            else:
                self._go("ready" if people else "idle", now)
        elif self.phase == "match_over" and t > self.MATCH_OVER:
            self.match, self.round = {}, 0
            self._go("ready" if people else "idle", now)

    def _score(self):
        self.results = {}
        for ident, votes in self.votes.items():
            m = self.match.setdefault(ident, {"you": 0, "cpu": 0})
            if self._decided(m):
                continue  # this player's game is already over
            throw = votes.most_common(1)[0][0] if votes else None
            self.results[ident] = throw
            if throw is None or throw == self.computer:
                continue
            m["you" if BEATS[throw] == self.computer else "cpu"] += 1

    def _outcome(self, ident) -> str | None:
        throw = self.results.get(ident)
        if throw is None:
            return None
        return "tie" if throw == self.computer else ("win" if BEATS[throw] == self.computer else "lose")

    def _computer_mood(self) -> tuple[str, str]:
        if self.phase == "count":
            return "thinking", "Hmm…"
        if self.phase == "shoot":
            return "surprised", "Shoot!"
        if self.phase == "reveal":
            outs = [self._outcome(i) for i in self.results]
            kid, cpu = outs.count("win"), outs.count("lose")
            if cpu > kid:
                return "happy", "Yes! I win!"
            if kid > cpu:
                return "sad", "Aww, you got me!"
            if outs.count("tie"):
                return "surprised", "Whoa, same!"
            return "thinking", "Did you throw?"
        if self.phase == "match_over":
            ms = list(self.match.values())
            kid = sum(m["you"] >= WINS_NEEDED for m in ms)
            cpu = sum(m["cpu"] >= WINS_NEEDED for m in ms)
            return ("party", "I'm the champion!") if cpu > kid else ("dizzy", "You beat me!")
        if self.phase == "ready" and self.far and not ({p for p in self.live} - self.far):
            return "thinking", "Come closer!"
        return "idle", "Let's play!" if self.phase == "ready" else "Wanna play?"

    # -- reading throws -------------------------------------------------------
    def _read_throws(self, ctx, people):
        """Update self.live / self.far; add votes while the shoot window is open."""
        if self.mode == "hands":
            if self.hands is None:
                self.hands = HandSigns()
            if self.hands.error is not None:  # MediaPipe missing or failed: play with bodies instead
                self._set_mode("body")
                return
            self.far = {p.identity for p in people if too_far(p)}
            if ctx.result is not self.last_result:
                self.hands.submit(ctx.result.frames, [p for p in people if p.identity not in self.far])
            version, signs = self.hands.latest()
            fresh = version != self.hands_seen
            self.hands_seen = version
            self.live = {p.identity: signs.get(p.identity) for p in people}
        else:
            self.far = set()
            fresh = ctx.result is not self.last_result
            if fresh:
                self.live = {p.identity: classify_throw(p) for p in people}
        self.last_result = ctx.result
        if fresh and self.phase == "shoot":
            for p in people:
                if p.identity in self.far:
                    continue  # too far away to play this round
                c = self.votes.setdefault(p.identity, Counter())
                if self.live.get(p.identity):
                    c[self.live[p.identity]] += 1

    # -- drawing ----------------------------------------------------------------
    def render(self, ctx):
        if ctx.result is None:
            return center_text(ctx, ["Looking for people…"])
        people = ctx.result.people
        self._read_throws(ctx, people)
        self._step(ctx, people)

        fit = Fit(ctx.result.frame_size, ctx.area, ctx.mirror)
        dim_camera(self, ctx, fit, 0.3)
        tops = draw_people(ctx, fit, width_scale=1.1)
        if self.mode == "hands":
            self._hand_boxes(ctx, fit, people)
        self._player_labels(ctx, people, tops)
        self._banner(ctx)
        self._computer_panel(ctx)
        self._scoreboard(ctx, people)
        self._legend(ctx)

    def _hand_boxes(self, ctx, fit, people):
        """Show which hand the camera is reading."""
        for p in people:
            hb = hand_box(p)
            if hb is None or p.identity in self.far:
                continue
            (x0, y0, x1, y1), _ = hb
            pts = fit.points(np.array([[x0, y0], [x1, y1]], float))
            rect = pygame.Rect(int(pts[:, 0].min()), int(pts[:, 1].min()),
                               int(abs(pts[1, 0] - pts[0, 0])), int(abs(pts[1, 1] - pts[0, 1])))
            pygame.draw.rect(ctx.screen, p.color, rect, 3, border_radius=10)

    def _player_labels(self, ctx, people, tops):
        small = max(20, ctx.area.h // 28)
        face = max(60, ctx.area.h // 6)
        for p in people:
            if p.identity not in tops:
                continue
            top = tops[p.identity]
            m = self.match.get(p.identity)
            if self.phase == "match_over" and m:
                won = m["you"] >= WINS_NEEDED
                box = label_box(ctx, "Champion!" if won else "Good try!", top, WIN if won else TIE)
                draw_face(ctx.screen, "party" if won else "sad", (box.left - face * 0.6, box.centery),
                          face, ctx.now, "kid")
            elif self.phase == "reveal" and p.identity in self.results:
                out = self._outcome(p.identity)
                if out is None:
                    label_box(ctx, "No throw?", top, SUBTLE, size=small)
                    continue
                text, color, mood = {"win": ("You win!", WIN, "happy"), "tie": ("Tie!", TIE, "surprised"),
                                     "lose": ("Computer wins", LOSE, "sad")}[out]
                box = label_box(ctx, text, top, color)
                label_box(ctx, self.results[p.identity], box.midtop, p.color, size=small)
                draw_face(ctx.screen, mood, (box.left - face * 0.6, box.centery), face, ctx.now, "kid")
            elif m and self._decided(m) and self.phase != "idle":
                label_box(ctx, "Done! Wait for the others", top, SUBTLE, size=small)
            elif p.identity in self.far:
                label_box(ctx, "Come closer!", top, CLOSER, size=small)
            elif self.live.get(p.identity):
                label_box(ctx, self.live[p.identity], top, p.color, size=small)

    def _banner(self, ctx):
        a = ctx.area
        t = ctx.now - self.t0
        anyone_close = bool(set(self.live) - self.far)
        if self.phase == "idle":
            words = "Step in to play!"
        elif self.phase == "ready":
            words = "Come closer!" if not anyone_close and self.live else f"Round {self.round + 1}"
        elif self.phase == "count":
            words = self.COUNT[min(int(t / self.BEAT), len(self.COUNT) - 1)]
        elif self.phase == "shoot":
            words = "SHOOT!"
        else:  # reveal / match over: the labels and the computer's face tell the story
            return
        img = ctx.font(max(40, a.h // 9), bold=True).render(words, True, (255, 255, 255))
        box = img.get_rect(center=(a.centerx, a.y + a.h * 0.14)).inflate(40, 16)
        pygame.draw.rect(ctx.screen, CLOSER if words == "Come closer!" else (0, 0, 0), box, border_radius=14)
        ctx.screen.blit(img, img.get_rect(center=box.center))

    def _computer_panel(self, ctx):
        a = ctx.area
        w = max(200, a.w // 5)
        f = ctx.font(max(18, a.h // 32), bold=True)
        face = w * 0.62
        show_throw = self.phase in ("shoot", "reveal") and self.computer
        icon = face * 0.62
        h = int(face + f.get_linesize() * 2 + (icon + f.get_linesize() if show_throw else 0) + 36)
        panel = pygame.Rect(a.right - w - 20, a.y + 20, w, h)
        pygame.draw.rect(ctx.screen, (17, 17, 27), panel, border_radius=14)
        mood, says = self._computer_mood()
        y = panel.y + 10
        draw_face(ctx.screen, mood, (panel.centerx, y + face / 2), face, ctx.now)
        y += face + 4
        for text, color in (("Computer", TEXT), (says, SUBTLE)):
            img = f.render(text, True, color)
            ctx.screen.blit(img, img.get_rect(midtop=(panel.centerx, y)))
            y += f.get_linesize()
        if show_throw:
            y += 6
            if self.mode == "hands":
                draw_hand(ctx.screen, self.computer, (panel.centerx, y + icon / 2), icon, (137, 180, 250))
            else:
                draw_body_icon(ctx.screen, self.computer, (panel.centerx, y + icon / 2), icon * 0.55,
                               (137, 180, 250))
            img = f.render(self.computer, True, (137, 180, 250))
            ctx.screen.blit(img, img.get_rect(midtop=(panel.centerx, y + icon)))
        if self.phase == "reveal" and self.computer:
            y = panel.bottom + 8
            for throw in THROWS:  # explain each rule that decided a game this round
                if throw in self.results.values() and throw != self.computer:
                    winner = throw if BEATS[throw] == self.computer else self.computer
                    img = f.render(WHY[winner], True, TEXT)
                    ctx.screen.blit(img, img.get_rect(midtop=(panel.centerx, y)))
                    y += f.get_linesize()

    def _scoreboard(self, ctx, people):
        rows = [(p.color, self.match[p.identity]) for p in sorted(people, key=lambda q: q.identity)
                if p.identity in self.match]
        f = ctx.font(max(18, ctx.area.h // 32), bold=True)
        x, y = ctx.area.x + 20, ctx.area.y + 20
        ctx.screen.blit(f.render("Best of 3", True, TEXT), (x, y))
        y += f.get_linesize() + 6
        r = f.get_height() // 2
        pip = max(5, r // 2)
        for color, m in rows:
            pygame.draw.circle(ctx.screen, color, (x + r, y + r), r)
            px = x + 2 * r + 12
            for who, c in (("you", WIN), ("cpu", LOSE)):  # filled pips = rounds won
                for k in range(WINS_NEEDED):
                    cx = px + pip + k * (pip * 2 + 6)
                    pygame.draw.circle(ctx.screen, c, (cx, y + r), pip, 0 if m[who] > k else 2)
                px += WINS_NEEDED * (pip * 2 + 6) + 6
                if who == "you":
                    dash = f.render("vs", True, SUBTLE)
                    ctx.screen.blit(dash, dash.get_rect(midleft=(px, y + r)))
                    px += dash.get_width() + 8
            y += f.get_linesize() + 6
        if rows:
            note = ctx.font(max(14, ctx.area.h // 46)).render("you  vs  computer", True, SUBTLE)
            ctx.screen.blit(note, (x + 2 * r + 12, y))

    def _legend(self, ctx):
        a = ctx.area
        f = ctx.font(max(16, a.h // 40), bold=True)
        slot = min(220, a.w // 5)
        size = max(48, a.h // 9)
        y = a.bottom - f.get_linesize() * 2 - 12
        backing = pygame.Surface((slot * 3, int(size * 1.35) + f.get_linesize() * 2), pygame.SRCALPHA)
        backing.fill((17, 17, 27, 210))
        ctx.screen.blit(backing, backing.get_rect(midbottom=(a.centerx, a.bottom - 4)))
        for i, throw in enumerate(THROWS):
            cx = a.centerx + (i - 1) * slot
            if self.mode == "hands":
                draw_hand(ctx.screen, throw, (cx, y - size * 0.55), size, TEXT)
            else:
                draw_body_icon(ctx.screen, throw, (cx, y - size * 0.6), size * 0.75, TEXT)
            for j, line in enumerate((throw, HOW[self.mode][throw])):
                img = f.render(line, True, TEXT if j == 0 else SUBTLE)
                ctx.screen.blit(img, img.get_rect(midtop=(cx, y + j * f.get_linesize())))


# --------------------------------------------------------------------------
def fmt_height(mm: float, units: str) -> str:
    if units == "m":
        return f"{round(mm / 10)} cm"
    inches = round(mm / 25.4)
    return f"{inches // 12} ft {inches % 12} in"


class HeightMeter(View):
    name = "Height Meter"
    short = "Height"
    blurb = "The camera knows how far away you are, so it can work out how tall you are. Stand up straight and show your feet!"
    needs_depth = True
    needs_pose = True

    EYE_TO_CROWN = 115  # mm, used if the depth image can't find the top of the head
    ANKLE_TO_FLOOR = 65  # mm

    def __init__(self):
        super().__init__()
        self.samples: dict[int, deque] = {}
        self.marks: dict[int, tuple[float, float, float]] = {}  # identity -> smoothed (top, bottom, px per mm)
        self.seen: dict[int, float] = {}
        self.reason: dict[int, str] = {}
        self.last_result = None

    def _measure(self, p: Person, frames: Frames):
        """(height mm, top y, bottom y, px per mm) in frame pixels, or a hint string."""
        if p.distance_mm is None or frames.depth is None or frames.intrinsics is None:
            return "Measuring…"
        k, v = p.keypoints, p.visible
        depth = frames.depth
        h, w = depth.shape
        z = p.distance_mm
        ppm = frames.intrinsics[1] / z  # pixels per mm at this person's distance

        ankles = [k[i, 1] for i in (L_AN, R_AN) if v[i] and k[i, 1] < h - 4]
        if not ankles:
            return "Show me your feet!"
        bottom = max(ankles) + self.ANKLE_TO_FLOOR * ppm
        if bottom >= h:
            return "Step back a little!"
        eyes = [k[i] for i in (L_EYE, R_EYE) if v[i]]
        if eyes:
            eye = np.mean(eyes, axis=0)
        elif v[NOSE]:
            eye = k[NOSE] - (0, 30 * ppm)
        else:
            return "Look at the camera!"

        # top of the head: highest rows above the eyes that are at this person's distance
        top = eye[1] - self.EYE_TO_CROWN * ppm
        y0, y1 = int(eye[1] - 260 * ppm), int(eye[1] - 40 * ppm)
        if y0 < 0:
            return "Step back a little!"
        half = max(4, int(55 * ppm))
        x0, x1 = max(0, int(eye[0]) - half), min(w, int(eye[0]) + half)
        patch = depth[y0:y1, x0:x1].astype(np.int32)
        if patch.size:
            good = (((patch > 0) & (np.abs(patch - z) < 450)).mean(axis=1) > 0.25).astype(int)
            run = np.flatnonzero(np.convolve(good, np.ones(4, int), "valid") == 4)
            if run.size:
                top = y0 + run[0]

        height = (bottom - top) / ppm
        if not 700 < height < 2300:
            return "Stand up straight!"
        return height, top, bottom, ppm

    def _update(self, ctx):
        frames = ctx.result.frames
        for p in ctx.result.people:
            self.seen[p.identity] = ctx.now
            m = self._measure(p, frames)
            if isinstance(m, str):
                self.reason[p.identity] = m
                continue
            self.reason.pop(p.identity, None)
            height, top, bottom, ppm = m
            self.samples.setdefault(p.identity, deque(maxlen=40)).append(height)
            old = self.marks.get(p.identity, (top, bottom, ppm))
            self.marks[p.identity] = tuple(0.75 * o + 0.25 * n for o, n in zip(old, (top, bottom, ppm)))
        for ident in [i for i, t in self.seen.items() if ctx.now - t > 5]:
            for d in (self.seen, self.samples, self.marks, self.reason):
                d.pop(ident, None)

    def render(self, ctx):
        if ctx.result is None:
            return center_text(ctx, ["Looking for people…"])
        if ctx.result is not self.last_result:
            self.last_result = ctx.result
            self._update(ctx)

        fit = Fit(ctx.result.frame_size, ctx.area, ctx.mirror)
        dim_camera(self, ctx, fit, 0.45)
        tops = draw_people(ctx, fit)
        small = max(20, ctx.area.h // 28)
        tick_f = ctx.font(max(14, ctx.area.h // 48), bold=True)
        unit = 304.8 if ctx.units == "ft" else 500.0  # a tick every foot or every half metre

        for p in ctx.result.people:
            if p.identity not in tops:
                continue
            samples = self.samples.get(p.identity)
            if p.identity in self.reason or not samples or len(samples) < 6:
                label_box(ctx, self.reason.get(p.identity, "Measuring…"), tops[p.identity], SUBTLE, size=small)
                continue
            height = float(np.median(samples))
            top, bottom, ppm = self.marks[p.identity]

            # ruler beside the person, on whichever side has more room
            xs = fit.points(p.keypoints[p.visible])[:, 0]
            right = xs.max() + 40
            x = int(right if right < ctx.area.right - 80 else xs.min() - 40)
            cx = int(np.mean(xs))
            y_top, y_bot = fit.point(0, top)[1], fit.point(0, bottom)[1]
            for y in (y_top, y_bot):  # head and floor lines
                pygame.draw.line(ctx.screen, p.color, (cx, y), (x, y), 2)
            thick_line(ctx.screen, p.color, (x, y_top), (x, y_bot), 6)
            n = 1
            while n * unit < height:
                y = fit.point(0, bottom - n * unit * ppm)[1]
                pygame.draw.line(ctx.screen, p.color, (x - 12, y), (x + 12, y), 3)
                txt = f"{n} ft" if ctx.units == "ft" else f"{n * unit / 1000:g} m"
                img = tick_f.render(txt, True, TEXT)
                ctx.screen.blit(img, img.get_rect(midleft=(x + 16, y)))
                n += 1
            label_box(ctx, fmt_height(height, ctx.units), (cx, min(tops[p.identity][1], y_top - 6)), p.color)


def make_views() -> list[View]:
    return [StickFigures(), SkeletonVision(), DepthRainbow(), HowFar(), PointCloud(), TwoEyes(),
            RockPaperScissors(), HeightMeter()]
