"""OaKFigure: fun ways to see what an OAK-D Pro camera sees.

    ./run.sh                     # OAK-D Pro
    ./run.sh --auto 20           # kiosk mode: change view every 20 s
    ./run.sh --source 0          # webcam (only the stick-figure views work)

Mouse: click a tab to change view, click Exit to quit.
Keys: 1-8 / ←→ change view · A auto-cycle · F fullscreen · M mirror · H hide bars · Q/Esc quit
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
import traceback

import pygame

from camera import make_source
from pose import Person, PoseWorker
from views import BG, FADE_SECONDS, TEXT, Ctx, make_views

BAR = (17, 17, 27)
TAB_IDLE = (24, 24, 37)
TAB_HOVER = (49, 50, 68)
ACCENT = (137, 180, 250)
EXIT_HOVER = (243, 139, 168)
DISABLED = (88, 91, 112)


@functools.lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("inter,ubuntu,dejavusans,arial,sans", size, bold=bold)


def wrap(text: str, f: pygame.font.Font, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if f.size(trial)[0] <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


class Layout:
    """Screen regions for the flat top bar, bottom tab bar and the view area between them."""

    def __init__(self, size, views, show_bars: bool):
        sw, sh = size
        self.title_f = font(max(26, sh // 22), True)
        self.blurb_f = font(max(16, sh // 40))
        self.tab_f = font(max(15, sh // 46), True)
        self.pad = max(16, sh // 45)
        if not show_bars:
            self.top = self.bottom = pygame.Rect(0, 0, 0, 0)
            self.area = pygame.Rect(0, 0, sw, sh)
            self.exit = None
            self.tabs = []
            return
        exit_w = self.tab_f.size("Exit")[0] + self.pad * 3
        self.blurb_w = sw - exit_w - self.pad * 3
        top_h = self.pad * 2 + self.title_f.get_height() + self.blurb_f.get_linesize() * 2
        bottom_h = self.tab_f.get_height() + self.pad * 2
        self.top = pygame.Rect(0, 0, sw, top_h)
        self.bottom = pygame.Rect(0, sh - bottom_h, sw, bottom_h)
        self.area = pygame.Rect(0, top_h, sw, sh - top_h - bottom_h)
        self.exit = pygame.Rect(sw - exit_w, 0, exit_w, top_h)
        tab_w = sw / len(views)
        self.tabs = [pygame.Rect(round(i * tab_w), self.bottom.y, round((i + 1) * tab_w) - round(i * tab_w),
                                 bottom_h) for i in range(len(views))]


def draw_bars(screen, lay: Layout, views, current, enabled, auto_progress):
    mouse = pygame.mouse.get_pos()
    view = views[current]

    # top bar: title, explanation, exit button
    pygame.draw.rect(screen, BAR, lay.top)
    screen.blit(lay.title_f.render(view.name, True, ACCENT), (lay.pad, lay.pad))
    y = lay.pad + lay.title_f.get_height()
    blurb = view.blurb + (f"   ({view.hint})" if view.hint else "")
    for line in wrap(blurb, lay.blurb_f, lay.blurb_w)[:2]:
        screen.blit(lay.blurb_f.render(line, True, TEXT), (lay.pad, y))
        y += lay.blurb_f.get_linesize()
    if auto_progress is not None:
        pygame.draw.rect(screen, ACCENT, (0, lay.top.bottom - 3, int(lay.top.w * auto_progress), 3))

    hover = lay.exit.collidepoint(mouse)
    if hover:
        pygame.draw.rect(screen, EXIT_HOVER, lay.exit)
    pygame.draw.line(screen, TAB_HOVER, lay.exit.topleft, lay.exit.bottomleft, 1)
    label = lay.tab_f.render("Exit", True, BG if hover else TEXT)
    screen.blit(label, label.get_rect(center=lay.exit.center))

    # bottom bar: one flat tab per view
    for i, (v, rect) in enumerate(zip(views, lay.tabs)):
        if i == current:
            bg, fg = ACCENT, BG
        elif enabled[i] and rect.collidepoint(mouse):
            bg, fg = TAB_HOVER, TEXT
        else:
            bg, fg = TAB_IDLE, TEXT if enabled[i] else DISABLED
        pygame.draw.rect(screen, bg, rect)
        text = f"{i + 1}  {v.name}"
        if lay.tab_f.size(text)[0] > rect.w - 12:
            text = f"{i + 1}  {v.short}"
        if lay.tab_f.size(text)[0] > rect.w - 12:
            text = str(i + 1)
        img = lay.tab_f.render(text, True, fg)
        screen.blit(img, img.get_rect(center=rect.center))
    for rect in lay.tabs[1:]:  # thin separators
        pygame.draw.line(screen, BAR, rect.topleft, rect.bottomleft, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="oak", help="'oak' (default), a webcam index, or a video file")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="Ultralytics pose model, e.g. yolo11n-pose.pt (fast) or yolo11s-pose.pt (more accurate)")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size; 480 or 416 is faster on CPU")
    ap.add_argument("--device", default=None, help="force 'cpu' or 'cuda:0' (default: auto)")
    ap.add_argument("--size", default="1280x720", help="window size")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-mirror", action="store_true", help="don't mirror left/right")
    ap.add_argument("--units", choices=["ft", "m"], default="ft")
    ap.add_argument("--usb2", action="store_true", help="skip trying USB3 (faster start on a USB2 port/cable)")
    ap.add_argument("--view", type=int, default=1, help="view to start on")
    ap.add_argument("--auto", type=float, default=0, metavar="SECONDS",
                    help="start in auto-cycle mode, changing view every SECONDS")
    args = ap.parse_args()

    source = make_source(args.source, usb2=args.usb2)
    try:
        source.start()
    except Exception as exc:
        print(f"Could not open source {args.source!r}: {exc}", file=sys.stderr)
        if args.source == "oak":
            print("Is the OAK-D Pro plugged in and are the udev rules installed? See README.md.\n"
                  "To test without it: ./run.sh --source 0", file=sys.stderr)
        return 1

    worker = None
    try:
        worker = PoseWorker(source, model=args.model, imgsz=args.imgsz, device=args.device).start()
        run_ui(args, source, worker)
    except KeyboardInterrupt:
        pass
    finally:  # always release the camera (and switch its IR projector off)
        if worker is not None:
            worker.stop()
        source.stop()
        pygame.quit()
    return 0


def run_ui(args, source, worker) -> None:
    pygame.init()
    pygame.display.set_caption("OaKFigure")
    win_w, win_h = (int(v) for v in args.size.lower().split("x"))
    flags = pygame.RESIZABLE | (pygame.FULLSCREEN if args.fullscreen else 0)
    screen = pygame.display.set_mode((0, 0) if args.fullscreen else (win_w, win_h), flags)
    clock = pygame.time.Clock()

    views = make_views()
    enabled = [v.available(source) for v in views]
    current = min(max(args.view - 1, 0), len(views) - 1)
    if not enabled[current]:
        current = 0
    auto_seconds = args.auto if args.auto > 0 else 20.0
    auto = args.auto > 0
    view_started = time.monotonic()
    mirror = not args.no_mirror
    show_bars = True

    recent: dict[int, tuple[Person, float]] = {}  # identity -> (last pose, time seen)
    last_result = None
    render_errors: set[str] = set()  # each view's error is printed only once

    def switch(step: int = 0, to: int | None = None):
        nonlocal current, view_started
        idx = current if to is None else to
        for _ in range(len(views)):
            idx = (idx + step) % len(views)
            if enabled[idx]:
                break
            step = step or 1
        current = idx
        view_started = time.monotonic()

    while True:
        sw, sh = screen.get_size()
        lay = Layout((sw, sh), views, show_bars)
        now = time.monotonic()
        _, frames = source.latest()
        result = worker.latest()

        # remember people briefly so they can fade out instead of popping away
        if result is not None and result is not last_result:
            for p in result.people:
                recent[p.identity] = (p, now)
            last_result = result
        live = {p.identity for p in result.people} if result else set()
        fading = []
        for ident, (p, t) in list(recent.items()):
            age = now - t
            if age > FADE_SECONDS:
                del recent[ident]
            elif ident not in live:
                fading.append((p, 1 - age / FADE_SECONDS))

        ctx = Ctx(screen, lay.area, frames, result, mirror, args.units, font, now, fading)
        view = views[current]

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if lay.exit is not None and lay.exit.collidepoint(ev.pos):
                    return
                tab = next((i for i, r in enumerate(lay.tabs) if r.collidepoint(ev.pos)), None)
                if tab is not None:
                    if enabled[tab]:
                        switch(to=tab)
                    continue
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if pygame.K_1 <= ev.key <= pygame.K_9 and ev.key - pygame.K_1 < len(views):
                    if enabled[ev.key - pygame.K_1]:
                        switch(to=ev.key - pygame.K_1)
                elif ev.key in (pygame.K_RIGHT, pygame.K_SPACE):
                    switch(+1)
                elif ev.key == pygame.K_LEFT:
                    switch(-1)
                elif ev.key == pygame.K_a:
                    auto = not auto
                    view_started = now
                elif ev.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif ev.key == pygame.K_m:
                    mirror = not mirror
                elif ev.key == pygame.K_h:
                    show_bars = not show_bars
                else:
                    view.handle_event(ev, ctx)
            else:
                view.handle_event(ev, ctx)
            view = views[current]

        for part in (source, worker):
            if part.error is not None:
                print(f"Error: {part.error!r}", file=sys.stderr)
                return

        if auto and now - view_started > auto_seconds:
            switch(+1)
            view = views[current]

        screen.fill(BG)
        if frames is None:
            img = font(max(24, sh // 20), True).render("Waiting for the camera…", True, TEXT)
            screen.blit(img, img.get_rect(center=lay.area.center))
        else:
            screen.set_clip(lay.area)  # views never draw over the bars
            try:
                view.render(ctx)
            except Exception as exc:  # keep the kiosk running if one view breaks
                if view.name not in render_errors:
                    render_errors.add(view.name)
                    traceback.print_exc()
                screen.fill(BG)
                msg = font(max(22, sh // 26), True).render(
                    f"Oops, {view.name} hit a problem ({type(exc).__name__}). Try another view!", True, TEXT)
                screen.blit(msg, msg.get_rect(center=lay.area.center))
            screen.set_clip(None)

        if show_bars:
            progress = min(1.0, (now - view_started) / auto_seconds) if auto else None
            draw_bars(screen, lay, views, current, enabled, progress)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    sys.exit(main())
