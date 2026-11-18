"""Little pixel-art sprites: animated faces that react to the game, and hand signs.

Everything is drawn on a tiny grid (20x20 for faces, 18x18 for hands) and scaled up
with hard pixel edges. Outlines are added automatically around the filled shapes.
"""

from __future__ import annotations

import numpy as np
import pygame

OUTLINE = (17, 17, 27)
ROBOT = (137, 180, 250)
SCREEN = (30, 30, 46)
KID = (249, 226, 175)
CHEEK = (243, 139, 168)
TEAR = (116, 199, 236)
GOLD = (249, 226, 175)
GLOW = {"idle": (205, 214, 244), "thinking": (205, 214, 244), "happy": (166, 227, 161),
        "party": (166, 227, 161), "sad": (116, 199, 236), "surprised": (249, 226, 175),
        "dizzy": (243, 139, 168)}

MOODS = tuple(GLOW)


def _outline(mask: np.ndarray) -> np.ndarray:
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown & ~mask


def _blit(surf, rgba: np.ndarray, center, size: float) -> pygame.Rect:
    h, w = rgba.shape[:2]
    scale = max(1, int(size // max(w, h)))
    img = pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (w, h), "RGBA")
    img = pygame.transform.scale(img, (w * scale, h * scale))
    rect = img.get_rect(center=(int(center[0]), int(center[1])))
    surf.blit(img, rect)
    return rect


# --------------------------------------------------------------------------
# faces
def _eyes(mood: str, t: float):
    """Pixels (x, y) for the eyes, in the 16x16 head grid."""
    if mood in ("happy", "party"):  # ^ ^
        return [(4, 7), (5, 6), (6, 7), (9, 7), (10, 6), (11, 7)]
    if mood == "sad":  # tired, sloping eyes
        return [(4, 7), (5, 7), (5, 6), (10, 6), (10, 7), (11, 7)]
    if mood == "surprised":  # wide open rings
        ring = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
        return [(4 + x, 5 + y) for x, y in ring] + [(9 + x, 5 + y) for x, y in ring]
    if mood == "dizzy":  # X X
        x_ = [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)]
        return [(4 + x, 5 + y) for x, y in x_] + [(9 + x, 5 + y) for x, y in x_]
    if (t % 3.2) < 0.15:  # blink
        return [(4, 7), (5, 7), (10, 7), (11, 7)]
    dx = 0
    if mood == "thinking":  # look left, then right
        dx = (-1, 0, 1, 0)[int(t * 2.5) % 4]
    return [(4 + dx + x, 6 + y) for x in (0, 1) for y in (0, 1)] + \
           [(10 + dx + x, 6 + y) for x in (0, 1) for y in (0, 1)]


def _mouth(mood: str, t: float):
    if mood in ("happy", "party"):  # open grin
        return ([(x, 9) for x in range(5, 11)] + [(x, 10) for x in range(6, 10)] + [(7, 11), (8, 11)])
    if mood == "sad":
        return [(5, 11), (6, 10), (7, 10), (8, 10), (9, 10), (10, 11)]
    if mood == "surprised":
        return [(7, 9), (8, 9), (6, 10), (9, 10), (7, 11), (8, 11)]
    if mood == "dizzy":  # wobbly mouth that wiggles
        k = int(t * 6) % 2
        return [(x, 10 + ((x + k) % 2)) for x in range(5, 11)]
    if mood == "thinking":
        return [(x, 10) for x in range(6, 10)] + [(10, 9)]
    return [(5, 10), (6, 11), (7, 11), (8, 11), (9, 11), (10, 10)]  # smile


def draw_face(surf, mood: str, center, size: float, t: float, style: str = "robot") -> pygame.Rect:
    """An animated pixel face. style 'robot' (the computer) or 'kid' (a round smiley)."""
    W = 20
    rgba = np.zeros((W, W, 4), np.uint8)
    f = int(t * 4)
    ox, oy = 2, 2
    if mood in ("happy", "party"):
        oy += -1 if f % 2 else 0  # bounce
    elif mood == "surprised" and (t % 1.0) < 0.5:
        ox += (-1, 1)[f % 2]  # shake
    elif mood == "sad":
        oy += 1  # slump

    head = np.zeros((W, W), bool)
    yy, xx = np.mgrid[0:16, 0:16]
    if style == "robot":
        shape = np.zeros((16, 16), bool)
        shape[2:16, 1:15] = True
        shape[7:10, 0] = shape[7:10, 15] = True  # ears
        shape[0:2, 7:9] = True  # antenna
        fill = ROBOT
    else:
        shape = (xx - 7.5) ** 2 + (yy - 8.5) ** 2 <= 7.3 ** 2
        fill = KID
    head[oy:oy + 16, ox:ox + 16] = shape
    rgba[head] = (*fill, 255)
    rgba[_outline(head)] = (*OUTLINE, 255)

    def px(x, y, color):
        X, Y = ox + x, oy + y
        if 0 <= X < W and 0 <= Y < W:
            rgba[Y, X] = (*color, 255)

    if style == "robot":
        rgba[oy + 4:oy + 13, ox + 3:ox + 13] = (*SCREEN, 255)
        bulb = GLOW[mood] if (f % 2 or mood not in ("party", "surprised")) else OUTLINE
        px(7, 0, bulb)
        px(8, 0, bulb)
        feature = GLOW[mood]
    else:
        feature = OUTLINE
        if mood in ("happy", "party"):
            px(3, 9, CHEEK)
            px(12, 9, CHEEK)
    for x, y in _eyes(mood, t) + _mouth(mood, t):
        px(x, y, feature)

    if mood == "sad":  # a tear sliding down
        ty = 8 + int(t * 5) % 5
        px(4, ty, TEAR)
    if mood == "party":  # twinkling stars in the corners
        for i, (x, y) in enumerate(((-2, -1), (17, 0), (-1, 14), (17, 13))):
            if (f + i) % 2:
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    px(x + dx, y + dy, GOLD)
    if mood == "dizzy":  # stars circling the head
        for i in range(3):
            a = t * 4 + i * 2.09
            px(int(round(7.5 + 7 * np.cos(a))), int(round(0 + 1.5 * np.sin(a))), GOLD)
    return _blit(surf, rgba, center, size)


# --------------------------------------------------------------------------
# hand signs
def _hand_shape(throw: str) -> tuple[np.ndarray, list]:
    m = np.zeros((16, 16), bool)
    lines = []  # detail pixels drawn in the outline colour
    if throw == "Rock":  # a fist seen from the front
        m[4:14, 2:13] = True
        m[3, 3:12] = True
        m[14, 3:12] = True
        lines += [(x, y) for x in (4, 7, 10) for y in range(3, 7)]  # gaps between curled fingers
        lines += [(x, 8) for x in range(2, 10)]  # top of the thumb
    elif throw == "Paper":  # open hand, fingers spread
        m[8:15, 3:14] = True
        for (x, top) in ((3, 3), (6, 1), (9, 2), (12, 4)):
            m[top:8, x:x + 2] = True
        for y in range(6, 12):  # thumb out to the side
            x = (y - 6) // 2
            m[y, x:x + 2] = True
    else:  # Scissors: two fingers in a V
        m[8:15, 3:13] = True
        m[7, 4:12] = True
        for y in range(1, 9):
            a = 1 + (y - 1) * 3 // 7
            b = 12 - (y - 1) * 3 // 7
            m[y, a:a + 3] = True
            m[y, b:b + 3] = True
        lines += [(x, 11) for x in range(4, 10)]  # thumb across the curled fingers
    return m, lines


def draw_hand(surf, throw: str, center, size: float, color) -> pygame.Rect:
    rgba = np.zeros((18, 18, 4), np.uint8)
    shape, lines = _hand_shape(throw)
    mask = np.zeros((18, 18), bool)
    mask[1:17, 1:17] = shape
    rgba[mask] = (*color, 255)
    rgba[_outline(mask)] = (*OUTLINE, 255)
    for x, y in lines:
        if shape[y, x]:
            rgba[y + 1, x + 1] = (*OUTLINE, 255)
    return _blit(surf, rgba, center, size)
