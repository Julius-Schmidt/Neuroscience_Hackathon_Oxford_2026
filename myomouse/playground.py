"""Playground window for `python -m myomouse.run`: see what the bands do to the mouse.

The bands still move the REAL mouse. This window just listens to the mouse like any app,
so whatever happens here also happens on your desktop:
  - the cursor leaves a trail, every click shows a ripple (blue = left, orange = right)
  - LEFT click a block -> it changes colour
  - RIGHT button held on a block -> drag it somewhere else, release to drop
  - the side panel shows each band, its live fist probability and the last events

With --dry-run the real mouse isn't touched, so the window draws its own cursor instead.
"""

import os
import time
from collections import deque

from . import config as C

BG, PANEL = (18, 20, 26), (28, 31, 39)
FG, DIM = (225, 228, 235), (120, 126, 140)
BUTTON_COLOR = {"left": (70, 140, 255), "right": (255, 150, 60)}
PALETTE = [(231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182), (236, 240, 241)]
PANEL_W = 330


class Playground:
    """Blocks + clicks, independent of where the mouse events come from (testable)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.held = set()
        self.drag = None            # (block, dx, dy) while the right button holds a block
        self.trail = deque(maxlen=60)
        self.ripples = deque(maxlen=20)  # (x, y, button, t)
        self.log = deque(maxlen=12)      # (t, text, color)
        self.clicks = {"left": 0, "right": 0}
        self.blocks = []
        for i in range(6):
            col, row = i % 3, i // 3
            self.blocks.append({"rect": [80 + col * 200, 140 + row * 200, 120, 120], "color": i})

    def block_at(self, x, y):
        for b in reversed(self.blocks):  # topmost first
            bx, by, bw, bh = b["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return b
        return None

    def note(self, text, color=FG, now=None):
        self.log.appendleft((now or time.perf_counter(), text, color))

    def press(self, button, x, y, now):
        self.held.add(button)
        self.clicks[button] += 1
        self.ripples.append((x, y, button, now))
        b = self.block_at(x, y)
        if button == "left" and b is not None:
            b["color"] = (b["color"] + 1) % len(PALETTE)
            self.note("left click: new colour", BUTTON_COLOR["left"], now)
        elif button == "right" and b is not None:
            self.blocks.remove(b)
            self.blocks.append(b)  # bring to front
            self.drag = (b, x - b["rect"][0], y - b["rect"][1])
            self.note("right down: dragging block", BUTTON_COLOR["right"], now)
        else:
            self.note(f"{button} click", BUTTON_COLOR[button], now)

    def release(self, button, x, y, now):
        self.held.discard(button)
        if button == "right" and self.drag:
            self.drag = None
            self.note("right up: block dropped", BUTTON_COLOR["right"], now)

    def motion(self, x, y):
        if not self.trail or self.trail[-1] != (x, y):
            self.trail.append((x, y))
        if self.drag:
            b, dx, dy = self.drag
            b["rect"][0] = min(max(x - dx, 0), self.w - b["rect"][2])
            b["rect"][1] = min(max(y - dy, 0), self.h - b["rect"][3])


def run_window(ctrl, procs, sides, dry, check, seconds=0):
    """Main-thread pygame loop. ctrl (run.Controller) is fed by a background thread."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # the first click on the window should already count, not just focus it
    os.environ.setdefault("SDL_MOUSE_FOCUS_CLICKTHROUGH", "1")
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((1280, 800), pygame.RESIZABLE)
    pygame.display.set_caption("MyoMouse playground")
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 16)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 13)
    big = pygame.font.SysFont("Menlo,Consolas,monospace", 22, bold=True)
    clock = pygame.time.Clock()
    pg = Playground(1280 - PANEL_W, 800)
    if dry:
        ctrl.mouse.set_virtual(pg.w / 2, pg.h / 2, pg.w, pg.h)
    status = {s: "connecting..." for s in sides}
    btn_name = {1: "left", 3: "right"}
    t_start = time.perf_counter()

    def text(s, pos, color=FG, f=font):
        screen.blit(f.render(s, True, color), pos)

    running = True
    while running:
        now = time.perf_counter()
        if seconds and now - t_start > seconds:
            break
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                pg.w, pg.h = max(200, ev.w - PANEL_W), ev.h
                if dry:
                    ctrl.mouse.set_virtual(None, None, pg.w, pg.h)
            elif dry:
                continue  # dry run: only the bands drive the (virtual) cursor
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in btn_name and ev.pos[0] < pg.w:
                pg.press(btn_name[ev.button], *ev.pos, now)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button in btn_name:
                pg.release(btn_name[ev.button], *ev.pos, now)
            elif ev.type == pygame.MOUSEMOTION:
                pg.motion(min(ev.pos[0], pg.w - 1), ev.pos[1])

        # what the bands sent (the mouse itself was already moved by the controller thread)
        for kind, a, b in ctrl.drain():
            if kind == "btn":
                name = C.BUTTON_FOR_SIDE[a]
                pg.note(f"{a} fist -> {name.upper()} {'down' if b == 'press' else 'up'}",
                        BUTTON_COLOR[name], now)
                if dry:  # no real mouse: feed the playground directly
                    x, y = ctrl.mouse.vpos
                    (pg.press if b == "press" else pg.release)(name, x, y, now)
            elif kind == "ready":
                status[a] = "ready"
        if dry:
            pg.motion(*ctrl.mouse.vpos)
        for s, p in zip(sides, procs):
            if not p.is_alive():
                status[s] = "stopped (see terminal)"

        # --- draw: blocks, trail, ripples, cursor ring ---
        screen.fill(BG)
        for blk in pg.blocks:
            r = pygame.Rect(blk["rect"])
            pygame.draw.rect(screen, PALETTE[blk["color"]], r, border_radius=10)
            if pg.drag and pg.drag[0] is blk:
                pygame.draw.rect(screen, BUTTON_COLOR["right"], r.inflate(10, 10), 4, border_radius=14)
        if len(pg.trail) > 1:
            pygame.draw.lines(screen, (80, 86, 104), False, list(pg.trail), 2)
        for x, y, button, t in pg.ripples:
            age = now - t
            if age < 0.6:
                pygame.draw.circle(screen, BUTTON_COLOR[button], (x, y), 12 + 80 * age, 3)
        if pg.trail and (dry or pygame.mouse.get_focused()):
            cx, cy = pg.trail[-1]
            for i, button in enumerate(sorted(pg.held)):
                pygame.draw.circle(screen, BUTTON_COLOR[button], (cx, cy), 18 + 6 * i, 4)
            if dry:  # draw our own cursor
                col = BUTTON_COLOR[next(iter(pg.held))] if pg.held else FG
                pygame.draw.circle(screen, col, (cx, cy), 8, 0 if pg.held else 2)
        text("LEFT click a block: change colour     RIGHT button hold: drag a block", (16, 12), DIM)
        if not dry and not pygame.mouse.get_focused():
            text("cursor is outside this window", (16, 36), (230, 80, 80))

        # --- side panel ---
        px = pg.w
        pygame.draw.rect(screen, PANEL, (px, 0, PANEL_W, pg.h))
        y = 16
        text("MyoMouse" + ("  (dry run)" if dry else ""), (px + 16, y), FG, big)
        y += 40
        for button, side in (("left", "right"), ("right", "left")):
            on = button in pg.held
            r = pygame.Rect(px + 16 + (0 if button == "left" else 150), y, 140, 58)
            pygame.draw.rect(screen, BUTTON_COLOR[button] if on else (50, 54, 64), r, border_radius=8)
            text(f"{button.upper()}", (r.x + 12, r.y + 6), FG, big)
            text(f"{side} fist  {pg.clicks[button]}x", (r.x + 12, r.y + 36), FG if on else DIM, small)
        y += 78
        label = "activity" if check else "fist p"
        for s in sides:
            bcol = BUTTON_COLOR[C.BUTTON_FOR_SIDE[s]]
            text(f"{s} band: {status[s]}", (px + 16, y), FG if status[s] == "ready" else DIM)
            y += 22
            pygame.draw.rect(screen, (50, 54, 64), (px + 16, y, 290, 14))
            pygame.draw.rect(screen, bcol, (px + 16, y, 290 * ctrl.probs.get(s, 0.0), 14))
            if not check:
                mx = px + 16 + 290 * C.P_ON
                pygame.draw.line(screen, FG, (mx, y - 3), (mx, y + 17), 1)
            text(label, (px + 16, y + 16), DIM)
            y += 42
        y += 6
        text("events", (px + 16, y), DIM)
        y += 22
        for t, msg, col in pg.log:
            fade = max(0.35, 1 - (now - t) / 8)
            text(msg[:32], (px + 16, y), tuple(int(c * fade) for c in col))
            y += 20
        text("Esc / Q quit", (px + 16, pg.h - 26), DIM)

        pygame.display.flip()
        clock.tick(120)
    pygame.quit()
