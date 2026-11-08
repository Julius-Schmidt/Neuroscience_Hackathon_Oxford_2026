"""Practice arena: move an on-screen cursor with your forearm and hit targets with fists.

The arena draws its own cursor, so the real mouse is never touched: no Accessibility
permission needed and no stray clicks on your desktop. It uses the same band processes
as `myomouse.run`, so what works here works in the live demo.

    python -m myomouse.arena                 # both bands, trained classifiers
    python -m myomouse.arena --sides right   # pointer band only (left-click targets only)
    python -m myomouse.arena --check         # no models yet: cursor only, Space/Enter click
    python -m myomouse.arena --fake          # no hardware

Targets:  blue = left click (right fist),  orange = right click (left fist).
Keys:  +/- gain   X/Y flip direction   C re-centre   R reset score   F fullscreen   Esc quit
       Space / Enter = left / right click from the keyboard
"""

import argparse
import queue
import random
import time
from collections import deque

from . import config as C
from .run import add_pointer_args, apply_pointer_args, start_bands

BG = (18, 20, 26)
FG = (225, 228, 235)
DIM = (120, 126, 140)
BUTTON_COLOR = {"left": (70, 140, 255), "right": (255, 150, 60)}
SIDE_FOR_BUTTON = {b: s for s, b in C.BUTTON_FOR_SIDE.items()}
HIT, MISS = (80, 210, 120), (230, 80, 80)


class Arena:
    def __init__(self, w, h, buttons):
        self.w, self.h = w, h
        self.buttons = buttons  # which buttons targets can ask for
        self.x, self.y = w / 2, h / 2
        self.gain = 1.0
        self.inv_x = self.inv_y = False
        self.held = set()
        self.freeze_until = 0.0
        self.trail = deque(maxlen=40)
        self.flash = None  # (color, until)
        self.reset_score()
        self.new_target()

    def reset_score(self):
        self.hits = self.misses = 0
        self.times = []

    def new_target(self):
        m = 80
        r = random.choice([28, 40, 55])
        # never spawn right under the cursor
        for _ in range(50):
            tx, ty = random.uniform(m, self.w - m), random.uniform(m + 60, self.h - m - 100)
            if (tx - self.x) ** 2 + (ty - self.y) ** 2 > (4 * r) ** 2:
                break
        self.target = (tx, ty, r, random.choice(self.buttons))
        self.t_target = time.perf_counter()

    def resize(self, w, h):
        self.w, self.h = w, h
        self.x, self.y = min(self.x, w - 1), min(self.y, h - 1)
        self.new_target()

    def move(self, dx, dy, now):
        if now < self.freeze_until:
            return
        dx *= self.gain * (-1 if self.inv_x else 1)
        dy *= self.gain * (-1 if self.inv_y else 1)
        self.x = min(max(self.x + dx, 0), self.w - 1)
        self.y = min(max(self.y + dy, 0), self.h - 1)
        self.trail.append((self.x, self.y))

    def button(self, name, down, now):
        self.freeze_until = now + C.FREEZE_MS / 1000
        if not down:
            self.held.discard(name)
            return
        self.held.add(name)
        tx, ty, r, want = self.target
        inside = (self.x - tx) ** 2 + (self.y - ty) ** 2 <= r * r
        if inside and name == want:
            self.hits += 1
            self.times.append(now - self.t_target)
            self.flash = (HIT, now + 0.25)
            self.new_target()
        else:
            self.misses += 1
            self.flash = (MISS, now + 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lda", "threshold"], default="lda")
    ap.add_argument("--sides", default="right,left")
    ap.add_argument("--fake", action="store_true", help="simulated bands")
    ap.add_argument("--check", action="store_true", help="no classifier: cursor only, click with Space/Enter")
    ap.add_argument("--size", default="1280x800")
    add_pointer_args(ap)
    args = ap.parse_args()
    apply_pointer_args(args)

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    if C.POINTER_SIDE not in sides:
        print(f"note: the pointer band ({C.POINTER_SIDE}) isn't running, so the cursor won't move")
    q, procs = start_bands(sides, args.mode, args.fake, True, args.check, args.y_axis)
    status = {s: "connecting..." for s in sides}
    probs = {s: 0.0 for s in sides}

    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame  # imported here so the band processes (spawned) don't load it

    pygame.init()
    w, h = (int(v) for v in args.size.split("x"))
    screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
    pygame.display.set_caption("MyoMouse arena")
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 18)
    big = pygame.font.SysFont("Menlo,Consolas,monospace", 26, bold=True)
    clock = pygame.time.Clock()

    # with --check (or one band) only offer targets for buttons that can actually be clicked
    buttons = ["left", "right"] if args.check else [C.BUTTON_FOR_SIDE[s] for s in sides]
    arena = Arena(w, h, buttons)

    def text(s, pos, color=FG, f=font):
        screen.blit(f.render(s, True, color), pos)

    running = True
    try:
        while running:
            now = time.perf_counter()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    arena.resize(ev.w, ev.h)
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        arena.gain *= 1.15
                    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        arena.gain /= 1.15
                    elif k == pygame.K_x:
                        arena.inv_x = not arena.inv_x
                    elif k == pygame.K_y:
                        arena.inv_y = not arena.inv_y
                    elif k == pygame.K_c:
                        arena.x, arena.y = arena.w / 2, arena.h / 2
                        arena.trail.clear()
                    elif k == pygame.K_r:
                        arena.reset_score()
                        arena.new_target()
                    elif k == pygame.K_f:
                        pygame.display.toggle_fullscreen()
                        arena.resize(*screen.get_size())
                    elif k == pygame.K_SPACE:
                        arena.button("left", True, now)
                    elif k == pygame.K_RETURN:
                        arena.button("right", True, now)
                elif ev.type == pygame.KEYUP:
                    if ev.key == pygame.K_SPACE:
                        arena.button("left", False, now)
                    elif ev.key == pygame.K_RETURN:
                        arena.button("right", False, now)

            # drain everything the bands sent since the last frame
            while True:
                try:
                    kind, a, b, _t = q.get_nowait()
                except queue.Empty:
                    break
                if kind == "move":
                    arena.move(a, b, now)
                elif kind == "btn":
                    arena.button(C.BUTTON_FOR_SIDE[a], b == "press", now)
                elif kind == "p":
                    probs[a] = b
                elif kind == "ready":
                    status[a] = "ready"
            for s, p in zip(sides, procs):
                if not p.is_alive():
                    status[s] = "stopped (see terminal)"

            # --- draw ---
            screen.fill(BG)
            if arena.flash and now < arena.flash[1]:
                pygame.draw.rect(screen, arena.flash[0], screen.get_rect(), 6)

            tx, ty, r, want = arena.target
            col = BUTTON_COLOR[want]
            pygame.draw.circle(screen, col, (tx, ty), r, 4)
            pygame.draw.circle(screen, col, (tx, ty), 5)
            text(f"{want} click ({SIDE_FOR_BUTTON[want]} fist)", (tx - r, ty + r + 6), col)

            if len(arena.trail) > 1:
                pygame.draw.lines(screen, (70, 76, 92), False, list(arena.trail), 2)
            frozen = now < arena.freeze_until
            ccol = DIM if frozen and not arena.held else FG
            for b in arena.held:
                ccol = BUTTON_COLOR[b]
            cx, cy = arena.x, arena.y
            pygame.draw.circle(screen, ccol, (cx, cy), 10, 0 if arena.held else 2)
            pygame.draw.line(screen, ccol, (cx - 18, cy), (cx - 12, cy), 2)
            pygame.draw.line(screen, ccol, (cx + 12, cy), (cx + 18, cy), 2)
            pygame.draw.line(screen, ccol, (cx, cy - 18), (cx, cy - 12), 2)
            pygame.draw.line(screen, ccol, (cx, cy + 12), (cx, cy + 18), 2)

            avg = sum(arena.times) / len(arena.times) if arena.times else 0
            last = arena.times[-1] if arena.times else 0
            text(f"hits {arena.hits}   misses {arena.misses}   avg {avg:.2f} s   last {last:.2f} s",
                 (16, 12), FG, big)
            text(f"gain x{arena.gain:.2f} (= {C.GAIN_X * arena.gain:.0f} px/rad)   "
                 f"flip x: {'on' if arena.inv_x else 'off'}   flip y: {'on' if arena.inv_y else 'off'}",
                 (16, 48), DIM)

            y = arena.h - 30 - 26 * len(sides)
            label = "activity" if args.check else "fist p"
            for s in sides:
                bar_w = 200
                bcol = BUTTON_COLOR[C.BUTTON_FOR_SIDE[s]]
                text(f"{s:5s} {label}", (16, y))
                pygame.draw.rect(screen, (50, 54, 64), (150, y + 3, bar_w, 14))
                pygame.draw.rect(screen, bcol, (150, y + 3, bar_w * probs[s], 14))
                if not args.check:  # press threshold marker
                    px = 150 + bar_w * C.P_ON
                    pygame.draw.line(screen, FG, (px, y), (px, y + 20), 1)
                text(status[s], (150 + bar_w + 12, y), DIM)
                y += 26
            text("+/- gain   X/Y flip   C centre   R reset   F fullscreen   Space/Enter click   Esc quit",
                 (16, arena.h - 26), DIM)

            pygame.display.flip()
            clock.tick(120)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        for p in procs:
            p.terminate()
        print(f"hits {arena.hits}, misses {arena.misses}")
        if arena.gain != 1.0 or arena.inv_x or arena.inv_y:
            print("To keep these settings, put this in myomouse/config.py:")
            if (args.y_axis or C.POINTER_Y_AXIS) == "roll":
                print(f"  GAIN_X = {C.GAIN_X * arena.gain:.0f}\n  GAIN_ROLL = {C.GAIN_ROLL * arena.gain:.0f}")
                print(f"  INVERT_X = {arena.inv_x != C.INVERT_X}\n  INVERT_ROLL = {arena.inv_y != C.INVERT_ROLL}")
            else:
                print(f"  GAIN_X = {C.GAIN_X * arena.gain:.0f}\n  GAIN_Y = {C.GAIN_Y * arena.gain:.0f}")
                print(f"  INVERT_X = {arena.inv_x != C.INVERT_X}\n  INVERT_Y = {arena.inv_y != C.INVERT_Y}")


if __name__ == "__main__":
    main()
