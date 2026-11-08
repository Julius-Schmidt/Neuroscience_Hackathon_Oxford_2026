"""Gesture picking test: two gestures on screen, you do one, the recognised one is marked "Predicted".

    python -m myomouse.gestures11.pick --user julius --side right
    python -m myomouse.gestures11.pick --user julius --side right --target   # it tells you which one
    python -m myomouse.gestures11.pick --user julius --side right --fake     # no hardware

Each round shows two random gestures (photo + name), one left, one right. Do one of them:
when the band recognises it, that card gets a green border and "Predicted" above it, then
the next random pair appears. (Keeping the same gesture held doesn't count again: let go first.) A recognised gesture that isn't one of the two shows in red as "wrong gesture".

  free mode (default): you choose; scored = picked / wrong gesture / nothing within the time limit
  target mode (T):     one card is marked DO THIS; scored = right / other card / wrong gesture / nothing

Keys:  T  free / target mode     + / -  stricter / looser rejection threshold (live)
       Space  skip this pair     R  reset score     F  fullscreen     Esc  quit

The bottom bar shows what the band sees right now (same output as gestures11.live).
Results are saved to data/<user>/gesture_pick_<side>_<timestamp>.json when you quit.
"""

import argparse
import json
import os
import random
import sys
import time

import joblib
import numpy as np

from ..device import open_band
from ..users import add_user_arg, data_dir
from . import GESTURES, IDLE_KEYS, model_path
from .decide import NONE
from .live import GestureClassifier
from .vision import ASSETS

TIMEOUT_S = 8.0   # no gesture recognised within this -> "nothing", next pair
SHOW_S = 0.6      # how long "Predicted" stays on the card before the next pair


class PickGame:
    """Rounds and scoring, independent of the window (testable). Feed it the shown gesture every frame."""

    def __init__(self, gestures, target_mode=False, rng=None):
        if len(gestures) < 2:
            raise ValueError("need at least two gestures")
        self.gestures = list(gestures)
        self.target_mode = target_mode
        self.rng = rng or random.Random()
        self.results = []          # dicts per finished round
        self.pair = None
        self.target = None         # 0 = left, 1 = right (target mode)
        self.state = "start"       # start -> choose -> shown -> choose ...
        self.t_state = 0.0
        self.held = None           # gesture just picked: doesn't count again until it's let go
        self.picked = None         # 0 / 1 while "Predicted" shows
        self.stray = None          # (gesture, time) of the last wrong gesture
        self.last_pair = None

    def new_pair(self, now):
        while True:
            pair = self.rng.sample(self.gestures, 2)
            if set(pair) != set(self.last_pair or ()):
                break
        self.pair, self.last_pair = pair, pair
        self.target = self.rng.randrange(2) if self.target_mode else None
        self.state, self.t_state, self.picked, self.stray = "choose", now, None, None

    def skip(self, now):
        self.new_pair(now)

    def update(self, now, shown):
        """shown: the live output (a gesture key or "neutral")."""
        if self.held is not None and shown != self.held:
            self.held = None
        if self.state == "start":
            self.new_pair(now)
        elif self.state == "choose":
            if shown == self.held:
                pass                                 # still the last pick: wait until it's let go
            elif shown in self.pair:
                self._finish(now, self.pair.index(shown))
            elif shown != NONE and shown not in IDLE_KEYS:
                if self.stray is None or self.stray[0] != shown:
                    self.stray = (shown, now)
                    self.results.append(self._result(now, "wrong gesture", shown))
            elif now - self.t_state > TIMEOUT_S:
                self.results.append(self._result(now, "nothing", None))
                self.new_pair(now)
        elif self.state == "shown" and now - self.t_state > SHOW_S:
            self.new_pair(now)

    def _finish(self, now, side):
        outcome = ("right" if side == self.target else "other card") if self.target_mode else "picked"
        self.results.append(self._result(now, outcome, self.pair[side]))
        self.picked, self.held = side, self.pair[side]
        self.state, self.t_state = "shown", now

    def _result(self, now, outcome, got):
        return {"pair": list(self.pair), "target": None if self.target is None else self.pair[self.target],
                "outcome": outcome, "got": got, "seconds": round(now - self.t_state, 3),
                "mode": "target" if self.target_mode else "free"}

    def score(self):
        """{outcome: count} and the median time to a pick (s)."""
        counts = {}
        for r in self.results:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        times = [r["seconds"] for r in self.results if r["outcome"] in ("picked", "right", "other card")]
        return counts, (float(np.median(times)) if times else None)

    def reset_score(self):
        self.results = []


# --------------------------------------------------------------------------- window

BG, CARD, FG, DIM = (18, 20, 26), (32, 36, 46), (225, 228, 235), (120, 126, 140)
BLUE, GREEN, RED, YELLOW = (70, 140, 255), (60, 200, 110), (235, 80, 70), (245, 200, 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_user_arg(ap)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--target", action="store_true", help="start in target mode (it tells you which one)")
    ap.add_argument("--reject", type=float, default=None, help="rejection threshold (default: what train saved)")
    ap.add_argument("--fake", action="store_true", help="simulated band, for testing")
    ap.add_argument("--seconds", type=float, default=0, help=argparse.SUPPRESS)  # auto-quit, for tests
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)      # save the last frame, for tests
    args = ap.parse_args(argv)

    p = model_path(args.user, args.side)
    if not p.exists():
        raise SystemExit(f"No gesture model at {p}. Run gestures11.record + gestures11.train first.")
    model = joblib.load(p)
    clf = GestureClassifier(model, args.reject)
    gestures = [k for k in clf.classes if k not in IDLE_KEYS]
    game = PickGame(gestures, target_mode=args.target)
    names = {g.key: g.name for g in GESTURES}
    how = {g.key: g.how for g in GESTURES}

    band = open_band(args.side, fake=args.fake)
    band.add_emg_handler(clf.push)
    band.start()

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((1280, 800), pygame.RESIZABLE)
    pygame.display.set_caption("Gesture pick")
    f_big = pygame.font.SysFont("Helvetica,Arial", 40, bold=True)
    f_mid = pygame.font.SysFont("Helvetica,Arial", 24, bold=True)
    f_small = pygame.font.SysFont("Helvetica,Arial", 18)
    photos = {}
    for k in gestures:
        f = ASSETS / f"{k}.png"
        photos[k] = pygame.image.load(str(f)).convert() if f.exists() else None
    clock = pygame.time.Clock()
    t_start = time.perf_counter()

    def text(s, font, color, center=None, topleft=None):
        img = font.render(s, True, color)
        r = img.get_rect(center=center) if center else img.get_rect(topleft=topleft)
        screen.blit(img, r)
        return r

    def card(rect, key, side, now):
        predicted = game.state == "shown" and game.picked == side
        is_target = game.target_mode and game.target == side
        pygame.draw.rect(screen, CARD, rect, border_radius=18)
        if predicted:
            pygame.draw.rect(screen, GREEN, rect, width=6, border_radius=18)
            text("Predicted", f_mid, GREEN, center=(rect.centerx, rect.top - 18))
        elif is_target and game.state == "choose":
            pygame.draw.rect(screen, YELLOW, rect, width=6, border_radius=18)
        else:
            pygame.draw.rect(screen, DIM, rect, width=2, border_radius=18)
        if is_target and game.state == "choose":
            text("DO THIS", f_mid, YELLOW, center=(rect.centerx, rect.top + 28))
        text(names[key], f_big, FG, center=(rect.centerx, rect.top + 75))
        ph = photos.get(key)
        if ph is not None:
            maxw, maxh = rect.w - 60, rect.h - 190
            s = min(maxw / ph.get_width(), maxh / ph.get_height())
            img = pygame.transform.smoothscale(ph, (int(ph.get_width() * s), int(ph.get_height() * s)))
            screen.blit(img, img.get_rect(center=(rect.centerx, rect.top + 105 + maxh / 2)))
        text(how[key], f_small, DIM, center=(rect.centerx, rect.bottom - 30))

    running, full = True, False
    try:
        while running:
            now = time.perf_counter()
            if args.seconds and now - t_start > args.seconds:
                break
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif ev.key == pygame.K_t:
                        game.target_mode = not game.target_mode
                        game.skip(now)
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        clf.reject = min(0.99, round(clf.reject + 0.05, 2))
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        clf.reject = max(0.0, round(clf.reject - 0.05, 2))
                    elif ev.key == pygame.K_SPACE:
                        game.skip(now)
                    elif ev.key == pygame.K_r:
                        game.reset_score()
                    elif ev.key == pygame.K_f:
                        full = not full
                        screen = pygame.display.set_mode((0, 0) if full else (1280, 800),
                                                         pygame.FULLSCREEN if full else pygame.RESIZABLE)
            if not band.alive:
                raise SystemExit(f"band disconnected: {band.error}")
            shown = clf.current
            game.update(now, shown)

            w, h = screen.get_size()
            screen.fill(BG)
            # score line
            counts, med = game.score()
            mode = "TARGET mode: do the marked one" if game.target_mode else "FREE mode: do either one"
            parts = [f"{k} {v}" for k, v in counts.items()] or ["no rounds yet"]
            if med is not None:
                parts.append(f"median {med:.1f} s")
            text(mode, f_mid, FG, topleft=(30, 20))
            text("   ".join(parts), f_small, DIM, topleft=(30, 55))
            text(f"reject < {clf.reject:.2f}  (+/-)   T mode   Space skip   R reset   Esc quit",
                 f_small, DIM, topleft=(w - 560, 25))

            # the two cards
            if game.pair:
                top = 125  # room for "Predicted" above the cards
                cw, ch, gap = (w - 90) // 2, h - 290, 30
                for side in (0, 1):
                    card(pygame.Rect(30 + side * (cw + gap), top, cw, ch), game.pair[side], side, now)
                if game.state == "choose":
                    left = TIMEOUT_S - (now - game.t_state)
                    text(f"{left:3.1f} s", f_small, DIM, center=(w // 2, top + ch + 18))
            if game.stray and game.state == "choose" and now - game.stray[1] < 1.5:
                text(f"wrong gesture: {names[game.stray[0]]}", f_mid, RED, center=(w // 2, h - 120))

            # what the band sees right now
            bar = pygame.Rect(0, h - 80, w, 80)
            pygame.draw.rect(screen, CARD, bar)
            cur = names.get(shown, shown)
            text(f"band sees: {cur}", f_mid, GREEN if shown != NONE else FG, topleft=(30, h - 65))
            if clf.proba is not None:
                i = int(np.argmax(clf.proba))
                guess = f"guess {names[clf.classes[i]]} {clf.proba[i]:.0%}" + (f" ({clf.why})" if clf.why else "")
                text(f"level {clf.env:.2f}    {guess}", f_small, DIM, topleft=(30, h - 32))
                bw = int(300 * float(clf.proba[i]))
                pygame.draw.rect(screen, DIM, (w - 340, h - 50, 300, 16), width=1)
                pygame.draw.rect(screen, BLUE, (w - 340, h - 50, bw, 16))
                x_rej = w - 340 + int(300 * clf.reject)
                pygame.draw.line(screen, YELLOW, (x_rej, h - 56), (x_rej, h - 28), 2)
            pygame.display.flip()
            clock.tick(60)
    finally:
        if args.screenshot:
            pygame.image.save(screen, args.screenshot)
        band.stop()
        pygame.quit()

    counts, med = game.score()
    print("results:", counts or "no rounds", f"median {med:.2f} s" if med is not None else "")
    if game.results:
        out = data_dir(args.user) / f"gesture_pick_{args.side}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"user": args.user, "side": args.side, "reject": clf.reject,
                                   "rounds": game.results}, indent=1))
        print(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
