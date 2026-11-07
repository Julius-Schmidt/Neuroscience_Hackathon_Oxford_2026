"""
Neuratect — main runner.
cv/ → mapping-engine/ → os-integration/

Input modes:
  tap  — fires once per gesture with cooldown, repeats every TAP_COOLDOWN while held
  held — key/button held while gesture active, released when gesture stops
  joystick — wrist position drives WASD (left hand) or mouse look (right hand)
  pinch    — thumb-to-fingertip distance holds the mouse buttons (left/right),
             in every context; see PinchClicker

Threading model:
  detection thread — camera read + gesture classification (~30 fps, camera-limited)
  mouse thread     — applies latest velocity at MOUSE_HZ with sub-pixel accumulation
  main thread      — held-key reconciliation, taps, context tracking (camera-rate is fine here)
"""

import sys
import os
import math
import time
import json
import atexit
import logging
import threading
import contextlib
from collections import deque
import AppKit
import Quartz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'cv'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'mapping-engine'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'os-integration'))

from detector import GestureDetector
from mapper import map_gesture
from context import get_active_context
from executor import tap, hold, release, move_mouse_relative, post_cursor_move
from commands import Command, HoldW, HoldA, HoldS, HoldD, HoldBreak, HoldPlace, Sprint
from joystick import HandJoystick, MAX_RANGE
from filters import OneEuroFilter
from keypress import KeyListener

# Every state transition (gesture entry/exit, click engage/release, drag-ready,
# WASD press/release, anchor calibrate/release, context/GUI changes) is logged
# with a timestamp here and mirrored to logs/session.log. Feel is hard to
# describe in words but easy to read off a timestamped event timeline — run a
# test sequence once, then the log gets read back directly instead of guessing
# from a description of how something felt.
_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "session.log")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger("neuratect")

# Persisted tunables — currently just gui_sensitivity, set live via calibrate.py
# and picked up here on the next run.
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "settings.json")


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(settings: dict) -> None:
    os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


TAP_COOLDOWN       = 1.0     # default repeat interval for held tap gestures
# Per-command overrides. This is both the gap between two isolated taps and the
# repeat rate while the gesture is held — one number covers both, since the
# check is "how long since this command last fired". Hotbar cycling is the one
# case where you routinely want several in a row (walking a selection along the
# bar), so it runs at half the default; Inventory/Escape/OpenSpotlight
# deliberately do NOT, because they toggle a screen and a fast repeat fights
# itself.
_TAP_COOLDOWN      = {"HotbarNext": 0.5, "HotbarPrev": 0.5}
SPOTLIGHT_DWELL    = 0.4     # OpenSpotlight only fires after the gesture is held this long —
                             # transient recognition flicker while the hand does something
                             # else repeatedly opened Spotlight (multiple sessions' logs)
MOUSE_HZ           = 60      # mouse event rate — match Minecraft's render rate
MOUSE_SENSITIVITY  = 1800    # pixels/sec at raw offset = 1.0; tune this for speed

# Right-hand gestures that drive the mouse cursor during *gameplay* (camera
# look). Camera look is a rate control, so it needs an explicit engage pose
# that also anchors the stick where the hand currently is — a closed fist.
# "pinch" is included as a fist-equivalent: the pinch template is visually
# close to fist, so classification flickers between them — without this the
# camera look (and WASD, below) stuttered on every flicker frame. It also
# means a thumb pinch (= a click, see PinchClicker) never drops camera look,
# so mining while looking around is one continuous motion.
# index_pointer/peace used to be here as click-and-drag poses; clicks are
# pinch-based everywhere now, so those poses (and the drag-arm delay that
# existed to keep a pose click from jitter-dragging) are gone.
_MOUSE_GESTURES  = {"fist", "pinch"}

# GUI cursor: instead of the velocity/joystick model used for camera-look
# (a rate control — can't give clean precision pointing, since you have to
# return to an exact center to stop and any offset keeps drifting), this
# moves the cursor by the *delta* between consecutive hand positions, like a
# mouse's relative displacement — it stops the instant the hand stops, no
# drift. Depth is cancelled by dividing the delta by the live hand size
# (Gesture.hand_scale) — "how far did the hand move in hand-widths" is the
# same number regardless of distance from the camera. See GuiPointer for the
# interaction model (One Euro filtering, pinch clicks, clutch, click rewind).
# GUI_SENSITIVITY is the main tunable knob (window-heights of cursor travel
# per hand-width of physical movement, at rest — the accel curve multiplies
# it with speed). Default here; calibrate.py lets you retype it live while
# playing and persists the last value you set to data/settings.json, which
# this default gets overridden by on load. pinch_engage / pinch_release can
# also be overridden from data/settings.json (no code edit needed to tune).
GUI_POINTER_SMOOTH = 0.5    # EMA alpha on hand_scale (position uses One Euro, below)
# Sizing sanity check for GUI_SENSITIVITY (2026-08-08): one inventory slot is
# ~4% of window height; at sensitivity S a slot corresponds to 0.04/S
# hand-widths of movement — at S=1.5 that was ~1 camera pixel (impossible to
# hit; live log showed clicks slammed into window corners from overshoot).
# S=0.6 puts a slot at ~2.5 hand-hundredths ≈ several camera px, and the
# clutch (open palm) covers the lost reach.
GUI_SENSITIVITY    = 0.6
GUI_MIN_CUTOFF     = 0.5    # One Euro low-speed cutoff (Hz) — lower = steadier hover.
                            # Picked by synthetic sweep (±1px landmark noise @320px):
                            # 0.5/5.0 ≈ 3px hover jitter on a 1080p window at ~0.7
                            # frames of lag during fast moves.
GUI_BETA           = 5.0    # One Euro cutoff growth with speed — higher = less lag when moving fast
GUI_ACCEL          = 0.15   # gain multiplier growth per hand-width/sec of hand speed
GUI_ACCEL_MAX      = 2.5    # cap on that multiplier (accel gives reach; base gain gives precision)
# Pinch thresholds, tuned live rather than derived: 0.26/0.40 (from the
# 2026-08-07 log) held clicks down after the fingers separated → halved to
# 0.13/0.20, which was too tight to engage comfortably → +25% to the values
# below. Engage is easier to reach than at 0.13 and release still fires on a
# partly-opened hand, well short of the old 0.40.
# Still worth knowing: release sits inside the range a deliberate pinch reads,
# so a pinch that loosens mid-drag can drop the button. If either direction
# misbehaves, re-measure rather than guess — `python3 calibrate.py` logs the
# live ratios at ~1 Hz in every mode; pinch, relax and drag while watching
# them, then put the two numbers in data/settings.json (pinch_engage /
# pinch_release, no code edit). Both must stay inside the gap between what a
# real pinch reads and what a relaxed hand reads, release above engage.
PINCH_ENGAGE       = 0.16   # smoothed tip distance / hand_scale below this = mouse down
PINCH_RELEASE      = 0.25   # above this = mouse up (relaxing the hand is enough — by design)
PINCH_SMOOTH       = 0.5    # EMA alpha on the ratio before thresholding (raw jitters ±0.1)
PINCH_COOLDOWN     = 0.15   # min seconds between a pinch release and the next engage
CLICK_REWIND       = 0.12   # clicks land at the cursor position this many seconds ago
CLICK_FREEZE       = 0.15   # pointer frozen this long around each click edge
TRAIL_KEEP         = 0.6    # seconds of cursor history kept for CLICK_REWIND
PINCH_HAND_LOST_RELEASE = 1.0 # force pinch mouse-up after the hand is unseen this long

# ── Shared state (detection thread → mouse thread) ──────────────────────────
_vel_lock     = threading.Lock()
_mouse_vel    = [0.0, 0.0]    # px/sec velocity; zeroed when hand not active
_mouse_mode   = ["velocity"]  # "velocity" (camera-look) | "absolute" (GUI pointer)
_mouse_target = [0.5, 0.5]    # fraction of window (0..1); used when mode == "absolute"
# Which mouse button is currently held down: None | "left" | "right". Cursor
# motion has to be posted as a drag event while a button is down (see
# executor.post_cursor_move), and the mouse thread can't see _held_commands
# safely, so the main loop publishes it here after each reconciliation.
_mouse_button = [None]

# Context shared from detection → mouse thread (for Minecraft clamping)
_ctx_lock   = threading.Lock()
_shared_ctx = ["default"]

_running = True

# ── Held keys (main/detection thread only) ──────────────────────────────────
_held_commands: set[Command] = set()


def _release_all() -> None:
    for cmd in list(_held_commands):
        release(cmd)
    _held_commands.clear()


# Hysteresis on the analogue→digital WASD conversion: jitter sitting right at
# a single threshold causes key chatter (press/release every frame). Needs a
# higher bar to engage than to release, like a Schmitt trigger.
_WASD_ENGAGE  = 0.15
_WASD_RELEASE = 0.05
_wasd_state = {"w": False, "a": False, "s": False, "d": False}

# Sprint engages when the left hand is pushed forward *past* the stick's full
# deflection (MAX_RANGE, where WASD output has already saturated — travel past
# it is otherwise unused), with hysteresis so sitting at the boundary doesn't
# chatter. Raise SPRINT_ENGAGE if sprint triggers during normal walking.
SPRINT_ENGAGE  = MAX_RANGE          # forward offset from the anchor, normalized frame units
SPRINT_RELEASE = MAX_RANGE * 0.8


def _held(active: bool, value: float) -> bool:
    return value > _WASD_RELEASE if active else value > _WASD_ENGAGE


def _wasd_from_joystick(vec) -> set[Command]:
    _wasd_state["w"] = _held(_wasd_state["w"], -vec.y)
    _wasd_state["s"] = _held(_wasd_state["s"],  vec.y)
    _wasd_state["a"] = _held(_wasd_state["a"], -vec.x)
    _wasd_state["d"] = _held(_wasd_state["d"],  vec.x)

    keys = set()
    if _wasd_state["w"]: keys.add(HoldW)
    if _wasd_state["s"]: keys.add(HoldS)
    if _wasd_state["a"]: keys.add(HoldA)
    if _wasd_state["d"]: keys.add(HoldD)
    return keys


def _reset_wasd_state() -> None:
    for k in _wasd_state:
        _wasd_state[k] = False


# ── GUI-open tracking (inventory / pause menu / chat) ────────────────────────
# Ground truth, not bookkeeping: Minecraft hides the OS cursor while the
# mouse is captured for gameplay and shows it whenever any screen (inventory,
# pause menu, chat) is open — CGCursorIsVisible reads that directly. This
# replaced counting our own Inventory/Escape keypresses, which silently
# desynced whenever the game itself changed screens (death screen, a real
# keyboard press, a misrecognized gesture). CGCursorIsVisible is deprecated
# (macOS 10.9) but present and correct — verified live on this machine; if a
# macOS update ever removes it, fall back to keypress-counting (git history
# has the old implementation).
_gui_open = False


def _sync_gui_open() -> None:
    global _gui_open
    visible = bool(Quartz.CGCursorIsVisible())
    if visible != _gui_open:
        _gui_open = visible
        log.info(f"[gui] {'open' if visible else 'closed'} (cursor visibility)")


# ── Minecraft window bounds (cached — CGWindowList is expensive at 120 Hz) ──
_mc_bounds_cache: tuple | None = None
_mc_bounds_ts    = 0.0
_MC_BOUNDS_TTL   = 2.0   # seconds between re-queries


def _get_mc_bounds() -> tuple[float, float, float, float] | None:
    global _mc_bounds_cache, _mc_bounds_ts
    now = time.monotonic()
    if now - _mc_bounds_ts > _MC_BOUNDS_TTL:
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        # Fullscreen Minecraft has *two* layer-0 "java" windows: the real game
        # canvas and a thin menu-bar-reveal strip. Taking the first match (as
        # z-order would suggest) can lock onto the strip instead — pick the
        # largest by area so the actual canvas always wins, windowed or not.
        best = None
        best_area = 0.0
        for w in windows:
            owner = (w.get("kCGWindowOwnerName") or "").lower()
            if "java" in owner or "minecraft" in owner:
                b = w.get("kCGWindowBounds")
                if b:
                    area = b["Width"] * b["Height"]
                    if area > best_area:
                        best = (b["X"], b["Y"], b["Width"], b["Height"])
                        best_area = area
        _mc_bounds_cache = best
        _mc_bounds_ts = now
    return _mc_bounds_cache


# kCGWindowBounds is the FULL window frame: it includes the title bar, and
# macOS's invisible resize zones extend several px inside every edge. A
# synthetic mouse-down within those regions grabs window chrome — a held
# click gesture then *drags the window edge* instead of playing (observed
# live 2026-08-08: cursor clamped to the bottom edge + HoldBreak = window
# resized to zero height, java frozen mid-live-resize). Every synthetic
# cursor position must stay inside this inset content rect, never the raw
# bounds. Insets are also applied in fullscreen (no chrome there) — harmless,
# nothing interactive lives in the outer 30px.
_TITLE_INSET = 30.0   # title bar height
_EDGE_INSET  = 12.0   # clearance from the resize zones on all edges


def _content_bounds() -> tuple[float, float, float, float] | None:
    """Minecraft window bounds inset to the safe content area (see above)."""
    bounds = _get_mc_bounds()
    if not bounds:
        return None
    wx, wy, ww, wh = bounds
    cw = ww - 2 * _EDGE_INSET
    ch = wh - _TITLE_INSET - _EDGE_INSET
    if cw <= 0 or ch <= 0:
        return None
    return wx + _EDGE_INSET, wy + _TITLE_INSET, cw, ch


def _screen_bounds() -> tuple[float, float, float, float]:
    """Main-display bounds — the desktop pointer's coordinate space. No
    insets: unlike a window, every screen pixel (menu bar, Dock) is a
    legitimate mouse target."""
    r = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return r.origin.x, r.origin.y, r.size.width, r.size.height


def _pointer_bounds(ctx: str) -> tuple[float, float, float, float] | None:
    """Coordinate space for absolute-pointer fractions in a given context.
    Minecraft is the one context with its own space (its window's content
    rect); every other context — default desktop, Spotlight, anything added
    later — is a normal app being pointed at on the full screen."""
    if ctx == "minecraft":
        return _content_bounds()
    return _screen_bounds()


def _pointer_mode(ctx: str, gui_open: bool) -> str | None:
    """
    Which cursor model is in charge: an absolute pointer (GuiPointer, returning
    the coordinate space's name) or None for velocity camera-look.

    The absolute pointer is the default — every context gets it, in one of two
    coordinate spaces: Minecraft GUI screens (window content rect) or the full
    main display (desktop, Spotlight, any other app). Spotlight used to fall
    through to the velocity joystick, which meant its results list could only
    be reached via the left hand's arrow keys while the right hand did
    rate-control mouse-look over a search panel; it's a normal pointable UI, so
    it gets the normal pointer. Gameplay camera look — Minecraft with no screen
    open — is the one thing left on the velocity model, since aiming a camera
    *is* a rate control.

    Desktop and Spotlight deliberately share the "desktop" mode name: moving
    between them is not a coordinate-space change, so the pointer keeps its
    position instead of re-seeding from the cursor.
    """
    if ctx == "minecraft":
        return "mc-gui" if gui_open else None
    return "desktop"


def _ensure_cursor_in_bounds() -> None:
    """
    Cursor *movement* is clamped to the Minecraft window, but a click event
    (HoldBreak/HoldPlace mouse-down) posts at whatever the real OS cursor
    position happens to be, unclamped. If it had drifted outside the window
    (stale cached bounds, a resize, or just sitting still with no movement
    event to re-clamp it), the click lands on
    the desktop/another window instead of Minecraft — which steals OS focus,
    and a still-held gesture then gets reinterpreted under whatever context
    that other app maps to (e.g. index_pointer → OpenSpotlight outside
    Minecraft). Call this right before any click-down in Minecraft context.
    """
    bounds = _content_bounds()
    if not bounds:
        return
    cx, cy, cw, ch = bounds
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    cur = Quartz.CGEventGetLocation(Quartz.CGEventCreate(src))
    if cx <= cur.x <= cx + cw and cy <= cur.y <= cy + ch:
        return   # already inside — nothing to do
    nx = min(max(cur.x, cx), cx + cw)
    ny = min(max(cur.y, cy), cy + ch)
    # Plain move, not a drag: this runs *before* the mouse-down it is preparing
    # for, so no button is held yet (and only one can ever be).
    evt = Quartz.CGEventCreateMouseEvent(
        src, Quartz.kCGEventMouseMoved, Quartz.CGPoint(nx, ny), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    log.info(f"[mouse] cursor was outside the content area ({cur.x:.0f},{cur.y:.0f}) — snapped back in bounds")


def _center_cursor_in_window() -> None:
    """Snap the cursor to the middle of the content area before a *gameplay*
    click. The game aims with the crosshair, not the cursor, so position is
    free — and center is the one spot guaranteed clear of window chrome even
    if the cached bounds are a resize stale.

    Plain move, not a drag, for the same reason as _ensure_cursor_in_bounds: it
    runs before the mouse-down it prepares for."""
    bounds = _content_bounds()
    if not bounds:
        return
    cx, cy, cw, ch = bounds
    post_cursor_move(cx + cw / 2, cy + ch / 2, 0, 0)


def _warp_cursor_to_fraction(fx: float, fy: float, bounds: tuple | None) -> None:
    """Move the OS cursor to a fraction of `bounds` *now*, synchronously —
    used for click rewind, where the mouse-down/up that follows must post at
    this position rather than wherever the 60 Hz mouse thread has gotten to.

    Uses the drag event type when a button is already down, which is the case
    on the *release* edge (the button is released after this warp) — posting a
    plain move there would break the drag one event before it ends."""
    if not bounds:
        return
    cx, cy, cw, ch = bounds
    tx = cx + fx * cw
    ty = cy + fy * ch
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    cur = Quartz.CGEventGetLocation(Quartz.CGEventCreate(src))
    with _vel_lock:
        button = _mouse_button[0]
    post_cursor_move(tx, ty, tx - cur.x, ty - cur.y, button)


# ── Pinch clicks ─────────────────────────────────────────────────────────────
class PinchClicker:
    """
    Thumb-distance click state machine — the click mechanism *everywhere*:
    absolute pointer (desktop, Spotlight, Minecraft GUI screens) and gameplay
    camera-look alike.

    It reports which fingers are pinched; the mapper decides what that means
    for the hand and context (right hand → mouse buttons, left hand in
    Minecraft → Jump / Crouch).

    Why pinch instead of the pose-template click gestures (index_pointer /
    peace) it replaced: forming those poses translates landmark 9 (the tracked
    anchor) and pitches the hand — which shrinks hand_scale, spiking the
    1/scale gain — at exactly the moment of clicking, so every click dragged
    the cursor off its own target. Pose clicks also flicker with the
    classifier, which is what the old 1.5s drag-arm delay existed to paper
    over. A fingertip pinch barely moves the anchor, is measured as a
    continuous distance with hysteresis (no classifier in the loop), and can
    be held while a fist drives camera look.
    """

    def __init__(self, engage: float, release: float, exclusive: bool = True,
                 fingers: tuple[str, ...] = ("index", "middle")):
        self._engage = engage
        self._release = release
        self._exclusive = exclusive
        self._fingers = fingers
        self._released_at = {f: 0.0 for f in fingers}
        self._missing_since: float | None = None
        self.reset()

    def reset(self) -> None:
        self._held = {f: False for f in self._fingers}
        self._smoothed: dict[str, float | None] = {f: None for f in self._fingers}
        self._missing_since = None

    def update(self, g, t: float) -> list[tuple[str, str, float]]:
        """Hysteresis state machine on the pinch distances. Returns click-edge
        events as (finger, "down"|"up", ratio). What a finger *means* is not
        decided here — the caller resolves it through the mapper, so the same
        mechanism serves mouse buttons on the right hand and Jump/Crouch on
        the left."""
        self._missing_since = None
        events = []
        ratios = {"index": g.pinch_index, "middle": g.pinch_middle,
                  "ring": g.pinch_ring, "pinky": g.pinch_pinky}
        for finger, raw in ((f, ratios[f]) for f in self._fingers):
            # Smooth before thresholding — the raw ratio jitters ±0.1 per
            # frame, which is the size of the entire hysteresis band.
            prev = self._smoothed[finger]
            ratio = raw if prev is None else PINCH_SMOOTH * raw + (1.0 - PINCH_SMOOTH) * prev
            self._smoothed[finger] = ratio
            if not self._held[finger]:
                if (ratio < self._engage
                        and not self._exclusive_block(finger)
                        and t - self._released_at[finger] >= PINCH_COOLDOWN):
                    self._held[finger] = True
                    events.append((finger, "down", ratio))
            elif ratio > self._release:
                self._held[finger] = False
                self._released_at[finger] = t
                events.append((finger, "up", ratio))
        return events

    def _exclusive_block(self, finger: str) -> bool:
        """One at a time, or independent? Mouse buttons must be exclusive (a
        left+right click at once is not a thing); the left hand's pinches are
        independent — jumping while crouched is normal play."""
        if not self._exclusive:
            return False
        return any(self._held[f] for f in self._fingers if f != finger)

    def hand_lost(self, t: float) -> list[tuple[str, str, float]]:
        """No hand this frame: force-release a held pinch only after a
        timeout, so a one-frame dropout doesn't break a drag but a real
        disappearance can't leave a mouse button stuck down."""
        if self._missing_since is None:
            self._missing_since = t
            self._smoothed = {f: None for f in self._fingers}
        events = []
        if t - self._missing_since > PINCH_HAND_LOST_RELEASE:
            for finger in self._fingers:
                if self._held[finger]:
                    self._held[finger] = False
                    self._released_at[finger] = t
                    events.append((finger, "up", -1.0))
        return events

    def held_fingers(self) -> set[str]:
        return {f for f, held in self._held.items() if held}


# ── GUI pointer ──────────────────────────────────────────────────────────────
class GuiPointer:
    """
    Absolute-pointer cursor: One Euro-filtered, depth-normalized relative
    pointer with PinchClicker clicks. Runs on the desktop, in Spotlight, and
    on Minecraft GUI screens.

    Residual motion from forming the pinch is handled by clicking at the
    cursor position CLICK_REWIND in the past (position trail) and freezing the
    pointer for CLICK_FREEZE around each click edge.

    Interaction model:
      any tracked pose  — cursor follows the hand (relative delta × gain)
      thumb–index pinch — left mouse held (HoldBreak); keep moving to drag
      thumb–middle pinch— right mouse held (HoldPlace)
      open_palm         — clutch: "mouse lifted", reposition the hand freely
      hand not detected — cursor holds position (a dropout is not a mode change)
    """

    def __init__(self, sensitivity: list, pinch_engage: float, pinch_release: float):
        self._sens = sensitivity   # boxed [float] — shared with calibrate's input thread
        self._fx = OneEuroFilter(min_cutoff=GUI_MIN_CUTOFF, beta=GUI_BETA)
        self._fy = OneEuroFilter(min_cutoff=GUI_MIN_CUTOFF, beta=GUI_BETA)
        self._scale: float | None = None
        self._prev: tuple[float, float] | None = None
        self._prev_t: float | None = None
        self._trail: deque[tuple[float, float, float]] = deque()   # (t, fx, fy)
        self._pinch = PinchClicker(pinch_engage, pinch_release)
        self._hand_missing_since: float | None = None
        self.frozen_until = 0.0

    def reset(self) -> None:
        self.reset_tracking()
        self._trail.clear()
        self._pinch.reset()
        self._hand_missing_since = None
        self.frozen_until = 0.0

    def reset_tracking(self) -> None:
        """Drop position continuity (clutch lift / hand lost) without touching
        pinch state — re-entry re-seeds instead of producing a delta jump."""
        self._fx.reset()
        self._fy.reset()
        self._prev = None
        self._prev_t = None

    def track(self, g, t: float) -> tuple[float, float] | None:
        """
        Feed one frame of the tracked hand. Returns the pointer delta in
        window-height fractions for BOTH axes (isotropic — caller converts x
        by wh/ww), or None on a seed frame.
        """
        self._hand_missing_since = None
        x = g.wrist_x * g.aspect   # aspect-correct: both axes in frame-height units
        y = g.wrist_y
        fx = self._fx.filter(x, t)
        fy = self._fy.filter(y, t)
        if self._scale is None:
            self._scale = g.hand_scale
        else:
            self._scale = (GUI_POINTER_SMOOTH * g.hand_scale
                           + (1.0 - GUI_POINTER_SMOOTH) * self._scale)
        if self._prev is None:
            self._prev, self._prev_t = (fx, fy), t
            return None
        dt = max(t - self._prev_t, 1e-4)
        dx = -(fx - self._prev[0])   # flip x: camera view is mirrored vs. the user
        dy =   fy - self._prev[1]
        self._prev, self._prev_t = (fx, fy), t
        s = max(self._scale, 1e-4)
        hx, hy = dx / s, dy / s      # hand-widths moved this frame
        speed = math.hypot(hx, hy) / dt
        gain = self._sens[0] * min(GUI_ACCEL_MAX, 1.0 + GUI_ACCEL * speed)
        return hx * gain, hy * gain

    def record(self, t: float, fx: float, fy: float) -> None:
        self._trail.append((t, fx, fy))
        while self._trail and t - self._trail[0][0] > TRAIL_KEEP:
            self._trail.popleft()

    def rewind(self, t_cut: float) -> tuple[float, float] | None:
        """Newest trail position at or before t_cut (or the oldest we have)."""
        pos = None
        for tt, fx, fy in self._trail:
            if tt > t_cut:
                if pos is None:
                    pos = (fx, fy)   # everything is newer — best we can do is the oldest
                break
            pos = (fx, fy)
        return pos

    def update_pinch(self, g, t: float) -> list[tuple[str, str, float]]:
        self._hand_missing_since = None
        return self._pinch.update(g, t)

    def hand_lost(self, t: float) -> list[tuple[str, str, float]]:
        """No right hand this frame: hold cursor position (a one-frame dropout
        is not a mode change) and let the clicker decide about releasing."""
        if self._hand_missing_since is None:
            self._hand_missing_since = t
            self.reset_tracking()
        return self._pinch.hand_lost(t)

    def held_fingers(self) -> set[str]:
        return self._pinch.held_fingers()


# Commands blocked while a Minecraft GUI screen is open, the same reasoning as
# the WASD gate — a key that only means something during gameplay should not
# leak into a screen:
#   Jump         — space activates a focused button in the pause menu, and
#                  types a space in chat
#   Sprint       — meaningless with a screen up
#   Hotbar*      — the number keys 1-9 are NOT inert in a container: they swap
#                  whatever slot you're hovering into that hotbar slot, so a
#                  stray hotbar cycle silently rearranges your inventory
# Deliberately NOT blocked: Crouch (held shift + a right-hand pinch click IS
# inventory shift-click, so it has to work exactly when a screen is open),
# Inventory and Escape (they're how you close the screen), mouse buttons.
_GUI_BLOCKED_COMMANDS = {"Jump", "Sprint", "HotbarNext", "HotbarPrev"}


def _blocked_by_gui(cmd: Command, context: str, gui_open: bool) -> bool:
    """gui_open is only tracked (and only meaningful) in Minecraft — it goes
    stale the moment another app is focused, so the context check is load-
    bearing, not defensive."""
    return context == "minecraft" and gui_open and cmd.name in _GUI_BLOCKED_COMMANDS


def _pinch_actions(hand: str, context: str, gui_open: bool,
                   fingers: set[str]) -> tuple[set[Command], list[Command]]:
    """What the currently-pinched fingers of `hand` mean in `context`, split
    into (held commands, tap commands) — a pinch can carry either, e.g. Crouch
    is held for as long as you pinch while a hotbar cycle fires on the edge and
    repeats on its own cadence.

    Resolved fresh every frame rather than latched at the click edge: if the
    context changes mid-pinch the binding just re-resolves, and the held-key
    reconciliation releases the old command and presses the new one. Latching
    would leave the old key held with nothing to release it.
    """
    held: set[Command] = set()
    taps: list[Command] = []
    for finger in sorted(fingers):
        cmd = map_gesture(f"pinch_{finger}", hand, context)
        if cmd is None or _blocked_by_gui(cmd, context, gui_open):
            continue
        if cmd.mode == "held":
            held.add(cmd)
        else:
            taps.append(cmd)
    return held, taps


# ── Mouse thread ─────────────────────────────────────────────────────────────
def _mouse_loop() -> None:
    accum   = [0.0, 0.0]
    last_abs: tuple[float, float] | None = None   # last target actually posted in absolute mode
    interval = 1.0 / MOUSE_HZ

    while _running:
        t0 = time.monotonic()

        with _vel_lock:
            mode   = _mouse_mode[0]
            vx, vy = _mouse_vel
            fx, fy = _mouse_target
            button = _mouse_button[0]

        if mode == "absolute":
            # GUI pointer: hand position maps to a fraction of the content
            # area. Only post when the target has actually moved — warping
            # every tick regardless pinned the cursor against any real
            # trackpad input ("mouse trapped" reports) for zero benefit.
            with _ctx_lock:
                ctx = _shared_ctx[0]
            bounds = _pointer_bounds(ctx)
            if bounds and (fx, fy) != last_abs:
                cx, cy, cw, ch = bounds
                tx = cx + fx * cw
                ty = cy + fy * ch
                src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
                cur = Quartz.CGEventGetLocation(Quartz.CGEventCreate(src))
                # Drag event, not a plain move, whenever a button is held —
                # this is what makes window-dragging follow the cursor live
                # and lets Finder start a file drag at all.
                post_cursor_move(tx, ty, tx - cur.x, ty - cur.y, button)
                last_abs = (fx, fy)
            accum[0] = accum[1] = 0.0   # don't let stale velocity carry over if mode switches back

        elif vx != 0.0 or vy != 0.0:
            last_abs = None   # next absolute-mode entry must post unconditionally
            accum[0] += vx * interval
            accum[1] += vy * interval
            dx = int(accum[0])
            dy = int(accum[1])
            accum[0] -= dx
            accum[1] -= dy

            if dx != 0 or dy != 0:
                with _ctx_lock:
                    ctx = _shared_ctx[0]

                if ctx == "minecraft":
                    bounds = _content_bounds()
                    if bounds:
                        cx, cy, cw, ch = bounds
                        src = Quartz.CGEventSourceCreate(
                            Quartz.kCGEventSourceStateHIDSystemState)
                        cur = Quartz.CGEventGetLocation(Quartz.CGEventCreate(src))
                        nx = max(cx, min(cx + cw, cur.x + dx))
                        ny = max(cy, min(cy + ch, cur.y + dy))
                        # Deliberately a plain move even with a button held,
                        # unlike the absolute branch above. This is gameplay
                        # camera-look: GLFW reads the delta fields and treats
                        # moved/dragged identically, and mining-while-turning
                        # was confirmed working live this way — no reason to
                        # disturb it. The drag event type matters for AppKit
                        # drag *sessions* (windows, Finder files), which don't
                        # exist here.
                        post_cursor_move(nx, ny, dx, dy)
                    # else: bounds unavailable this tick (e.g. transient
                    # CGWindowList miss) — skip rather than risk an unclamped
                    # move escaping the window. A dropped frame is harmless;
                    # an escaped cursor clicking the desktop is not.
                else:
                    move_mouse_relative(dx, dy)

        elapsed = time.monotonic() - t0
        sleep = interval - elapsed
        if sleep > 0:
            time.sleep(sleep)


# ── Main loop (detection-rate: ~30 fps) ──────────────────────────────────────
def main(auto_calibrate: bool = False,
         stop_event: threading.Event | None = None,
         detector: "GestureDetector | None" = None):
    """
    auto_calibrate: run identically, except GUI_SENSITIVITY is live-editable —
    type a number + Enter anytime to set it, 'q' + Enter to quit (see
    calibrate.py). The last value set is persisted to data/settings.json and
    picked up as the default on the next run, calibrate or not.

    stop_event / detector: the GUI's embedding hooks. stop_event replaces the
    terminal KeyListener (no cbreak stdin when there's no terminal) and setting
    it ends the loop. A passed-in detector is borrowed, not owned — the GUI
    keeps one detector for the process and hands it between preview and
    control; detector.close() only releases the camera, so reuse is safe.
    """
    global _running
    _running = True   # main() may run more than once per process (GUI Start/Stop)

    log.info("Neuratect running. Press 'q' to quit." if not auto_calibrate
              else "Neuratect running (calibrate mode). Type 'q' + Enter to quit.")
    log.info("Close a fist to anchor that stick where your hand currently is.")
    log.info(f"log file: {_LOG_PATH}")

    settings = _load_settings()
    gui_sensitivity = [settings.get("gui_sensitivity", GUI_SENSITIVITY)]  # boxed: mutated by the input thread below
    pinch_engage  = settings.get("pinch_engage", PINCH_ENGAGE)
    pinch_release = settings.get("pinch_release", PINCH_RELEASE)
    quit_flag = threading.Event()

    if auto_calibrate:
        def _save_on_exit():
            _save_settings({**settings, "gui_sensitivity": gui_sensitivity[0]})
            log.info(f"[calibrate] saved gui_sensitivity={gui_sensitivity[0]:.3f} → {_SETTINGS_PATH}")
        # Registered here rather than after the main loop, so it still runs
        # on Ctrl+C (KeyboardInterrupt unwinds past the loop and this
        # function without executing anything after it — atexit is what
        # guarantees the save regardless of which way the loop exits).
        atexit.register(_save_on_exit)

        def _calibrate_input_loop():
            print(f"[calibrate] GUI_SENSITIVITY = {gui_sensitivity[0]:.3f}. "
                  f"Type a number + Enter to change it, or 'q' + Enter to quit.")
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                if line.lower() == "q":
                    break
                try:
                    value = float(line)
                except ValueError:
                    log.info(f"[calibrate] '{line}' isn't a number — ignored")
                    continue
                log.info(f"[calibrate] GUI_SENSITIVITY {gui_sensitivity[0]:.3f} → {value:.3f}")
                gui_sensitivity[0] = value
            quit_flag.set()

        threading.Thread(target=_calibrate_input_loop, daemon=True).start()

    last_tap:        dict[tuple[str, str], float] = {}   # (hand, cmd.name) → time
    last_gesture:    dict[str, str]   = {"left": "none", "right": "none", "unknown": "none"}
    gesture_since:   dict[str, float] = {}   # hand → monotonic time its current gesture began
    context_since    = time.monotonic()      # when the current context became active
    last_context     = None
    gui_pointer = GuiPointer(gui_sensitivity, pinch_engage, pinch_release)
    # Gameplay clicks: same pinch mechanism as the pointer, but the cursor is
    # driven by the camera-look joystick rather than GuiPointer. Separate
    # instance so switching between gameplay and a GUI screen can't inherit
    # half a click from the other side.
    game_pinch  = PinchClicker(pinch_engage, pinch_release)
    # Left hand: same mechanism on all four fingers — Jump / Crouch / hotbar
    # next / hotbar prev, per the mapper. Not exclusive: jumping while crouched
    # is ordinary play, unlike two mouse buttons at once. Runs whether or not a
    # GUI screen is open, because held Crouch + a right-hand pinch is inventory
    # shift-click.
    left_pinch  = PinchClicker(pinch_engage, pinch_release, exclusive=False,
                               fingers=("index", "middle", "ring", "pinky"))
    _sprint_on  = False
    last_ptr_mode: str | None = None  # pointer mode last frame ("mc-gui"|"desktop"|None) — entry/exit transitions
    frame_i = 0
    left_stick  = HandJoystick(name="left")
    right_stick = HandJoystick(
        name="right",
        deadzone=0.02,
        mouse_curve=1.0,
        smooth=0.6,   # snappier than the left stick — less lag on camera look
    )

    def fire_tap(cmd: Command, hand: str) -> None:
        """Fire a tap-mode command if its cooldown has elapsed. Shared by the
        pose path and the pinch path so a hotbar cycle has the same cadence
        whichever gesture drove it — the cooldown is both the gap between
        isolated taps and the repeat rate while the gesture is held."""
        now = time.time()
        key = (hand, cmd.name)
        if now - last_tap.get(key, 0.0) >= _TAP_COOLDOWN.get(cmd.name, TAP_COOLDOWN):
            tap(cmd)
            last_tap[key] = now

    mouse_thread = threading.Thread(target=_mouse_loop, daemon=True)
    mouse_thread.start()

    # calibrate mode reads numbers off stdin a line at a time, which needs
    # normal line-buffered input — that's incompatible with KeyListener's
    # cbreak mode (single keystrokes, no line editing), so it's skipped
    # entirely in favor of the 'q'-as-a-line handling above. Under the GUI
    # (stop_event set) there's no terminal at all, so no KeyListener either.
    headless = auto_calibrate or stop_event is not None
    key_cm = contextlib.nullcontext() if headless else KeyListener()
    with (detector or GestureDetector()) as detector, key_cm as keys:
        while not ((stop_event is not None and stop_event.is_set())
                   or (quit_flag.is_set() if headless else keys.should_quit)):
            gestures = detector.next_gestures()
            context  = get_active_context()

            with _ctx_lock:
                _shared_ctx[0] = context

            if context != last_context:
                log.info(f"[context] {context}")
                last_context = context
                context_since = time.monotonic()

            if context == "minecraft":
                _sync_gui_open()   # cursor visibility = ground truth for GUI state

            desired_held: set[Command] = set()
            active_hands: set[str]     = set()
            by_hand: dict[str, object] = {g.hand: g for g in gestures}

            for gesture in gestures:
                hand = gesture.hand
                active_hands.add(hand)

                if gesture.name != last_gesture.get(hand):
                    log.info(f"[gesture] {hand} → {gesture.name} ({gesture.confidence:.2f})")
                    gesture_since[hand] = time.monotonic()

                if gesture.name == "fist" and context == "minecraft":
                    last_gesture[hand] = gesture.name
                    continue

                if gesture.name == "open_palm":
                    stick = left_stick if hand == "left" else right_stick
                    stick.release_anchor()
                    # falls through — open_palm may still map to a command
                    # (e.g. ArrowDown in spotlight)

                cmd = map_gesture(gesture.name, hand, context)
                if cmd is None or _blocked_by_gui(cmd, context, _gui_open):
                    last_gesture[hand] = gesture.name
                    continue

                if cmd.mode == "held":
                    # No pose maps to a mouse button any more (HoldBreak /
                    # HoldPlace come from PinchClicker only), so nothing here
                    # needs click-specific gating.
                    desired_held.add(cmd)
                elif cmd.mode == "tap":
                    now = time.time()
                    already_in_context = (cmd.name == "OpenSpotlight" and context == "spotlight")
                    # OpenSpotlight needs a *deliberate* point: the gesture
                    # must have started in the current context (a break-blocks
                    # point held through a context flip satisfied a plain
                    # dwell instantly — live 2026-08-08, opened Spotlight
                    # mid-game) and been held for the dwell.
                    if cmd.name == "OpenSpotlight":
                        started = gesture_since.get(hand, 0.0)
                        if (started < context_since
                                or time.monotonic() - started < SPOTLIGHT_DWELL):
                            last_gesture[hand] = gesture.name
                            continue
                    if not already_in_context:
                        fire_tap(cmd, hand)

                last_gesture[hand] = gesture.name

            for hand in list(last_gesture.keys()):
                if hand not in active_hands:
                    last_gesture[hand] = "none"

            left_g = by_hand.get("left")

            # ── Left-hand pinches (Minecraft only) ───────────────────────────
            # Jump / Crouch / hotbar next / hotbar prev, resolved through the
            # mapper like any other binding. Tap-mode bindings (the hotbar)
            # fire on their own cadence for as long as the pinch is held, the
            # same as a held pose would. Runs before the WASD gate below, which
            # reads the pinch state.
            if context == "minecraft":
                if left_g is None:
                    left_pinch_events = left_pinch.hand_lost(time.monotonic())
                else:
                    left_pinch_events = left_pinch.update(left_g, time.monotonic())
                for finger, edge, ratio in left_pinch_events:
                    log.info(f"[pinch] left {finger} {edge} (ratio {ratio:.2f})")
                left_held, left_taps = _pinch_actions("left", context, _gui_open,
                                                      left_pinch.held_fingers())
                desired_held |= left_held
                for cmd in left_taps:
                    fire_tap(cmd, "left")
            elif left_pinch.held_fingers():
                left_pinch.reset()   # left Minecraft context with a pinch held

            # ── WASD joystick (Minecraft, left fist anchors on entry) ────────
            # Gated on GUI state: WASD does nothing while a screen (inventory,
            # pause menu) is open, but held keys would otherwise silently
            # stick and fire the moment the screen closes.
            # "pinch" counts as fist here: the templates are visually close and
            # classification flickers between them mid-drive, which dropped
            # WASD for a frame each time (stuttering walk, 2026-08-07 log).
            # A *held* pinch also keeps the stick live regardless of what the
            # pose classifier says: a ring/pinky pinch (hotbar) doesn't
            # necessarily read as fist, and dropping the walk every time you
            # changed slots would be the same stutter by another route.
            left_active = (context == "minecraft" and not _gui_open and bool(left_g)
                           and (left_g.name in ("fist", "pinch")
                                or bool(left_pinch.held_fingers())))
            vec = left_stick.update(left_active, left_g.wrist_x if left_g else 0.0,
                                     left_g.wrist_y if left_g else 0.0)
            if left_active:
                desired_held |= _wasd_from_joystick(vec)
            else:
                _reset_wasd_state()

            # ── Sprint: stick pushed past full deflection ─────────────────────
            # The WASD output saturates at MAX_RANGE, so all travel beyond it is
            # currently wasted — that's what sprint reads, like shoving a real
            # stick against its stop. Forward only (Minecraft can't sprint
            # sideways or backwards) and only while W is actually held.
            # Held rather than latched: a toggle would be a second piece of
            # game state to desync (the bug class the cursor-visibility fix
            # retired). It still *behaves* like the toggle you described,
            # because Minecraft keeps sprinting once started as long as W stays
            # down, even if you ease back off the threshold.
            sprint_now = (left_active and _wasd_state["w"]
                          and -vec.offset_y > (SPRINT_RELEASE if _sprint_on else SPRINT_ENGAGE))
            if sprint_now != _sprint_on:
                _sprint_on = sprint_now
                log.info(f"[sprint] {'on' if sprint_now else 'off'} "
                         f"(forward offset {-vec.offset_y:.3f}, full deflection {MAX_RANGE})")
            if _sprint_on:
                desired_held.add(Sprint)

            # ── Mouse — right hand anchors on entry ──────────────────────────
            right_g = by_hand.get("right")
            right_gesture = right_g.name if right_g else None

            ptr_mode = _pointer_mode(context, _gui_open)

            if ptr_mode is not None:
                # GuiPointer (relative delta + pinch clicks). No arm-delay —
                # there's no drift/overshoot risk here, tracking starts the
                # instant the hand is visible.
                now_m = time.monotonic()
                ptr_bounds = _pointer_bounds(context)
                if ptr_mode != last_ptr_mode:
                    # Entered pointer mode (or switched coordinate spaces) —
                    # seed the target from wherever the real cursor already
                    # is, so there's no jump.
                    gui_pointer.reset()
                    if ptr_bounds:
                        cx, cy, cw, ch = ptr_bounds
                        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
                        cur = Quartz.CGEventGetLocation(Quartz.CGEventCreate(src))
                        with _vel_lock:
                            _mouse_target[0] = min(1.0, max(0.0, (cur.x - cx) / cw))
                            _mouse_target[1] = min(1.0, max(0.0, (cur.y - cy) / ch))

                if right_g is None:
                    pinch_events = gui_pointer.hand_lost(now_m)
                elif right_gesture == "open_palm":
                    # Clutch: "mouse lifted" — hand repositions freely, cursor
                    # stays put. Open fingertips also read as pinch-released,
                    # so any held button drops via the normal hysteresis path.
                    gui_pointer.reset_tracking()
                    pinch_events = gui_pointer.update_pinch(right_g, now_m)
                else:
                    delta = gui_pointer.track(right_g, now_m)
                    if delta is not None and now_m >= gui_pointer.frozen_until and ptr_bounds:
                        cx, cy, cw, ch = ptr_bounds
                        with _vel_lock:
                            # deltas are isotropic (bounds-height units on
                            # both axes) — convert x by ch/cw so cursor
                            # speed doesn't depend on direction or the
                            # bounds' shape
                            _mouse_target[0] = min(1.0, max(0.0, _mouse_target[0] + delta[0] * ch / cw))
                            _mouse_target[1] = min(1.0, max(0.0, _mouse_target[1] + delta[1]))
                    pinch_events = gui_pointer.update_pinch(right_g, now_m)

                with _vel_lock:
                    fx_now, fy_now = _mouse_target
                gui_pointer.record(now_m, fx_now, fy_now)

                for finger, edge, ratio in pinch_events:
                    # Click at where the cursor was CLICK_REWIND ago — any
                    # residual motion from forming/releasing the pinch itself
                    # happened *after* the user had finished aiming — then
                    # hold the pointer still through the edge. The warp is
                    # synchronous so the mouse-down/up (posted in the held-key
                    # reconciliation below, at the current cursor position)
                    # can't race the 60 Hz mouse thread.
                    pos = gui_pointer.rewind(now_m - CLICK_REWIND) or (fx_now, fy_now)
                    with _vel_lock:
                        _mouse_target[0], _mouse_target[1] = pos
                    _warp_cursor_to_fraction(pos[0], pos[1], ptr_bounds)
                    gui_pointer.frozen_until = now_m + CLICK_FREEZE
                    log.info(f"[pinch] {finger} {edge} (ratio {ratio:.2f}) @ ({pos[0]:.2f}, {pos[1]:.2f})")

                # Right hand is the pointer: its bindings are mouse buttons,
                # never taps, so the tap half is empty by construction here.
                right_held, _ = _pinch_actions("right", context, _gui_open,
                                               gui_pointer.held_fingers())
                desired_held |= right_held

                with _vel_lock:
                    _mouse_mode[0] = "absolute"
                    _mouse_vel[0] = _mouse_vel[1] = 0.0
            else:
                # Gameplay: velocity camera look (right fist anchors on entry)
                # + pinch clicks, which are independent of each other — a fist
                # can be looking around while a thumb pinch holds the mouse
                # button down (mine while turning).
                now_m = time.monotonic()
                if last_ptr_mode is not None:
                    # Leaving pointer mode: drop everything it accumulated —
                    # stale EMA state caused a first-frame jump on GUI close.
                    # game_pinch starts clean too, so a pinch held across the
                    # transition has to be re-formed rather than continuing as
                    # a gameplay click.
                    gui_pointer.reset()
                    game_pinch.reset()
                    right_stick.reset_smoothing()

                if right_g is None:
                    pinch_events = game_pinch.hand_lost(now_m)
                else:
                    pinch_events = game_pinch.update(right_g, now_m)
                for finger, edge, ratio in pinch_events:
                    log.info(f"[pinch] right {finger} {edge} (ratio {ratio:.2f}) — gameplay click")
                right_held, _ = _pinch_actions("right", context, _gui_open,
                                               game_pinch.held_fingers())
                desired_held |= right_held

                right_active = right_gesture in _MOUSE_GESTURES
                vec = right_stick.update(right_active, right_g.wrist_x if right_g else 0.0,
                                          right_g.wrist_y if right_g else 0.0)
                with _vel_lock:
                    _mouse_mode[0] = "velocity"
                    if not right_active or vec.is_neutral:
                        _mouse_vel[0] = _mouse_vel[1] = 0.0
                    else:
                        _mouse_vel[0] = vec.raw_x * MOUSE_SENSITIVITY
                        _mouse_vel[1] = vec.raw_y * MOUSE_SENSITIVITY
            last_ptr_mode = ptr_mode

            frame_i += 1
            if auto_calibrate and frame_i % 30 == 0:
                # ~1 Hz pinch telemetry for tuning pinch_engage/release in
                # data/settings.json — watch what your natural pinch and
                # relaxed hand actually measure. Both hands and all four
                # fingers: the ring/pinky pinches (hotbar) live on the LEFT
                # hand, so right-hand-only telemetry couldn't tune them.
                for label, gg in (("left", left_g), ("right", right_g)):
                    if gg is None:
                        continue
                    log.info(f"[pinch] {label} ratios index={gg.pinch_index:.2f} "
                             f"middle={gg.pinch_middle:.2f} ring={gg.pinch_ring:.2f} "
                             f"pinky={gg.pinch_pinky:.2f} scale={gg.hand_scale:.3f}")

            # ── Held key reconciliation ──────────────────────────────────────
            for cmd in _held_commands - desired_held:
                log.info(f"[hold] release {cmd.name}")
                release(cmd)
            for cmd in desired_held - _held_commands:
                if context == "minecraft" and cmd.name in ("HoldBreak", "HoldPlace"):
                    if _gui_open:
                        _ensure_cursor_in_bounds()   # GUI click — position matters, keep it
                    else:
                        # Gameplay click — the game aims with the crosshair,
                        # not the cursor. Center it so a mouse-down can never
                        # land on window chrome (the resize-to-zero freeze).
                        _center_cursor_in_window()
                log.info(f"[hold] engage  {cmd.name}")
                hold(cmd)
            _held_commands.clear()
            _held_commands.update(desired_held)

            # Publish which button is actually down now, for the mouse thread's
            # move-vs-drag decision. Derived from _held_commands (what has been
            # posted) rather than the pinch state, so it can never claim a
            # button is down before the mouse-down event exists.
            names = {c.name for c in _held_commands}
            with _vel_lock:
                _mouse_button[0] = ("left" if "HoldBreak" in names else
                                    "right" if "HoldPlace" in names else None)

    _running = False
    log.info("Releasing held keys...")
    _release_all()
    log.info("Stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _running = False
        log.info("Releasing held keys...")
        _release_all()
        log.info("Stopped.")
