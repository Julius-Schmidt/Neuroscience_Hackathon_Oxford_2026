"""
Context detection: returns the active control scheme.

Uses CGWindowListCopyWindowInfo exclusively, not NSWorkspace.frontmostApplication().
NSWorkspace can misreport apps that are launched via a separate wrapper/launcher
process (e.g. Minecraft's launcher spawns the game window but doesn't hand off
"frontmost" status to it) — the window list's owner name for the topmost window
is the reliable signal. CGWindowListCopyWindowInfo with kCGWindowListOptionOnScreenOnly
returns windows front-to-back, so the first layer-0 window is the active app.
"""

import Quartz

_WINDOW_CONTEXTS = {
    "spotlight": "spotlight",
}

_APP_CONTEXTS = {
    "minecraft": "minecraft",
    "java": "minecraft",       # Minecraft runs as "java" on macOS
}

import time as _time
_ctx_cache: str = "default"
_ctx_ts:    float = 0.0
_CTX_TTL:   float = 1.0   # seconds between window-list queries; context rarely changes


def get_active_context() -> str:
    global _ctx_cache, _ctx_ts
    now = _time.monotonic()
    if now - _ctx_ts < _CTX_TTL:
        return _ctx_cache

    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )

    # Overlay windows (Spotlight etc) first — these may float above the
    # active app without being a normal layer-0 window themselves.
    for w in windows:
        owner = (w.get("kCGWindowOwnerName") or "").lower()
        for key, ctx in _WINDOW_CONTEXTS.items():
            if key in owner:
                _ctx_cache, _ctx_ts = ctx, now
                return ctx

    # Topmost normal window's owner = the active app.
    for w in windows:
        if w.get("kCGWindowLayer") == 0:
            owner = (w.get("kCGWindowOwnerName") or "").lower()
            for key, ctx in _APP_CONTEXTS.items():
                if key in owner:
                    _ctx_cache, _ctx_ts = ctx, now
                    return ctx
            break

    _ctx_cache, _ctx_ts = "default", now
    return "default"
