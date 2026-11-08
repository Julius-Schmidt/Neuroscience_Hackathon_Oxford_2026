"""Ready-made rule sets: a starting point the editor can load and the user then changes.

    one_switch   one fist = one click: claw / switch-scanning style (works with 1 band)
    two_band     right arm points + clicks, left fist = right click, double tap pauses
    controller   two bands as a game controller: left band = movement keys, right band =
                 mouse look + buttons (needs the bands' built-in poses)
    bci          EEG/BCI markers over LSL drive arrow keys + select
"""

import rules as R


def _r(src, name, trig, acts, conds=(), title=""):
    r = R.new_rule(src, name, trig)
    r["title"] = title
    r["actions"] = [dict(R.new_action(k), **kw) for k, kw in acts]
    r["conditions"] = [dict(R.new_condition(k), value=v) for k, v in conds]
    return r


def one_switch():
    return {"mode": "normal", "rules": [
        _r("myo", "right.fist", "hold", [("mouse_hold", {"button": "left"})], [("not_mode", "paused")],
           "Fist = click (hold to drag)"),
    ]}


def claw():
    return {"mode": "normal", "rules": [
        _r("myo", "right.fist", "press", [("claw_step", {})], (), "Fist = claw: start / stop / click"),
        _r("myo", "right.double_tap", "press", [("claw_cancel", {})], (), "Double tap = cancel the claw"),
        _r("lsl", "select", "press", [("claw_step", {})], (), "BCI 'select' also drives the claw"),
        _r("keyboard", "f8", "press", [("claw_step", {})], (), "Test: Try this to step the claw by hand"),
    ]}


def two_band():
    return {"mode": "normal", "rules": [
        _r("myo", "right.pointer", "value", [("mouse_move", {"gain": 1.0})], [("not_mode", "paused")],
           "Right arm moves the mouse"),
        _r("myo", "right.fist", "hold", [("mouse_hold", {"button": "left"})], [("not_mode", "paused")],
           "Right fist = left click"),
        _r("myo", "left.fist", "press", [("mouse_click", {"button": "right"})], [("not_mode", "paused")],
           "Left fist = right click"),
        _r("myo", "right.double_tap", "press", [("toggle_mode", {"a": "normal", "b": "paused"}),
                                               ("sound", {"name": "pop"})], (), "Double tap pauses / resumes"),
    ]}


def controller():
    return {"mode": "normal", "rules": [
        _r("myo", "left.fist", "hold", [("key_hold", {"key": "w"})], (), "Left fist = forward (W)"),
        _r("myo", "left.fingers_spread", "hold", [("key_hold", {"key": "s"})], (), "Left spread = back (S)"),
        _r("myo", "left.wave_in", "hold", [("key_hold", {"key": "a"})], (), "Left wave in = left (A)"),
        _r("myo", "left.wave_out", "hold", [("key_hold", {"key": "d"})], (), "Left wave out = right (D)"),
        _r("myo", "left.double_tap", "press", [("key_tap", {"key": "space"})], (), "Left double tap = jump"),
        _r("myo", "right.pointer", "value", [("mouse_move", {"gain": 1.0})], (), "Right arm = look / aim"),
        _r("myo", "right.fist", "hold", [("mouse_hold", {"button": "left"})], (), "Right fist = left button"),
        _r("myo", "right.fingers_spread", "hold", [("mouse_hold", {"button": "right"})], (), "Right spread = right button"),
        _r("myo", "right.wave_in", "press", [("scroll", {"amount": 1})], (), "Right wave in = scroll up / next item"),
        _r("myo", "right.wave_out", "press", [("scroll", {"amount": -1})], (), "Right wave out = scroll down"),
        _r("myo", "right.double_tap", "press", [("key_tap", {"key": "esc"})], (), "Right double tap = menu (Esc)"),
    ]}


def bci():
    return {"mode": "normal", "rules": [
        _r("lsl", "left", "press", [("key_tap", {"key": "left"})], (), "BCI 'left' = left arrow"),
        _r("lsl", "right", "press", [("key_tap", {"key": "right"})], (), "BCI 'right' = right arrow"),
        _r("lsl", "select", "press", [("key_tap", {"key": "enter"})], (), "BCI 'select' = Enter"),
        _r("lsl", "any marker", "press", [("say", {"text": "BCI event"})], (), "Show every BCI event"),
    ]}


PRESETS = {"claw": claw, "one_switch": one_switch, "two_band": two_band, "controller": controller, "bci": bci}
TITLES = {"claw": "Claw: click anywhere with one gesture", "one_switch": "One switch (1 band)", "two_band": "Two-band mouse",
          "controller": "Two-band game controller", "bci": "EEG / BCI markers"}
