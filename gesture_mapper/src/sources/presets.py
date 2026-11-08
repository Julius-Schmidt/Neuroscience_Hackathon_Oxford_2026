"""Ready-made rule sets: a starting point the editor can load and the user then changes.

    one_switch   one fist = one click: claw / switch-scanning style (works with 1 band)
    two_band     right arm points + clicks, left fist = right click, double tap pauses
    controller   two bands as a game controller: left band = movement keys, right band =
                 mouse look + buttons (needs the bands' built-in poses)
    bci          EEG/BCI markers over LSL drive arrow keys + select
    minecraft5   Minecraft with our trained gestures on the left band (--myo gestures, pinch not used),
                 right band = look + fist break
    minecraft5_one  Minecraft with one band (right): arm = look, 5 gestures = everything else
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


def claw_drive():
    return {"mode": "normal", "rules": [
        _r("myo", "left.fist", "hold", [("claw_drive", {"axis": "x"})], (),
           "Hold left fist = glide left-right (tap = click)"),
        _r("myo", "right.fist", "hold", [("claw_drive", {"axis": "y"})], (),
           "Hold right fist = glide up-down (tap = click)"),
        _r("keyboard", "f8", "hold", [("claw_drive", {"axis": "x"})], (), "Test: Try = tap = click"),
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


def _toggle(side):
    return _r("myo", f"{side}.ulnar_deviation", "press",
              [("toggle_mode", {"a": "normal", "b": "alt"}), ("sound", {"name": "pop"})], (),
              "Ulnar deviation = switch layer (normal / alt)")


# "move": "game" = camera look that keeps turning past ~14 deg (myomouse.pointer.GamePointer)
def _only_in(app, data):
    """Every rule also needs `app` in front, so gestures never type into other windows."""
    for r in data["rules"]:
        r["conditions"].append(dict(R.new_condition("context"), value=app))
    return data


def minecraft5():
    """Left band: 4 trained gestures (pinch dropped: while moving the arm, neutral read as pinch
    42% of the time), two layers. Right band: look + fist = break."""
    n, a = [("mode", "normal")], [("mode", "alt")]
    # gesture_sides: only the left band runs the gesture model; the right one stays the fist click
    return _only_in("minecraft", {"mode": "normal", "gesture_sides": ["left"], "move": "game", "rules": [
        _r("myo", "right.pointer", "value", [("mouse_move", {"gain": 1.0})], (), "Right arm = look / aim"),
        _r("myo", "right.fist", "hold", [("command_hold", {"name": "HoldBreak"})], (), "Right fist = break / attack"),
        _toggle("left"),
        _r("myo", "left.wrist_extension", "hold", [("key_hold", {"key": "w"})], n, "Wrist extension = walk forward"),
        _r("myo", "left.power_grip", "hold", [("command_hold", {"name": "HoldPlace"})], n, "Power grip (fist) = place / use"),
        _r("myo", "left.chuck_grip", "hold", [("command_hold", {"name": "Jump"})], n, "Chuck = jump"),
        _r("myo", "left.power_grip", "hold", [("key_hold", {"key": "s"})], a, "Alt: power grip = walk back"),
        _r("myo", "left.wrist_extension", "press", [("command", {"name": "HotbarNext"})], a, "Alt: wrist extension = next hotbar slot"),
        _r("myo", "left.chuck_grip", "press", [("command", {"name": "Inventory"})], a, "Alt: chuck = inventory"),
    ]})


def minecraft5_one():
    """One band (right): arm = look, the 5 trained gestures do everything else, two layers."""
    n, a = [("mode", "normal")], [("mode", "alt")]
    return _only_in("minecraft", {"mode": "normal", "move": "game", "rules": [
        _r("myo", "right.pointer", "value", [("mouse_move", {"gain": 1.0})], (), "Arm = look / aim"),
        _toggle("right"),
        _r("myo", "right.power_grip", "hold", [("command_hold", {"name": "HoldBreak"})], n, "Power grip = break / attack"),
        _r("myo", "right.wrist_extension", "hold", [("key_hold", {"key": "w"})], n, "Wrist extension = walk forward"),
        _r("myo", "right.pinch_grip", "hold", [("command_hold", {"name": "HoldPlace"})], n, "Pinch = place / use"),
        _r("myo", "right.chuck_grip", "hold", [("command_hold", {"name": "Jump"})], n, "Chuck = jump"),
        _r("myo", "right.power_grip", "hold", [("key_hold", {"key": "s"})], a, "Alt: power grip = walk back"),
        _r("myo", "right.wrist_extension", "press", [("command", {"name": "HotbarNext"})], a, "Alt: wrist extension = next hotbar slot"),
        _r("myo", "right.pinch_grip", "press", [("command", {"name": "Inventory"})], a, "Alt: pinch = inventory"),
        _r("myo", "right.chuck_grip", "press", [("command", {"name": "Escape"})], a, "Alt: chuck = menu (Esc)"),
    ]})


def bci():
    return {"mode": "normal", "rules": [
        _r("lsl", "left", "press", [("key_tap", {"key": "left"})], (), "BCI 'left' = left arrow"),
        _r("lsl", "right", "press", [("key_tap", {"key": "right"})], (), "BCI 'right' = right arrow"),
        _r("lsl", "select", "press", [("key_tap", {"key": "enter"})], (), "BCI 'select' = Enter"),
        _r("lsl", "any marker", "press", [("say", {"text": "BCI event"})], (), "Show every BCI event"),
    ]}


PRESETS = {"claw": claw, "claw_drive": claw_drive, "one_switch": one_switch, "two_band": two_band, "controller": controller, "bci": bci,
           "minecraft5": minecraft5, "minecraft5_one": minecraft5_one}
TITLES = {"claw": "Claw: click anywhere with one gesture",
          "claw_drive": "Claw, two fists: hold to glide, tap to click", "one_switch": "One switch (1 band)", "two_band": "Two-band mouse",
          "controller": "Two-band game controller", "bci": "EEG / BCI (LSL markers)",
          "minecraft5": "Minecraft: 4 gestures (left) + look/break (right)",
          "minecraft5_one": "Minecraft: one band, 5 gestures + arm look"}
