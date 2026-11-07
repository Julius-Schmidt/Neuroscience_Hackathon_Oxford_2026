# Neuratect

Gesture-controlled command mapper for macOS. A webcam watches your hands; hand poses and
finger pinches become real keyboard and mouse input. Camera in, OS actions out.

Status: working MVP, macOS only. Two contexts are built out — the desktop (the hand *is* the
mouse) and Minecraft (movement, camera, clicks, hotbar, inventory).

## Requirements

macOS, a webcam, and Python 3.10+ (developed on 3.13).

```
pip install -r requirements.txt
```

Versions there are pinned to what this actually runs on, so a second machine behaves the
same. The MediaPipe model file `src/cv/hand_landmarker.task` is in the repo — no download
step.

## Permissions (do this first)

System Settings → Privacy & Security:

- **Camera** — for the terminal or app you launch from
- **Accessibility** — same, required to synthesize input

Both fail **silently**. Without Camera you get no hands; without Accessibility every
keystroke and click is dropped with no error at all. Grant them per machine — a new laptop
needs them again.

## Run

```
python3 gui.py          # the app: onboarding, live preview, Start/Stop control,
                        # gesture recording, mapping editor, settings
python3 run.py          # headless control loop only — quit with a bare 'q'
```

The GUI onboards on first run (replay it from Settings), previews the camera with a
live landmark overlay, and its Start Control button runs the same loop as `run.py`.
Bindings edited in the Mappings page are saved to `data/mappings.json` and apply
immediately, control running or not.

The control scheme, in one paragraph: your **right hand is the pointer** — any pose moves the
cursor, thumb-to-index pinch is left click, thumb-to-middle is right click, an open palm is a
clutch (lift and reposition without moving the cursor). Your **left hand is commands** — a
fist anchors a virtual joystick (WASD in Minecraft; push past full deflection to sprint), and
its four thumb-pinches fire bindings. In Minecraft gameplay the right fist drives camera look
instead of the cursor, because aiming a camera is a rate control, not pointing.

Exact bindings per context live in `data/mappings.json` (edited by the GUI's Mappings
page; defaults in `src/mapping-engine/mapper.py`) — deliberately not duplicated here.

## Tuning

```
python3 calibrate.py    # quit with 'q' + Enter (line-buffered, unlike run.py)
```

Type a number + Enter to change pointer sensitivity live, and watch the ~1 Hz pinch-ratio
telemetry for both hands and all four fingers. Whatever you last set is saved to
`data/settings.json` and becomes the default for both scripts.

`data/settings.json` keys: `gui_sensitivity` (screen-heights of cursor travel per hand-width
of movement), `pinch_engage` / `pinch_release` (pinch distance / hand size — hysteresis, so
release must be above engage). Set thresholds from measured ratios, not guesses.

## Tests

```
python3 tests/test_click_routing.py    # …or any of the five suites
```

No camera and no Minecraft needed; each prints PASS/FAIL per check and exits non-zero on
failure. Two suites drive the real `main()` loop with a scripted hand and stubbed OS calls,
which is the only coverage of the wiring.

## Other tools

- `python3 src/cv/manage_gestures.py` — headless record/list/delete of pose templates
  (`data/gestures.json`); the GUI's Gestures page is the friendlier way to do the same
- `python3 src/cv/joystick_debug.py` — serves an annotated MJPEG stream for joystick tuning
- `python3 log_manual_input.py` — logs real mouse/keyboard input; needs only Input
  Monitoring, not Accessibility

## Gotchas

- Running `run.py` with your hands in frame posts **real clicks**. Expect stray desktop
  clicks while testing.
- `logs/session.log` is overwritten on every run — copy anything worth keeping first.
- A killed process can leave the camera light on; kill the stale pid.
- Context detection reads the *focused* window, so the target app has to actually be in
  front.

## Layout

`src/cv/` camera and classification · `src/mapping-engine/` (gesture, context) → command ·
`src/os-integration/` all macOS API calls · `notes/` state, decisions, deferred ·
`build-plan.md` how the next phase is meant to go.

Each module folder has a `CLAUDE.md` with its non-obvious constraints — read that before
changing the module.
