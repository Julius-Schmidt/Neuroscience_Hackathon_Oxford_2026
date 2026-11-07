# MyoMouse: control a mouse with two Myo armbands

- **Right band:** forearm pointing (IMU orientation) moves the cursor, fist = **left click** (hold = drag)
- **Left band:** fist = **right click**

```
Band R ──▶ process 1: orientation → cursor speed
       └─▶ EMG → fist classifier R ─┐
                                    ├─▶ mouse controller (main process) ─▶ OS
Band L ──▶ process 2: EMG → fist classifier L ─┘
```

Each band runs in its own process and connects over the computer's built-in Bluetooth
(`bleak`, no USB dongle), so one band dropping out doesn't stall the other.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No dongle needed. On macOS, give your terminal / VS Code these permissions in
System Settings → Privacy & Security:
- **Bluetooth**, to find and connect to the bands
- **Accessibility**, otherwise the mouse won't move
- **Camera**, only for recording training data (see below)

Close Myo Connect and any other script using the bands first: a band only accepts one connection.

## Demo-day steps

```bash
python -m myomouse.identify            # each band vibrates, you say which arm -> bands.json
python -m myomouse.run --check --dry-run  # no model needed: EMG rate (~200 Hz) + muscle activity bars
python -m myomouse.run --check            # same, and the right band moves the cursor (tune gain/invert)
python -m myomouse.record --side right # ~2 min, follow the cues (webcam watches your hand)
python -m myomouse.record --side left
python -m myomouse.train  --side right # prints accuracy + "caught X/Y fists, Z false clicks"
python -m myomouse.train  --side left
python -m myomouse.run --dry-run --show   # watch live fist probabilities, no mouse control
python -m myomouse.arena                  # practice: move a cursor + hit targets (real mouse untouched)
python -m myomouse.run                    # the live demo + playground window (Esc / Ctrl+C to stop)
python -m myomouse.off                    # let the bands sleep again (--deep powers them off)
```

- **Camera labels:** while you record, the webcam (MediaPipe hand tracking) checks when your
  hand *actually* closed and opened, and the fist labels follow that instead of the cue timing
  (your reaction time, holding too long, a fist while you should relax). Put the laptop so it
  sees the hand of the arm you're recording. The camera window shows the instructions and cues;
  SPACE there starts each part, ESC quits. If the window says it sees the wrong hand, use
  `--swap-hands`. Where the camera can't tell, those bits are left out or keep the cue timing.
  It's only used for training: `run` controls the mouse with the Myo bands alone.
  `--no-camera` records the old way; `train --labels cue` trains without the camera labels,
  so you can compare the "caught X/Y fists, Z false clicks" lines.
  Settings: `VISION_*` in [config.py](myomouse/config.py).
- **Playground (`run`):** the bands move the real mouse, and a window shows the cursor trail,
  every click (blue ripple = left, orange = right), which buttons are held, live fist
  probabilities and an event log. Left-click a block to change its colour; hold the right
  button (left fist) on a block to drag it. Works anywhere else on the desktop too.
  `--no-window` = terminal only; `--dry-run` = window draws its own cursor, real mouse untouched.
- **Different demo user:** re-record and re-train both arms. Recording more sessions for the
  same person adds to the training data, because `train` uses every `data/<side>_*.npz`.
- **If the classifier misbehaves:** `python -m myomouse.run --mode threshold` uses a simple
  calibrated activation threshold instead.
- **One band only:** `python -m myomouse.run --sides right`.
- **Practice / tuning the cursor:** `python -m myomouse.arena` opens a window with its own cursor
  and targets (blue = right fist, orange = left fist). `+`/`-` change the gain and `X`/`Y` flip
  directions live; on exit it prints the values to paste into `config.py`. Before training,
  use `--check` (cursor only, Space/Enter click).
- **No hardware:** `python -m myomouse.run --fake --dry-run` (needs trained models).

## How it stays fast

| Stage | Time |
|---|---|
| Bluetooth from the band | ~10–20 ms |
| EMG window | 150 ms (30 samples), but a strong fist triggers before the window is full |
| Features (MAV, RMS, WL, ZC, SSC × 8 channels) + LDA | ~0.2 ms, run every 10 ms |
| Debounce (3 fist decisions in a row) | ~30 ms |
| Inter-process queue → mouse | < 2 ms |

Clicks use hysteresis (press above 0.80, release below 0.35), so a click doesn't flicker.
The cursor freezes for 150 ms around each press/release so clenching doesn't move it.

## Tuning ([myomouse/config.py](myomouse/config.py))

- **Cursor too slow/fast:** `GAIN_X`, `GAIN_Y`
- **Cursor goes the wrong way:** `INVERT_X`, `INVERT_Y`
- **Jittery cursor:** raise `DEADZONE_RAD` or `SMOOTHING`
- **Clicks too slow:** lower `N_ON` or `WINDOW`
- **False clicks:** raise `P_ON` / `N_ON`, and record more "relax while moving" data
- **Swap buttons / pointer arm:** `BUTTON_FOR_SIDE`, `POINTER_SIDE`

## Tests (no hardware)

```bash
pytest -q tests
```
