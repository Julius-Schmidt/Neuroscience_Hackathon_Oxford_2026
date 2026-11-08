# MyoMouse: control a mouse with two Myo armbands

- **Right band:** moves the cursor, click gesture = **left click** (hold = drag)
- **Left band:** click gesture = **right click**

Pick the style with `--click` and `--move` (the `--move` default is `MOVE` in [config.py](myomouse/config.py)):

| | default | alternative |
|---|---|---|
| `--move` | `table`: forearm on the table. Turn the forearm = left/right, tilt the hand onto its side = up/down (`TABLE_Y = "pitch"`: lift the hand instead). Orientation only, so it doesn't drift | `air`: point the forearm in 3D (yaw/pitch); `slide`: slide on the table, measured with the accelerometer (experimental, drifts) |
| `--click` | `fist`: squeeze a fist | `index`: lift the index finger off the table (**experimental, not reliable**: only used when you pass `--click index`, never the default) |

Each click gesture has its own recordings and model (`fist_<side>` / `index_<side>`), so you can
train both and switch with a flag. The original setup is `--move air` (with the default fist).

```
Band R ──▶ process 1: IMU (table / air) → cursor speed
       └─▶ EMG → click classifier R (fist / index) ─┐
                                                    ├─▶ mouse controller (main process) ─▶ OS
Band L ──▶ process 2: EMG → click classifier L ─────┘
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
python -m myomouse.record --user julius --side right # ~2 min, follow the cues (webcam watches your hand)
python -m myomouse.record --user julius --side left
python -m myomouse.train  --user julius --side right # prints accuracy + "caught X/Y fists, Z false clicks"
python -m myomouse.train  --user julius --side left
python -m myomouse.run --user julius --dry-run --show   # watch live click probabilities, no mouse control
python -m myomouse.arena --user julius    # practice: move a cursor + hit targets (real mouse untouched)
python -m myomouse.run --user julius      # the live demo + playground window (Esc / Ctrl+C to stop)
python -m myomouse.off                    # let the bands sleep again (--deep powers them off)
```

All of `record`, `train`, `run` and `arena` take `--click fist|index` (default `fist`); `record`,
`run` and `arena` also take `--move table|air|slide`. To try the experimental index lift, pass
`--click index` to every step (`record`, `train`, `run` / `arena`): it has its own recordings and
model, so it never mixes with the fist ones. Add `--move air` for the 3D pointing.
`record --move` only changes the instructions, so you practise the same movement you'll use live (the classifier must learn that it isn't a click).

- **Users:** every recording and model belongs to one user, so `record`, `train`, `run`,
  `arena` and `devtool` need `--user <name>` (only `--check` works without it).
  Recordings go to `data/<user>/`, models to `models/<user>/fist_<side>.joblib`, and `train`
  only uses that user's recordings. A new person records + trains under their own name.
- **Camera labels:** while you record, the webcam (MediaPipe hand tracking) checks when your
  hand *actually* closed and opened (fist), or when your index finger *actually* lifted (angle
  between the index and the middle/ring fingers), and the labels follow that instead of the cue timing
  (your reaction time, holding too long, a fist while you should relax). Put the laptop so it
  sees the hand of the arm you're recording. The camera window shows the instructions and cues;
  SPACE there starts each part, ESC quits. If the window says it sees the wrong hand, use
  `--swap-hands`. Where the camera can't tell, those bits are left out or keep the cue timing.
  It's only used for training: `run` controls the mouse with the Myo bands alone.
  `--no-camera` records the old way; `train --labels cue` trains without the camera labels,
  so you can compare the "caught X/Y ..., Z false clicks" lines. For the index, put the camera
  so it sees the hand from the side/front: a lifted finger is hard to see from straight above.
  **Which camera:** `VISION_CAMERA` in [config.py](myomouse/config.py) (now `"web camera"`, the
  external one; `""` = built-in). Override per run with `--camera "web camera"` or `--camera 0`;
  `python -m myomouse.record --list-cameras` shows them all and which one is used.
  Settings: `VISION_*` in [config.py](myomouse/config.py).
- **Playground (`run`):** the bands move the real mouse, and a window shows the cursor trail,
  every click (blue ripple = left, orange = right), which buttons are held, live fist
  probabilities and an event log. Left-click a block to change its colour; hold the right
  button (left hand's click) on a block to drag it. Works anywhere else on the desktop too.
  `--no-window` = terminal only; `--dry-run` = window draws its own cursor, real mouse untouched.
- **Different demo user:** re-record and re-train both arms. Recording more sessions for the
  same person adds to the training data, because `train` uses every recording of that gesture
  (`data/<user>/index_<side>_*.npz` or `data/<user>/<side>_*.npz` for the fist).
- **If the classifier misbehaves:** `python -m myomouse.run --user julius --mode threshold` uses a simple
  calibrated activation threshold instead.
- **One band only:** `python -m myomouse.run --user julius --sides right`.
- **Practice / tuning the cursor:** `python -m myomouse.arena --user julius` opens a window with its own cursor
  and targets (blue = right hand, orange = left hand). `+`/`-` change the gain and `X`/`Y` flip
  directions live; on exit it prints the values to paste into `config.py`. Before training,
  use `--check` (cursor only, Space/Enter click).
- **No hardware:** `python -m myomouse.run --user julius --fake --dry-run` (needs trained models).

## Arena, one-gesture modes and analysis tools (added Sun 8 Nov)

**Arena** (`python -m myomouse.arena --user <you>`), keys while it runs:

| key | does |
|---|---|
| `M` | control mode: band (the `--move` pointer) / relative / absolute (arm angle = cursor position) / joystick (tilt = cursor speed) |
| `A` | which arm angles: yaw+pitch (air) / yaw+roll (table) / roll+pitch |
| `K` | calibrate: 10 s sweeping your *comfortable* range; screen fits the shape you swept (lopsided is fine) |
| `V` | angle map: what K recorded, the fitted region, where the screen corners sit, your arm now |
| `P` | **claw** (point scanning, one gesture, no arm movement): a line sweeps across, fist stops it; a second sweeps down, fist clicks. `,` `.` sweep speed, `;` `'` reaction-time compensation |
| arrows, `+` `-`, `[` `]` | x / y / both sensitivity, joystick dead zone |

**Other one-gesture tools**
- `python -m myomouse.sound_switch --user <you> --sound mypop`: every fist plays a sound, so an
  iPhone's Switch Control (Sound switch) hears a switch press. `--test` plays it on Enter;
  `--record NAME` records your own (needs `pip install sounddevice`). Your own recorded pop works far better than a synthetic one.
- `python -m myomouse.onboard`: the Myo's **built-in** 5-pose classifier (no training). Do the
  sync gesture (wrist bent back, hold until it vibrates). `--sound mypop` = phone switch on every fist.

**Analysis** (offline, on a dev tool mode 4 fatigue recording in `data/devtool/sessions/`):
- `python -m devtool.fatigue_features <id>_fatigue_raw.npz`: RMS, mean/median frequency, Dimitrov,
  15-45 Hz share, zero crossings per squeeze, ranked by how well they track fatigue.
- `python -m devtool.weakness_sim <id>_fatigue_raw.npz --models models/<you>`: replays the recording with
  the muscle signal shrunk (real noise kept) to find where each click detector breaks.
- `python -m devtool.tremor_sim <id>_fatigue_raw.npz --models models/<you>`: adds simulated 4-6 Hz
  tremor and compares the debouncer hold time (`N_ON` in config.py; 3 = 30 ms, 20 = 200 ms "tremor mode").

**Gesture mapper** (`gesture_mapper/`) now runs on Windows and has a **Blocks** page: a
Scratch-style editor where any input (camera poses, Myo bands, EEG/BCI markers over LSL) drives any
action. See [gesture_mapper/README.md](gesture_mapper/README.md#blocks-any-input--any-action).

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

- **Cursor too slow/fast:** `GAIN_X`, `GAIN_Y` (air), `TABLE_GAIN_TURN`, `TABLE_GAIN_SLIDE` (table)
- **Cursor goes the wrong way:** `INVERT_X`, `INVERT_Y` (air), `TABLE_INVERT_X`, `TABLE_INVERT_Y` (table)
- **Table: up/down too slow/fast:** `TABLE_GAIN_Y`; tilting feels wrong: `TABLE_Y = "pitch"` (lift the hand).
- **Slide mode drifting after you stop:** lower `TABLE_LEAK`, raise `TABLE_ACC_DEADZONE_G`;
  **sliding barely moves it:** raise `TABLE_GAIN_SLIDE`, lower `TABLE_STILL_N`.
- **Lock to one axis:** `AXES` / `--axes` (default `both` = free; `dominant` = only along the axis you're
  moving more right now, makes paths blocky; `x` or `y` = only that axis). `AXES_SWITCH` = how much stronger the
  other axis must be to take over.
- **Jittery cursor:** raise `DEADZONE_RAD` or `SMOOTHING`
- **Clicks too slow:** lower `N_ON` or `WINDOW`
- **False clicks:** raise `P_ON` / `N_ON`, and record more "relax while moving" data
- **Swap buttons / pointer arm:** `BUTTON_FOR_SIDE`, `POINTER_SIDE`

## Tests (no hardware)

```bash
pytest -q tests
```
