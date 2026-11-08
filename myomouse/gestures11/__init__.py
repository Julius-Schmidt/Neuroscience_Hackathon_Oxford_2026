"""Multi-gesture recognition: 5 hand/wrist gestures from one band's EMG.

The 5 are the best-separated set on the Myo out of the 10 from the EMG-gesture literature
(tested every combination on a real recording): ulnar deviation, wrist extension, power grip,
chuck grip, pinch grip. Dropped: radial deviation, wrist flexion, supination, pronation, open
hand (too often confused with neutral / each other). Older recordings that contain them still
load; those parts are simply skipped.

Separate from the fist / index click (myomouse.record / train / run): own recordings,
own model, nothing here changes how clicking works.

    python -m myomouse.gestures11.record --user julius --side right    # ~5 min, follow the cues
    python -m myomouse.gestures11.train  --user julius --side right    # accuracy + confusion matrix
    python -m myomouse.gestures11.live   --user julius --side right    # live: which gesture now

    data/<user>/gestures_<side>_<timestamp>.npz      recordings
    models/<user>/gestures_<side>.joblib             model (scaler + LDA, or SVM with --model svm)
"""

from dataclasses import dataclass

from ..users import data_dir, model_dir


@dataclass(frozen=True)
class Gesture:
    key: str     # stored in files and models
    name: str    # shown while recording
    short: str   # confusion-matrix column
    how: str     # instruction


GESTURES = [
    Gesture("neutral", "Neutral", "neut", "forearm on the table, thumb up, hand loose, wrist straight"),
    Gesture("ulnar_deviation", "Ulnar deviation", "uln", "flat hand, bend the wrist towards the little-finger side"),
    Gesture("wrist_extension", "Wrist extension", "ext", "bend the wrist back, back of the hand towards the forearm"),
    Gesture("power_grip", "Power grip", "power", "squeeze a firm fist"),
    Gesture("chuck_grip", "Chuck grip", "chuck", "thumb, index and middle fingertips together"),
    Gesture("pinch_grip", "Pinch grip", "pinch", "thumb and index fingertips together"),
    # not a gesture: the fully relaxed hand. Like neutral it means "no gesture" in the live output,
    # but it's a separate class because its EMG is quieter than holding the hand in neutral.
    Gesture("rest", "Rest", "rest", "forearm on the table, hand completely limp - no effort at all"),
]
KEYS = [g.key for g in GESTURES]
NEUTRAL = 0
REST = KEYS.index("rest")
IDLE = (NEUTRAL, REST)          # classes that mean "no gesture"
IDLE_KEYS = ("neutral", "rest")
IGNORE = -1

# --- recording protocol ---------------------------------------------------------
HOLD_S = 3.0       # hold each gesture this long
RELAX_S = 2.5      # back to neutral between gestures
SETTLE_S = 0.6     # first part of every hold/relax is left out (reaction time, moving into the pose)
END_TRIM_S = 0.2   # ... and the last bit (people let go early)
REPS = 5           # repetitions of every gesture per recording, arm still (one block per repetition) ...
MOVE_REPS = 5      # ... and then while moving the arm the whole time, like when you play: without these
                   # the model has never seen arm movement and guesses (the fist protocol does the same)
MOVE_NEUTRAL_S = 15.0  # neutral hand while moving the arm, before the moving repetitions: "moving is not a gesture"
REST_STILL_S = 10.0  # rest block at the start: hand limp, arm still ...
REST_MOVE_S = 12.0   # ... then hand limp while moving the arm (arm position changes the EMG)
REST_END_S = 8.0     # and a short limp rest at the end (the band warms up / sweat changes the signal)

# --- camera labels (record only, see gestures11/vision.py) ---------------------------
CAM_MIN_GAP_CM = 1.0   # a hold must change the hand pose at least this much (cm) for the camera to time it
CAM_STRAY_FRAC = 0.6   # in neutral phases, a pose change above this fraction of CAM_MIN_GAP_CM is left out

# --- classifier --------------------------------------------------------------
WINDOW = 40        # samples per window (200 ms)
STEP = 5           # a new decision every 5 samples (25 ms)
# Live decisions (gestures11/decide.py); everything below falls back to "neutral" = no gesture
REJECT = 0.6       # the classifier must be at least this sure, else neutral (literature: 0.6-0.75 works best)
REST_GATE = False  # off: with these 5 gestures it hid power grip (a quiet gesture) and REJECT + N_ON already
                   # keep false gestures at ~1-2%. train --rest-gate turns it on.
GATE_K = 3.0       # rest gate: EMG quieter than rest median + GATE_K * spread of rest -> neutral
N_ON = 3           # this many decisions in a row (75 ms) to switch INTO a gesture ...
N_OFF = 2          # ... but only this many (50 ms) to drop back to neutral


def recording_prefix(side: str) -> str:
    return f"gestures_{side}_"


def recordings(user: str, side: str):
    return sorted(data_dir(user).glob(f"{recording_prefix(side)}*.npz"))


def model_path(user: str, side: str):
    return model_dir(user) / f"gestures_{side}.joblib"


def parse_gestures(text: str | None) -> list[int]:
    """'power_grip,open_hand' (keys or short names) -> gesture indices; None -> all. Neutral + rest always kept."""
    if not text:
        return list(range(len(GESTURES)))
    lookup = {g.key: i for i, g in enumerate(GESTURES)} | {g.short: i for i, g in enumerate(GESTURES)}
    out = set(IDLE)
    for part in text.split(","):
        part = part.strip()
        if part not in lookup:
            raise SystemExit(f"unknown gesture {part!r}; use one of: {', '.join(KEYS)}")
        out.add(lookup[part])
    return sorted(out)
