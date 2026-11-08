"""All tunable numbers in one place. Change these during the demo prep, not in the code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
BANDS_FILE = ROOT / "bands.json"  # written by `python -m myomouse.identify`

# --- EMG -------------------------------------------------------------------
EMG_HZ = 200          # raw Myo EMG rate (8 channels, int8)
N_CHANNELS = 8
WINDOW = 30           # samples per classification window (150 ms)
STEP = 2              # classify every STEP new samples (10 ms)

# --- Click debouncing (hysteresis) -----------------------------------------
P_ON = 0.80           # fist probability needed to count towards a press
P_OFF = 0.35          # probability must drop below this to count towards a release
N_ON = 3              # consecutive "fist" decisions before pressing (~30 ms)
N_OFF = 4             # consecutive "rest" decisions before releasing (~40 ms)

# --- Which style (override per run with --click / --move) ----------------------
# The click gesture is always "fist" unless you pass --click index (experimental, not
# reliable yet). It is not a config option on purpose, so index never becomes the default.
MOVE = "table"             # "table" = forearm on the table (turn + tilt), "air" = point the forearm (3D),
                           # "slide" = slide on the table, accelerometer (experimental, can drift)

# --- Cursor (IMU on the pointer band) ---------------------------------------
POINTER_SIDE = "right"     # which band moves the cursor
GAIN_X = 1800.0            # pixels per radian of forearm yaw
GAIN_Y = 1800.0            # pixels per radian of forearm pitch
INVERT_X = False           # flip if the cursor moves the wrong way left/right
INVERT_Y = False           # flip if the cursor moves the wrong way up/down
# air Pointer with y from forearm roll instead of pitch (arena 'A' key / devtool --y-axis roll)
POINTER_Y_AXIS = "pitch"
GAIN_ROLL = 900.0          # pixels per radian of forearm roll (only when POINTER_Y_AXIS = "roll")
INVERT_ROLL = False        # flip if rolling moves the cursor the wrong way up/down
DEADZONE_RAD = 0.0015      # ignore tiny per-sample jitter (radians per IMU sample)
SMOOTHING = 0.5            # 0 = no smoothing, closer to 1 = smoother but laggier
FREEZE_MS = 150            # freeze the cursor this long after a press/release

# --- Cursor on the table (--move table, and --move slide) --------------------------
# table: left/right = turning the forearm on the table, up/down = TABLE_Y. Orientation only, no drift.
TABLE_GAIN_TURN = 1800.0   # pixels per radian of forearm turn on the table (left/right)
TABLE_Y = "roll"           # up/down from "roll" = tilting the hand onto its side, or "pitch" = lifting the hand
TABLE_GAIN_Y = 4000.0      # pixels per radian of roll / pitch (up/down). Higher than left/right:
                           # the band near the elbow only feels part of the hand's tilt
TABLE_DEADZONE_Y = 0.0004  # jitter filter for up/down (radians per IMU sample), smaller than DEADZONE_RAD
TABLE_INVERT_X = False     # flip if turning left moves the cursor right
TABLE_INVERT_Y = False     # flip if up/down is the wrong way round
# slide only: forward/back = sliding the arm (accelerometer, integrated to a speed, reset when still)
TABLE_GAIN_SLIDE = 4000.0  # pixels per metre of sliding
TABLE_SLIDE_X = 1.0        # 0 = left/right only from turning, 1 = sideways sliding adds to it
TABLE_ACC_DEADZONE_G = 0.02  # ignore accelerations below this (sensor noise), in g
TABLE_LEAK = 0.92          # speed decays by this per IMU sample (50 Hz), so drift dies out
TABLE_STILL_G = 0.03       # "arm is still" when acceleration stays below this ...
TABLE_STILL_DPS = 6.0      # ... and rotation below this (deg/s) ...
TABLE_STILL_N = 6          # ... for this many IMU samples (120 ms): speed is reset to 0

# --- Which cursor axes move (both styles; override with --axes) -----------------
AXES = "both"              # "both" = free movement, "x" = left/right only, "y" = up/down only,
                           # "dominant" = only along whichever axis you're moving more right now
AXES_SWITCH = 1.5          # dominant: the other axis must be this many times stronger to take over
AXES_SMOOTH = 0.8          # dominant: how slowly the "which axis" decision follows (0..1)

# --- What each fist does -----------------------------------------------------
# right fist -> left button (hold = drag), left fist -> right button
BUTTON_FOR_SIDE = {"right": "left", "left": "right"}

# --- Camera labels (record only; the live demo never uses the camera) ---------
VISION_CAMERA = "web camera" # which webcam: "" = built-in, a number = that index, text = part of its name,
                           # e.g. "web camera". Override with record --camera. List: record --list-cameras
VISION_LAG_S = 0.05        # webcam frames arrive this much later than EMG (camera vs Bluetooth latency)
VISION_MARGIN_S = 0.08     # leave out EMG this close to a visible open<->close switch
VISION_EARLY_S = 0.3       # a fist may start this early before its cue ...
VISION_SPILL_S = 0.8       # ... and end this long after its cue ends
VISION_ON = 0.6            # how closed (0 = open, 1 = your fist) the hand must look to count as a fist
VISION_OFF = 0.4           # ... and how open again to count as released
VISION_MIN_GAP = 0.25      # open and fist hand shapes must differ at least this much, else ignore camera
VISION_MIN_GAP_INDEX = 0.2 # same for --click index: lifted vs resting index angle (radians, ~11 deg)
VISION_MAX_GAP_S = 0.15    # camera frames older than this don't label an EMG sample
