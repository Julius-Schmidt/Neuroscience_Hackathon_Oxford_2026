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

# --- Cursor (IMU on the pointer band) ---------------------------------------
POINTER_SIDE = "right"     # which band moves the cursor
GAIN_X = 1800.0            # pixels per radian of forearm yaw
GAIN_Y = 1800.0            # pixels per radian of forearm pitch
INVERT_X = False           # flip if the cursor moves the wrong way left/right
INVERT_Y = False           # flip if the cursor moves the wrong way up/down
DEADZONE_RAD = 0.0015      # ignore tiny per-sample jitter (radians per IMU sample)
SMOOTHING = 0.5            # 0 = no smoothing, closer to 1 = smoother but laggier
FREEZE_MS = 150            # freeze the cursor this long after a press/release

# --- What each fist does -----------------------------------------------------
# right fist -> left button (hold = drag), left fist -> right button
BUTTON_FOR_SIDE = {"right": "left", "left": "right"}

# --- Camera labels (record only; the live demo never uses the camera) ---------
VISION_LAG_S = 0.05        # webcam frames arrive this much later than EMG (camera vs Bluetooth latency)
VISION_MARGIN_S = 0.08     # leave out EMG this close to a visible open<->close switch
VISION_EARLY_S = 0.3       # a fist may start this early before its cue ...
VISION_SPILL_S = 0.8       # ... and end this long after its cue ends
VISION_ON = 0.6            # how closed (0 = open, 1 = your fist) the hand must look to count as a fist
VISION_OFF = 0.4           # ... and how open again to count as released
VISION_MIN_GAP = 0.25      # open and fist hand shapes must differ at least this much, else ignore camera
VISION_MAX_GAP_S = 0.15    # camera frames older than this don't label an EMG sample
