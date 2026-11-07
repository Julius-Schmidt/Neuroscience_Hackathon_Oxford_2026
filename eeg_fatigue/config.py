"""Settings for the fatigue session and its analysis."""

from pathlib import Path

from eeg_switch import config as EEG

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fatigue"                 # recordings (gitignored via data/)
RESULTS_DIR = ROOT / "research" / "eeg_fatigue_results"

# --- Streams --------------------------------------------------------------------
EEG_STREAM = EEG.STREAMS[EEG.SOURCE]                 # same switch as eeg_switch: "sim" | "unicorn"
EEG_CHANNELS = EEG.EEG_CHANNELS                      # Fz C3 Cz C4 Pz PO7 Oz PO8
EEG_SRATE = 250.0
EMG_STREAM = {"name": "MyoEMG", "type": "EMG", "n_channels": 8, "srate": 200.0}
EMG_SRATE = 200.0
MARKER_STREAM = {"name": "FatigueMarkers", "type": "Markers"}

# --- Task -------------------------------------------------------------------------
HAND = "right"                    # squeezing hand; motor cortex on the opposite side
MOTOR_CHANNEL = {"right": "C3", "left": "C4"}
SQUEEZES_PER_BLOCK = 20
SQUEEZE_S = 1.5                   # "squeeze as hard as you can"
RELAX_S = 2.5                     # +- RELAX_JITTER_S, so the brain can't time it exactly
RELAX_JITTER_S = 0.3
LEAD_IN_S = 3.0
BETWEEN_BLOCKS_S = 5.0
FATIGUE_BLOCKS = 5                # after the baseline block, no real breaks
REST_S = 180.0
RECOVERY_BLOCKS = 1
# sanity check
SANITY_EYES_S = 30.0
SANITY_SQUEEZES = 10

# --- EMG onsets (the trigger) --------------------------------------------------------
EMG_HP_HZ = 20.0
EMG_SMOOTH_S = 0.05
ONSET_K_MAD = 6.0                 # threshold = rest median + K * MAD
ONSET_FRACTION = 0.15             # ... or this fraction of the session's 95th pct, whichever is higher
ONSET_MIN_GAP_S = 2.0
ONSET_MIN_DUR_S = 0.3
PEAK_WIN = (0.0, 1.5)             # EMG click peak searched here after onset
CUE_MATCH_S = (-0.5, 1.5)         # onsets must follow a squeeze cue within this window

# --- EEG ------------------------------------------------------------------------------
REFERENCE = "car"                 # "car" (common average of 8 ch) | "none"
EPOCH = (-2.0, 3.0)               # s around EMG onset
BETA = (13.0, 30.0)
MU = (8.0, 12.0)
ERD_BASE = (-1.5, -0.5)
ERD_WIN = (0.0, 1.0)              # power drop measured here
MRCP_LOWPASS = 5.0
MRCP_BASE = (-2.0, -1.5)
MRCP_WIN = (-0.5, 0.0)
ARTIFACT_UV = 150.0               # drop epochs over this peak-to-peak on C3/Cz/C4
MEASURES = ("beta_erd", "mu_erd", "mrcp")
DEFAULT_MEASURE = "beta_erd"

# --- Statistics -----------------------------------------------------------------------
N_BOOT = 500
RISE_MIN = 1.10                   # last fatigue block ratio must exceed this ...
SEED = 0
