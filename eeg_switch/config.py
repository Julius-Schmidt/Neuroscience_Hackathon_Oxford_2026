"""Every tunable number in one place.

Switching between the simulator and the real headset = change SOURCE below.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "eeg"          # PhysioNet EDFs land here (gitignored)
RESULTS_DIR = ROOT / "research" / "eeg_switch_results"

# --- EEG stream ---------------------------------------------------------------
# The simulator publishes exactly what the detector expects from the headset.
# VERIFY against the real Unicorn LSL app before demo day: open it, then run
#   python -m eeg_switch.listen --list
# and copy its name / type / channel count here. Only the first 8 channels are
# read as EEG, in the order of EEG_CHANNELS.
STREAMS = {
    "sim": {"name": "Unicorn", "type": "EEG", "n_channels": 8, "srate": 250.0,
            "source_id": "eeg_switch_simulator"},
    "unicorn": {"name": "Unicorn", "type": "EEG", "n_channels": 8, "srate": 250.0,
                "source_id": None},
}
SOURCE = "sim"                    # "sim" | "unicorn"  <- the one line to change

# Unicorn Hybrid Black montage (8 EEG channels), in stream order
EEG_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
POSTERIOR = ["Pz", "PO7", "Oz", "PO8"]   # where eyes-closed alpha is strongest
SRATE = 250.0

# --- PhysioNet EEG Motor Movement/Imagery (eegmmidb) --------------------------
RUN_EYES_OPEN = 1
RUN_EYES_CLOSED = 2
# subjects with known recording problems in eegmmidb (sampling rate / length)
EXCLUDE_SUBJECTS = [88, 89, 92, 100, 104]
DEV_SUBJECTS = list(range(1, 41))         # thresholds are tuned on these
TEST_SUBJECTS = list(range(71, 110))      # numbers on the slides come from these

# --- Detector -----------------------------------------------------------------
ALPHA_BAND = (8.0, 12.0)
REF_BAND = (2.0, 30.0)        # for the "rel" feature: alpha / broadband power
FEATURE = "rel"               # "rel" (alpha share of 2-30 Hz) | "abs" (alpha power)
WIN_S = 1.0                   # spectral window length (s)
STEP_S = 0.1                  # one decision every 100 ms
BASELINE_S = 20.0             # eyes-open calibration at start (s)
ON_DB = 3.0                   # alpha this many dB above baseline = "eyes closed"
OFF_DB = 1.0                  # below this = "eyes open" (hysteresis)
DWELL_S = 2.0                 # eyes must stay closed this long to toggle (rejects blinks)
REARM_S = 1.0                 # eyes must be open this long before the next toggle
ARTIFACT_UV = 200.0           # skip windows whose posterior peak-to-peak exceeds this

# --- Intent events (shared format: do not change without telling Dylan) -------
#   {"source": "eeg" | "cv" | "emg", "intent": str, "confidence": 0-1, "timestamp": LSL time}
EVENT_STREAM = {"name": "NeuroIntents", "type": "Markers", "source_id": "neuro_intents"}
INTENT_TOGGLE = "safety_toggle"

# --- Simulator ----------------------------------------------------------------
SIM_CHUNK = 10                # samples per LSL push (40 ms at 250 Hz)
INJECT_CLOSED_S = 4.0         # length of an injected eyes-closed segment (keypress)
