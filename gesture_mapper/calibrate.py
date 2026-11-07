"""
Same as run.py, except GUI_SENSITIVITY is live-editable while it's running:
type a number and press Enter to set it, try crafting again immediately, type
another number, repeat until it feels right. Type 'q' + Enter to quit — the
last value you set is saved to data/settings.json and becomes the default
for both this script and run.py from then on.

    python calibrate.py

See run.py's GUI_SENSITIVITY docs for what the number means (window-heights
of cursor travel per hand-width of physical hand movement, before the speed-
accel multiplier).

Also logs pinch distance ratios at ~1 Hz — every mode (not just GUI screens),
both hands, all four fingers — for tuning pinch_engage / pinch_release in
data/settings.json. Watch what your natural pinch, your relaxed hand and a
mid-drag pinch actually read on the finger you care about, then set the two
thresholds from those numbers rather than guessing.
"""

import run

if __name__ == "__main__":
    run.main(auto_calibrate=True)
