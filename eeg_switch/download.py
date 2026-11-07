"""Download PhysioNet eegmmidb runs 1 (eyes open) and 2 (eyes closed).

    python -m eeg_switch.download              dev + test subjects (~2 x 1.2 MB each)
    python -m eeg_switch.download --subjects 1 2 3
"""

import argparse

from . import config as C
from .data import download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+")
    a = ap.parse_args()
    subs = a.subjects or [s for s in C.DEV_SUBJECTS + C.TEST_SUBJECTS if s not in C.EXCLUDE_SUBJECTS]
    print(f"downloading {len(subs)} subjects to {C.DATA_DIR}")
    download(subs)


if __name__ == "__main__":
    main()
