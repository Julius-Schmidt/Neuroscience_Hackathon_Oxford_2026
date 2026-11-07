"""Per-user storage: every recording and model belongs to one user.

    data/<user>/<side>_<timestamp>.npz     recordings   (myomouse.record)
    models/<user>/fist_<side>.joblib       LDA models   (myomouse.train)
"""

import re
from pathlib import Path

from . import config as C

_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def data_dir(user: str) -> Path:
    return C.DATA_DIR / user


def model_dir(user: str) -> Path:
    return C.MODEL_DIR / user


def add_user_arg(ap, required=True):
    ap.add_argument("--user", required=required, type=_check_name,
                    help="whose recordings / model to use, e.g. --user julius")


def require_user(ap, args, why="to pick whose model to load"):
    """For scripts where --user is only needed sometimes (e.g. not with --check)."""
    if not args.user:
        ap.error(f"--user is required {why}, e.g. --user julius")


def known_users() -> list[str]:
    return sorted(p.name for p in C.MODEL_DIR.glob("*/") if p.is_dir()) if C.MODEL_DIR.exists() else []


def _check_name(name: str) -> str:
    if not _NAME.match(name):
        import argparse
        raise argparse.ArgumentTypeError("use letters, digits, - or _ only")
    return name
