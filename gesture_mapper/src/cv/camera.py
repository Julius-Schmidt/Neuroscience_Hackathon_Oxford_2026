"""
Camera selection by name, using cv2-enumerate-cameras which replicates
OpenCV's internal AVFoundation sort (by uniqueID) to give correct indices.

Prefers built-in camera by default; avoids iPhone/iPad Continuity Camera.
"""

import sys

import cv2
try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:          # optional on Windows: fall back to plain indices
    enumerate_cameras = None

# the capture backend whose index order we must match
_BACKEND = {"darwin": cv2.CAP_AVFOUNDATION, "win32": cv2.CAP_MSMF}.get(sys.platform, cv2.CAP_V4L2)

_PHONE_KEYWORDS = ("iphone", "ipad", "continuity")
_PREFERRED = "FaceTime HD Camera" if sys.platform == "darwin" else "Integrated"


def list_cameras() -> list[tuple[int, str]]:
    """Returns [(opencv_index, name), ...] in correct OpenCV order."""
    if enumerate_cameras is None:
        return [(0, "default camera")]
    try:
        cams = [(cam.index, cam.name) for cam in enumerate_cameras(_BACKEND)]
    except Exception:
        cams = []
    return cams or [(0, "default camera")]


def index_for_name(preferred: str = _PREFERRED) -> int:
    """
    Returns OpenCV index for the preferred camera name.
    Falls back to first non-phone camera, then 0.
    """
    cameras = list_cameras()

    for idx, name in cameras:
        if preferred.lower() in name.lower():
            return idx

    for idx, name in cameras:
        if not any(kw in name.lower() for kw in _PHONE_KEYWORDS):
            return idx

    print(f"[camera] Could not find a non-phone camera. Available: {cameras}. Defaulting to 0.")
    return 0
