"""
Camera selection by name, using cv2-enumerate-cameras which replicates
OpenCV's internal AVFoundation sort (by uniqueID) to give correct indices.

Prefers built-in camera by default; avoids iPhone/iPad Continuity Camera.
"""

import cv2
from cv2_enumerate_cameras import enumerate_cameras

_PHONE_KEYWORDS = ("iphone", "ipad", "continuity")
_PREFERRED = "FaceTime HD Camera"


def list_cameras() -> list[tuple[int, str]]:
    """Returns [(opencv_index, name), ...] in correct OpenCV order."""
    return [(cam.index, cam.name) for cam in enumerate_cameras(cv2.CAP_AVFOUNDATION)]


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
