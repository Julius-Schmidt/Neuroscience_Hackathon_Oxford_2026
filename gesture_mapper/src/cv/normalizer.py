"""
Landmark normalization: translation + scale invariant, x/y only.

Translation: subtract wrist (landmark 0) so wrist is at origin.
Scale: divide by wrist→middle-finger-MCP (landmark 9) distance = 1.

Returns (21, 2) float32 array. Hand-agnostic — call once per hand for dual-hand later.
Rotation invariance not implemented; add alignment step here when needed.
"""

import numpy as np


def normalize(landmarks) -> np.ndarray:
    """
    landmarks: iterable of 21 objects with .x, .y attributes (MediaPipe NormalizedLandmark).
    Returns (21, 2) float32 array, translation+scale normalized.
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]  # translate: wrist to origin
    scale = float(np.linalg.norm(pts[9]))  # wrist→middle-MCP distance
    if scale > 1e-6:
        pts /= scale
    return pts


def mirror(pts: np.ndarray) -> np.ndarray:
    """Flip x axis of a normalized (21, 2) array — produces the mirror-image hand."""
    mirrored = pts.copy()
    mirrored[:, 0] *= -1
    return mirrored
