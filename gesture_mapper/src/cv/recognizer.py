"""
Template-matching gesture recognizer.

Normalizes incoming landmarks, computes Euclidean distance to each stored template,
returns the closest match if under threshold.

RECOGNITION_THRESHOLD: tune up if too many false positives, down if misses real gestures.
"""

import numpy as np

try:
    from .normalizer import normalize, mirror
    from .gesture_library import load as load_library
except ImportError:
    from normalizer import normalize, mirror
    from gesture_library import load as load_library

RECOGNITION_THRESHOLD = 1.5  # max L2 distance on 42-element normalized vector

_library_cache: dict | None = None


def _get_library() -> dict:
    """Load once and cache for the lifetime of the process."""
    global _library_cache
    if _library_cache is None:
        _library_cache = load_library()
    return _library_cache


def reload_library() -> None:
    """Drop the cache so the next recognize() re-reads data/gestures.json.
    Call after any add/delete — the cache is process-lifetime otherwise."""
    global _library_cache
    _library_cache = None


def recognize(landmarks) -> tuple[str, float] | None:
    """
    landmarks: 21 MediaPipe NormalizedLandmark objects (one hand).
    Tries both the normalized landmarks and their mirror image against each template,
    so gestures recorded on one hand work on the other (ambidextrous by default).
    Returns (gesture_name, confidence 0–1) or None if no match / library empty.
    """
    library = _get_library()
    if not library:
        return None

    pts = normalize(landmarks)
    pts_mirror = mirror(pts)

    vec = pts.flatten()
    vec_mirror = pts_mirror.flatten()

    best_name, best_dist = None, float('inf')
    for name, template in library.items():
        t = template.flatten()
        dist = min(
            float(np.linalg.norm(vec - t)),
            float(np.linalg.norm(vec_mirror - t)),
        )
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist > RECOGNITION_THRESHOLD:
        return None

    confidence = max(0.0, 1.0 - best_dist / RECOGNITION_THRESHOLD)
    return best_name, confidence
