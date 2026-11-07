"""
Hand-skeleton icons rendered from the gesture templates themselves.

Every gesture card, mapping row, and onboarding tile draws its icon straight
from the (21, 2) normalized template stored in data/gestures.json — the icon
*is* the recorded pose, so a user's custom gesture gets a real icon for free.

Pinch pseudo-gestures ("pinch_index", ...) render the stored "pinch" template
with the participating fingertips highlighted.
"""

import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap

from gesture_library import load as load_library
import theme

# MediaPipe hand topology
_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm edge
]
_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

_cache: dict[tuple, QPixmap] = {}
_library: dict[str, np.ndarray] | None = None


def reload() -> None:
    """Drop caches after the gesture library changes on disk."""
    global _library
    _library = None
    _cache.clear()


def _get_library() -> dict[str, np.ndarray]:
    global _library
    if _library is None:
        _library = load_library()
    return _library


def skeleton_pixmap(pts: np.ndarray, size: int, line: str = "#9a9aa4",
                    joints: str = "#c9c9d2", accents: dict[int, str] | None = None,
                    line_width: float = 2.0) -> QPixmap:
    """Draw a (21, 2) point set fitted into a size×size pixmap (2x retina).
    accents: landmark index → color, drawn as larger glowing dots."""
    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)

    # Fit with padding, preserve aspect. Template coords: wrist at origin,
    # y down, fingers typically at negative y.
    pad = size * 0.14
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    scale = (size - 2 * pad) / span
    center = (lo + hi) / 2.0
    def to_px(pt) -> QPointF:
        x = (float(pt[0]) - center[0]) * scale + size / 2
        y = (float(pt[1]) - center[1]) * scale + size / 2
        return QPointF(x, y)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    pen = QPen(QColor(line), line_width, Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    for a, b in _BONES:
        p.drawLine(to_px(pts[a]), to_px(pts[b]))

    p.setPen(Qt.PenStyle.NoPen)
    r = max(1.6, size * 0.030)
    p.setBrush(QColor(joints))
    for i in range(21):
        p.drawEllipse(to_px(pts[i]), r, r)

    if accents:
        for idx, color in accents.items():
            c = QColor(color)
            glow = QColor(c)
            glow.setAlpha(70)
            p.setBrush(glow)
            p.drawEllipse(to_px(pts[idx]), r * 3.1, r * 3.1)
            p.setBrush(c)
            p.drawEllipse(to_px(pts[idx]), r * 1.8, r * 1.8)
    p.end()
    return pm


def gesture_pixmap(name: str, size: int = 56, accent_tips: bool = True) -> QPixmap:
    """Icon for a pose gesture or a pinch_* pseudo-gesture."""
    key = (name, size, accent_tips)
    if key in _cache:
        return _cache[key]

    lib = _get_library()
    if name.startswith("pinch_"):
        finger = name.removeprefix("pinch_")
        pts = lib.get("pinch")
        if pts is None:
            pts = next(iter(lib.values()), None)
        if pts is None:
            pm = _unknown_pixmap(size)
        else:
            pm = skeleton_pixmap(
                pts, size,
                accents={_TIPS["thumb"]: theme.ACCENT,
                         _TIPS.get(finger, 8): theme.ACCENT})
    elif name in lib:
        accents = {}
        if accent_tips:
            accents = {i: theme.TEXT for i in ()}  # tips stay neutral by default
        pm = skeleton_pixmap(lib[name], size, accents=accents)
    else:
        pm = _unknown_pixmap(size)

    _cache[key] = pm
    return pm


def _unknown_pixmap(size: int) -> QPixmap:
    return theme.sf_symbol("questionmark.circle", size - 12, theme.TEXT_FAINT)
