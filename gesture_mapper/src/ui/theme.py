"""
Dark macOS theme: palette constants, the app-wide QSS, SF Symbols loader,
and the NSWindow polish (transparent title bar, dark appearance).

Everything visual that isn't a specific widget lives here so the pages stay
about structure, not styling.
"""

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap

# ── Palette (Apple dark-mode system colors) ──────────────────────────────────
BG          = "#1b1b1e"   # window
SIDEBAR     = "#151517"
CARD        = "#252528"
CARD_HOVER  = "#2c2c30"
CARD_BORDER = "#343439"
FIELD       = "#1f1f22"
TEXT        = "#f5f5f7"
TEXT_DIM    = "#9d9da6"
TEXT_FAINT  = "#6c6c74"
ACCENT      = "#0a84ff"
ACCENT_DIM  = "#0a84ff"
GREEN       = "#30d158"
ORANGE      = "#ff9f0a"
RED         = "#ff453a"
PURPLE      = "#bf5af2"
TEAL        = "#64d2ff"

HAND_LEFT_COLOR  = TEAL     # left hand accents (commands)
HAND_RIGHT_COLOR = PURPLE   # right hand accents (pointer)


def app_font(size: int = 13, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont()          # default family = macOS system font (SF Pro)
    f.setPointSize(size)
    f.setWeight(weight)
    return f


QSS = f"""
* {{
    color: {TEXT};
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget#sidebar {{ background: {SIDEBAR}; }}
QWidget#page {{ background: {BG}; }}

QLabel {{ background: transparent; }}
QLabel[class="h1"] {{ font-size: 26px; font-weight: 700; }}
QLabel[class="h2"] {{ font-size: 17px; font-weight: 600; }}
QLabel[class="dim"] {{ color: {TEXT_DIM}; }}
QLabel[class="faint"] {{ color: {TEXT_FAINT}; font-size: 12px; }}

QFrame[class="card"] {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
}}

QPushButton {{
    background: {CARD};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{ background: {CARD_HOVER}; }}
QPushButton:pressed {{ background: #232326; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; }}

QPushButton[class="primary"] {{
    background: {ACCENT};
    border: none;
    color: white;
    font-weight: 600;
    padding: 8px 20px;
}}
QPushButton[class="primary"]:hover {{ background: #2f97ff; }}
QPushButton[class="primary"]:pressed {{ background: #0870d8; }}
QPushButton[class="primary"]:disabled {{ background: #26262a; color: {TEXT_FAINT}; }}

QPushButton[class="danger"] {{
    background: transparent; border: 1px solid {CARD_BORDER}; color: {RED};
}}
QPushButton[class="danger"]:hover {{ background: rgba(255,69,58,0.12); }}

QPushButton[class="ghost"] {{
    background: transparent; border: none; color: {TEXT_DIM};
}}
QPushButton[class="ghost"]:hover {{ color: {TEXT}; background: {CARD_HOVER}; }}

QLineEdit {{
    background: {FIELD};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{ border: 1px solid {ACCENT}; }}

QListWidget {{
    background: {FIELD};
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    outline: none;
    padding: 4px;
}}
QListWidget::item {{
    border-radius: 6px;
    padding: 7px 10px;
    color: {TEXT};
}}
QListWidget::item:hover {{ background: {CARD_HOVER}; }}
QListWidget::item:selected {{ background: {ACCENT}; color: white; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 8px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #3d3d44; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #4d4d55; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}

QSlider::groove:horizontal {{
    height: 4px; background: #3a3a40; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: white; width: 18px; height: 18px;
    margin: -7px 0; border-radius: 9px;
}}

QToolTip {{
    background: {CARD}; color: {TEXT};
    border: 1px solid {CARD_BORDER}; padding: 4px 8px; border-radius: 6px;
}}
"""


# ── SF Symbols ───────────────────────────────────────────────────────────────
# Real macOS symbol images via AppKit, tinted to the requested color. Falls
# back to a plain text glyph if anything in the AppKit path is unavailable —
# the app must never fail to launch over an icon.

_FALLBACK = {
    "house.fill": "⌂", "hand.raised.fill": "✋", "arrow.left.arrow.right": "⇄",
    "gearshape.fill": "⚙", "record.circle": "◉", "play.fill": "▶",
    "stop.fill": "■", "plus": "+", "trash": "✕", "checkmark.circle.fill": "✓",
    "xmark.circle.fill": "✕", "camera.fill": "▣", "sparkles": "✦",
    "cursorarrow.motionlines": "➤", "keyboard.fill": "⌨",
    "chevron.right": "›", "chevron.left": "‹", "wand.and.stars": "✦",
    "bolt.fill": "⚡", "hand.point.up.left.fill": "☝",
    "accessibility": "◎", "arrow.counterclockwise": "↺",
    "cube.fill": "◆", "magnifyingglass": "◌", "menubar.rectangle": "▭",
    "hand.pinch.fill": "🤏", "questionmark.circle": "?",
}

_symbol_cache: dict[tuple, QPixmap] = {}


def sf_symbol(name: str, size: int = 18, color: str = TEXT,
              weight: str = "regular") -> QPixmap:
    key = (name, size, color, weight)
    if key in _symbol_cache:
        return _symbol_cache[key]
    pm = _render_symbol(name, size, color, weight)
    if pm is None:
        pm = _render_fallback(name, size, color)
    _symbol_cache[key] = pm
    return pm


def sf_icon(name: str, size: int = 18, color: str = TEXT,
            weight: str = "regular") -> QIcon:
    return QIcon(sf_symbol(name, size, color, weight))


def _render_symbol(name: str, size: int, color: str, weight: str) -> QPixmap | None:
    try:
        import AppKit
        img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            name, None)
        if img is None:
            return None
        w = {"regular": AppKit.NSFontWeightRegular,
             "medium": AppKit.NSFontWeightMedium,
             "semibold": AppKit.NSFontWeightSemibold,
             "bold": AppKit.NSFontWeightBold}.get(weight, 0.0)
        # Render at 2x for retina crispness, scale down in Qt.
        conf = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
            size * 2, w)
        img = img.imageWithSymbolConfiguration_(conf)
        tiff = img.TIFFRepresentation()
        if tiff is None:
            return None
        # Qt here has no TIFF decoder — round-trip through PNG via AppKit.
        rep = AppKit.NSBitmapImageRep.imageRepWithData_(tiff)
        if rep is None:
            return None
        png = rep.representationUsingType_properties_(
            AppKit.NSBitmapImageFileTypePNG, {})
        if png is None:
            return None
        qimg = QImage.fromData(QByteArray(bytes(png)))
        if qimg.isNull():
            return None
        # Tint: symbols come out black — repaint alpha with the target color.
        tinted = QImage(qimg.size(), QImage.Format.Format_ARGB32_Premultiplied)
        tinted.fill(Qt.GlobalColor.transparent)
        p = QPainter(tinted)
        p.drawImage(0, 0, qimg)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(tinted.rect(), QColor(color))
        p.end()
        pm = QPixmap.fromImage(tinted.scaledToHeight(
            size * 2, Qt.TransformationMode.SmoothTransformation))
        pm.setDevicePixelRatio(2.0)
        return pm
    except Exception:
        return None


def _draw_vector(name: str, p: QPainter, color: str) -> bool:
    """Simple vector icons on a 24x24 grid, for when SF Symbols aren't available
    (Windows / Linux). Returns False for names it doesn't know."""
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QBrush, QPainterPath, QPen
    c = QColor(color)
    fill = QBrush(c)
    pen = QPen(c, 2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    no_pen = Qt.PenStyle.NoPen
    if name == "house.fill":
        path = QPainterPath()
        path.moveTo(12, 3); path.lineTo(22, 11.5); path.lineTo(19.5, 11.5); path.lineTo(19.5, 21)
        path.lineTo(14.5, 21); path.lineTo(14.5, 15); path.lineTo(9.5, 15); path.lineTo(9.5, 21)
        path.lineTo(4.5, 21); path.lineTo(4.5, 11.5); path.lineTo(2, 11.5); path.closeSubpath()
        p.setPen(no_pen); p.setBrush(fill); p.drawPath(path)
    elif name in ("hand.raised.fill", "hand.point.up.left.fill", "hand.pinch.fill"):
        p.setPen(no_pen); p.setBrush(fill)
        for x, top in ((6.2, 6), (9.4, 3), (12.6, 2.5), (15.8, 4.5)):
            p.drawRoundedRect(QRectF(x, top, 2.9, 11 - top + 3), 1.45, 1.45)
        p.drawRoundedRect(QRectF(6.2, 11, 12.5, 10.5), 4, 4)
        path = QPainterPath()
        path.moveTo(18.2, 13.5); path.lineTo(20.8, 10.2); path.quadTo(22.4, 9.6, 22, 11.4)
        path.lineTo(18.6, 17.5); path.closeSubpath()
        p.drawPath(path)
    elif name == "arrow.left.arrow.right":
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(4, 8), QPointF(20, 8)); p.drawLine(QPointF(16, 4), QPointF(20, 8))
        p.drawLine(QPointF(16, 12), QPointF(20, 8))
        p.drawLine(QPointF(4, 16), QPointF(20, 16)); p.drawLine(QPointF(8, 12), QPointF(4, 16))
        p.drawLine(QPointF(8, 20), QPointF(4, 16))
    elif name == "square.stack.3d.up.fill":     # Blocks: two stacked puzzle blocks
        p.setPen(no_pen); p.setBrush(fill)
        for y in (3.5, 13):
            path = QPainterPath()
            path.addRoundedRect(QRectF(3, y, 18, 7.5), 2, 2)
            notch = QPainterPath()
            notch.addRect(QRectF(7, y, 4, 2))
            p.drawPath(path.subtracted(notch))
            p.drawRect(QRectF(7, y + 7.5, 4, 1.6))
    elif name == "gearshape.fill":
        p.setPen(no_pen); p.setBrush(fill)
        p.save(); p.translate(12, 12)
        for k in range(8):
            p.drawRoundedRect(QRectF(-1.8, -10.5, 3.6, 5), 1, 1)
            p.rotate(45)
        p.restore()
        ring = QPainterPath(); ring.addEllipse(QPointF(12, 12), 7.2, 7.2)
        hole = QPainterPath(); hole.addEllipse(QPointF(12, 12), 3, 3)
        p.drawPath(ring.subtracted(hole))
    elif name == "play.fill":
        path = QPainterPath(); path.moveTo(7, 4); path.lineTo(20, 12); path.lineTo(7, 20); path.closeSubpath()
        p.setPen(no_pen); p.setBrush(fill); p.drawPath(path)
    elif name == "stop.fill":
        p.setPen(no_pen); p.setBrush(fill); p.drawRoundedRect(QRectF(5, 5, 14, 14), 2.5, 2.5)
    elif name == "plus":
        thick = QPen(c, 2.6); thick.setCapStyle(Qt.PenCapStyle.RoundCap); p.setPen(thick)
        p.drawLine(QPointF(12, 5), QPointF(12, 19)); p.drawLine(QPointF(5, 12), QPointF(19, 12))
    elif name in ("trash", "xmark.circle.fill"):
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
        if name == "trash":
            p.drawLine(QPointF(4, 6.5), QPointF(20, 6.5)); p.drawLine(QPointF(9.5, 6.5), QPointF(10.5, 3.5))
            p.drawLine(QPointF(14.5, 6.5), QPointF(13.5, 3.5)); p.drawLine(QPointF(10.5, 3.5), QPointF(13.5, 3.5))
            p.drawRoundedRect(QRectF(6, 6.5, 12, 14), 2, 2)
        else:
            p.drawEllipse(QPointF(12, 12), 9, 9)
            p.drawLine(QPointF(9, 9), QPointF(15, 15)); p.drawLine(QPointF(15, 9), QPointF(9, 15))
    elif name == "record.circle":
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush); p.drawEllipse(QPointF(12, 12), 9, 9)
        p.setPen(no_pen); p.setBrush(fill); p.drawEllipse(QPointF(12, 12), 4.5, 4.5)
    elif name == "camera.fill":
        p.setPen(no_pen); p.setBrush(fill)
        body = QPainterPath(); body.addRoundedRect(QRectF(2.5, 7, 19, 13), 3, 3)
        body.addRect(QRectF(8, 4.5, 8, 3))
        lens = QPainterPath(); lens.addEllipse(QPointF(12, 13.5), 3.6, 3.6)
        p.drawPath(body.subtracted(lens))
    elif name in ("checkmark.circle.fill",):
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush); p.drawEllipse(QPointF(12, 12), 9, 9)
        p.drawLine(QPointF(8, 12.5), QPointF(11, 15.5)); p.drawLine(QPointF(11, 15.5), QPointF(16.5, 9))
    elif name == "questionmark.circle":
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush); p.drawEllipse(QPointF(12, 12), 9, 9)
        f = app_font(11, QFont.Weight.Bold); p.setFont(f)
        p.drawText(QRectF(0, 0, 24, 24), Qt.AlignmentFlag.AlignCenter, "?")
    elif name == "arrow.right":
        p.setPen(pen)
        p.drawLine(QPointF(4, 12), QPointF(20, 12)); p.drawLine(QPointF(14, 6), QPointF(20, 12))
        p.drawLine(QPointF(14, 18), QPointF(20, 12))
    elif name in ("chevron.right", "chevron.left"):
        p.setPen(pen)
        a_, b_ = (9, 15) if name == "chevron.right" else (15, 9)
        p.drawLine(QPointF(a_, 5), QPointF(b_, 12)); p.drawLine(QPointF(b_, 12), QPointF(a_, 19))
    else:
        return False
    return True


def _render_fallback(name: str, size: int, color: str) -> QPixmap:
    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / 24.0, size / 24.0)          # logical units -> 24x24 design grid
    try:
        ok = _draw_vector(name, p, color)
    except Exception:
        ok = False
    p.end()
    if ok:
        return pm
    glyph = _FALLBACK.get(name, "•")
    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setPen(QColor(color))
    f = app_font(size - 2)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    p.end()
    return pm


# ── NSWindow polish ──────────────────────────────────────────────────────────

def polish_macos_window(widget) -> None:
    """Dark appearance + transparent, full-bleed title bar on the widget's
    NSWindow. Safe no-op if the AppKit bridge misbehaves."""
    try:
        from PySide6.QtGui import QGuiApplication
        if QGuiApplication.platformName() != "cocoa":
            return   # winId() is only an NSView on the cocoa platform —
                     # anything else segfaults past the except below
        import objc
        import AppKit
        view = objc.objc_object(c_void_p=int(widget.winId()))
        window = view.window()
        if window is None:
            return
        window.setAppearance_(
            AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua))
        window.setTitlebarAppearsTransparent_(True)
        window.setTitleVisibility_(AppKit.NSWindowTitleHidden)
        window.setStyleMask_(window.styleMask() | AppKit.NSWindowStyleMaskFullSizeContentView)
        window.setBackgroundColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
            0x1b / 255, 0x1b / 255, 0x1e / 255, 1.0))
    except Exception:
        pass
