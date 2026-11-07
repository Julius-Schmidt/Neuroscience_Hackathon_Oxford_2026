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


def _render_fallback(name: str, size: int, color: str) -> QPixmap:
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
