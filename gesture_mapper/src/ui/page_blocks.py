"""
Blocks: a Scratch-style editor for the general rule engine (mapping-engine/rules.py).

Any input -> any action. Each rule is a stack of blocks:

    [when  Myo  right.fist  is pressed]      yellow "hat" - the event
    [only if mode is  normal]                green - optional conditions
    [hold mouse button  left]                blue / orange / purple - actions, top to bottom

Drag blocks from the palette onto a rule, or CLICK a palette block to add it to the selected
rule (no dragging needed - matters for the users this is for). "Try" fires the rule's event
by hand; rules light up when they fire for real. Everything saves to data/rules.json as you
edit, and the rules run live (Run rules) from the Myo bands, BCI/EEG markers (LSL) and the
camera at the same time.
"""

import os
import sys

from PySide6.QtCore import QMimeData, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtCore import QRectF
from PySide6.QtGui import QBrush, QColor, QDrag, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("../sources", "../mapping-engine", "../os-integration"):
    p = os.path.normpath(os.path.join(HERE, sub))
    if p not in sys.path:
        sys.path.insert(0, p)

import theme  # noqa: E402
import rules as R  # noqa: E402
from widgets import make_label  # noqa: E402

MIME = "application/x-neuratect-block"

# category colours from the app's own (Apple dark-mode) palette
C_EVENT, C_COND, C_INPUT, C_CONTROL, C_APP = theme.ORANGE, theme.GREEN, theme.ACCENT, theme.TEAL, theme.PURPLE

SOURCE_LABELS = {"myo": "Myo band", "camera": "camera pose", "lsl": "LSL marker", "keyboard": "test key"}
TRIGGER_SHORT = {"press": "starts", "release": "ends", "hold": "is held", "value": "moves"}

PALETTE = [
    ("Events", C_EVENT, [("hat:myo", "when Myo gesture"), ("hat:camera", "when camera pose"),
                         ("hat:lsl", "when LSL marker (EEG / BCI / apps)"), ("hat:keyboard", "when test key")]),
    ("Only if", C_COND, [("cond:mode", "only if mode is"), ("cond:not_mode", "only if mode is not"),
                         ("cond:context", "only if app in front is")]),
    ("Keys & mouse", C_INPUT, [("act:key_tap", "press key"), ("act:key_hold", "hold key"),
                               ("act:shortcut", "press shortcut"), ("act:type_text", "type"),
                               ("act:mouse_click", "click mouse"), ("act:mouse_hold", "hold mouse button"),
                               ("act:mouse_move", "move mouse with value"), ("act:scroll", "scroll")]),
    ("Control", C_CONTROL, [("act:claw_step", "claw: next step"), ("act:claw_drive", "claw: glide while held"),
                            ("act:claw_cancel", "claw: cancel"),
                            ("act:set_mode", "set mode to"), ("act:toggle_mode", "toggle mode"),
                            ("act:wait", "wait")]),
    ("Apps & feedback", C_APP, [("act:open", "open app / website"), ("act:command", "built-in command"),
                                ("act:command_hold", "hold built-in command"),
                                ("act:sound", "play sound"), ("act:say", "show message")]),
]
ACTION_COLOR = {k.split(":")[1]: c for _cat, c, items in PALETTE for k, _ in items if k.startswith("act:")}

KEY_NAMES = ["space", "enter", "esc", "tab", "backspace", "up", "down", "left", "right",
             "w", "a", "s", "d", "e", "q", "shift", "ctrl", "alt"] + list("bcfghijklmnoprtuvxyz") + list("0123456789")


def _darker(hex_color: str, f=130) -> str:
    return QColor(hex_color).darker(f).name()


HAT_TOP = 0       # kept for layout maths; modern blocks have no dome / notch
NOTCH = 0


def _mix(a: str, b: str, t: float) -> QColor:
    """Blend colour a towards b by t (0..1)."""
    ca, cb = QColor(a), QColor(b)
    return QColor(int(ca.red() + (cb.red() - ca.red()) * t), int(ca.green() + (cb.green() - ca.green()) * t),
                  int(ca.blue() + (cb.blue() - ca.blue()) * t))


def paint_block(widget, color: str, hat: bool = False) -> None:
    """A block = a dark card tinted with its category colour and a coloured accent bar on the
    left. 'when' blocks get a stronger tint so a rule's trigger stands out."""
    p = QPainter(widget)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(0.5, 0.5, widget.width() - 1, widget.height() - 1)
    path = QPainterPath()
    path.addRoundedRect(rect, 10, 10)
    p.fillPath(path, _mix(theme.CARD, color, 0.26 if hat else 0.15))
    p.setPen(QPen(_mix(theme.CARD_BORDER, color, 0.55 if hat else 0.4), 1.0))
    p.drawPath(path)
    p.setClipPath(path)
    p.fillRect(QRectF(0, 0, 5 if hat else 4, widget.height()), QColor(color))
    p.end()


def _block_qss(color: str) -> str:
    field = (f"background: {theme.FIELD}; color: {theme.TEXT}; border: 1px solid {theme.CARD_BORDER};"
             f" border-radius: 7px; padding: 3px 8px; min-height: 20px;")
    return (f"QFrame#block {{ background: transparent; border: none; }}"
            f"QFrame#block QLabel {{ color: {theme.TEXT}; font-weight: 600; background: transparent; }}"
            f"QFrame#block QComboBox, QFrame#block QLineEdit, QFrame#block QSpinBox, QFrame#block QDoubleSpinBox {{ {field} }}"
            f"QFrame#block QComboBox:hover, QFrame#block QLineEdit:focus {{ border: 1px solid {color}; }}"
            f"QFrame#block QComboBox QAbstractItemView {{ background: {theme.CARD}; color: {theme.TEXT};"
            f" selection-background-color: {color}; }}"
            f"QFrame#block QPushButton {{ background: transparent; border: none; color: {theme.TEXT_FAINT};"
            f" padding: 0 4px; font-weight: 700; }}"
            f"QFrame#block QPushButton:hover {{ color: {theme.TEXT}; }}")


class _Bridge(QObject):
    fired = Signal(str)
    event = Signal(str)
    claw = Signal(dict)


# ── one block ────────────────────────────────────────────────────────────────
class BlockWidget(QFrame):
    """A coloured block editing one dict in place (an action or a condition)."""
    changed = Signal()
    remove = Signal(object)
    move = Signal(object, int)

    def __init__(self, data: dict, kind: str, color: str, parent=None):
        super().__init__(parent)
        self.data, self.kind, self.color = data, kind, color
        self.setObjectName("block")
        self.setStyleSheet(_block_qss(color))
        self.setMinimumHeight(36)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 5, 6, 5)
        lay.setSpacing(6)
        self._build(lay)
        lay.addStretch()
        for txt, d in (("↑", -1), ("↓", 1)):
            b = QPushButton(txt)
            b.setToolTip("move up" if d < 0 else "move down")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, d=d: self.move.emit(self, d))
            lay.addWidget(b)
        rm = QPushButton("✕")
        rm.setToolTip("remove block")
        rm.setCursor(Qt.CursorShape.PointingHandCursor)
        rm.clicked.connect(lambda: self.remove.emit(self))
        lay.addWidget(rm)

    def paintEvent(self, e):
        paint_block(self, self.color)

    def _lbl(self, lay, text):
        lay.addWidget(QLabel(text))

    def _combo(self, lay, field, options, editable=False, width=None):
        cb = QComboBox()
        cb.setEditable(editable)
        cb.addItems([str(o) for o in options])
        cur = str(self.data.get(field, options[0] if options else ""))
        if cb.findText(cur) < 0:
            cb.addItem(cur)
        cb.setCurrentText(cur)
        if width:
            cb.setMinimumWidth(width)
        cb.currentTextChanged.connect(lambda v: self._set(field, v))
        lay.addWidget(cb)
        return cb

    def _line(self, lay, field, width=140):
        le = QLineEdit(str(self.data.get(field, "")))
        le.setMinimumWidth(width)
        le.textChanged.connect(lambda v: self._set(field, v))
        lay.addWidget(le)
        return le

    def _num(self, lay, field, lo, hi, step, decimals=1):
        sb = QDoubleSpinBox() if decimals else QSpinBox()
        if decimals:
            sb.setDecimals(decimals)
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setValue(float(self.data.get(field, 0)) if decimals else int(self.data.get(field, 0)))
        sb.valueChanged.connect(lambda v: self._set(field, v))
        lay.addWidget(sb)
        return sb

    def _check(self, lay, field, text):
        cb = QCheckBox(text)
        cb.setChecked(bool(self.data.get(field, False)))
        cb.setStyleSheet(f"QCheckBox {{ color: {theme.TEXT}; font-weight: 600; background: transparent; spacing: 6px; }}"
                         f"QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px;"
                         f" border: 1px solid {theme.TEXT_DIM}; background: {theme.FIELD}; }}"
                         f"QCheckBox::indicator:checked {{ background: {theme.ACCENT}; border: 1px solid {theme.ACCENT}; }}")
        cb.toggled.connect(lambda v: self._set(field, bool(v)))
        lay.addWidget(cb)
        return cb

    def _set(self, field, value):
        self.data[field] = value
        self.changed.emit()

    def _build(self, lay):
        t = self.data.get("type")
        if self.kind == "cond":
            self._lbl(lay, R.CONDITIONS.get(t, {}).get("label", t) if t in R.CONDITIONS else t)
            if t == "context":
                self._combo(lay, "value", ["default", "minecraft", "spotlight"], editable=True, width=110)
            else:
                self._combo(lay, "value", ["normal", "paused", "typing", "game"], editable=True, width=100)
            return
        label = R.ACTIONS.get(t, {}).get("label", t)
        if t in ("key_tap", "key_hold"):
            self._lbl(lay, label)
            self._combo(lay, "key", KEY_NAMES, editable=True, width=90)
        elif t in ("mouse_click", "mouse_hold"):
            self._lbl(lay, label)
            self._combo(lay, "button", ["left", "right"])
        elif t == "mouse_move":
            self._lbl(lay, "move mouse with value  x")
            self._num(lay, "gain", 0.1, 10.0, 0.1)
            self._check(lay, "flip_x", "flip left/right")
            self._check(lay, "flip_y", "flip up/down")
        elif t == "scroll":
            self._lbl(lay, "scroll")
            self._num(lay, "amount", -20, 20, 1, decimals=0)
            self._lbl(lay, "(+ up, − down)")
        elif t == "type_text":
            self._lbl(lay, "type")
            self._line(lay, "text")
        elif t == "shortcut":
            self._lbl(lay, "press shortcut")
            self._combo(lay, "keys", ["ctrl+c", "ctrl+v", "ctrl+z", "alt+tab", "ctrl+s", "cmd+space"],
                        editable=True, width=110)
        elif t in ("command", "command_hold"):
            self._lbl(lay, "built-in command" if t == "command" else "hold built-in command")
            try:
                import commands as C
                names = sorted(C.ALL_COMMANDS)
            except Exception:
                names = ["OpenSpotlight"]
            self._combo(lay, "name", names, width=130)
        elif t == "open":
            self._lbl(lay, "open")
            self._line(lay, "target", 220)
        elif t == "set_mode":
            self._lbl(lay, "set mode to")
            self._combo(lay, "value", ["normal", "paused", "typing", "game"], editable=True, width=100)
        elif t == "toggle_mode":
            self._lbl(lay, "toggle mode between")
            self._combo(lay, "a", ["normal", "typing", "game"], editable=True, width=90)
            self._lbl(lay, "and")
            self._combo(lay, "b", ["paused", "typing", "game"], editable=True, width=90)
        elif t == "wait":
            self._lbl(lay, "wait")
            self._num(lay, "seconds", 0.0, 5.0, 0.1)
            self._lbl(lay, "seconds")
        elif t == "sound":
            self._lbl(lay, "play sound")
            self._combo(lay, "name", ["pop", "mypop", "mypop_bass", "click", "ee"], editable=True, width=100)
        elif t == "claw_step":
            self._lbl(lay, "claw: next step, then")
            self._combo(lay, "button", ["left", "right", "double"], width=80)
            self._lbl(lay, "click · sweep")
            self._num(lay, "sweep_s", 1.0, 10.0, 0.5)
            self._lbl(lay, "s · reaction")
            self._num(lay, "lag_ms", 0, 400, 10, decimals=0)
            self._lbl(lay, "ms")
            self._check(lay, "refine", "fine-tune pass")
            self._check(lay, "restart", "keep scanning")
        elif t == "claw_drive":
            self._lbl(lay, "claw: glide along")
            self._combo(lay, "axis", ["x", "y"], width=60)
            self._lbl(lay, "while held · tap =")
            self._combo(lay, "button", ["left", "right", "double"], width=80)
            self._lbl(lay, "click")
        elif t == "claw_cancel":
            self._lbl(lay, "claw: cancel")
        elif t == "say":
            self._lbl(lay, "show message")
            self._line(lay, "text")
        else:
            self._lbl(lay, f"{t} (unknown block)")


class HatBlock(QFrame):
    changed = Signal()

    def __init__(self, when: dict, parent=None):
        super().__init__(parent)
        self.when = when
        self.setObjectName("block")
        self.setStyleSheet(_block_qss(C_EVENT))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 8, 10, 8)
        lay.setSpacing(6)
        lay.addWidget(QLabel("when"))
        self.src = QComboBox()
        for k, v in SOURCE_LABELS.items():
            self.src.addItem(v, k)
        self.src.setCurrentIndex(max(0, list(SOURCE_LABELS).index(when.get("source", "myo"))
                                     if when.get("source") in SOURCE_LABELS else 0))
        lay.addWidget(self.src)
        self.ev = QComboBox()
        self.ev.setEditable(True)
        self.ev.setMinimumWidth(150)
        lay.addWidget(self.ev)
        self.trig = QComboBox()
        for k, v in TRIGGER_SHORT.items():
            self.trig.addItem(v, k)
        self.trig.setCurrentIndex(list(TRIGGER_SHORT).index(when.get("trigger", "press"))
                                  if when.get("trigger") in TRIGGER_SHORT else 0)
        lay.addWidget(self.trig)
        lay.addStretch()
        self._fill_events()
        self.src.currentIndexChanged.connect(self._src_changed)
        self.ev.currentTextChanged.connect(self._ev_changed)
        self.trig.currentIndexChanged.connect(lambda _: self._set("trigger", self.trig.currentData()))

    def paintEvent(self, e):
        paint_block(self, C_EVENT, hat=True)

    def _fill_events(self):
        cur = self.when.get("name", "")
        self.ev.blockSignals(True)
        self.ev.clear()
        self.ev.addItems(R.SOURCE_EVENTS.get(self.src.currentData(), []))
        if cur and self.ev.findText(cur) < 0:
            self.ev.addItem(cur)
        self.ev.setCurrentText(cur or self.ev.itemText(0))
        self.ev.blockSignals(False)
        if not cur:
            self.when["name"] = self.ev.currentText()

    def _src_changed(self, _):
        self.when["source"] = self.src.currentData()
        self.when["name"] = ""
        self._fill_events()
        if self.when["name"].endswith(".pointer") and self.when.get("trigger") != "value":
            self.trig.setCurrentIndex(list(TRIGGER_SHORT).index("value"))
        self.changed.emit()

    def _ev_changed(self, v):
        self.when["name"] = v
        if v.endswith(".pointer") and self.when.get("trigger") != "value":
            self.trig.setCurrentIndex(list(TRIGGER_SHORT).index("value"))
        self.changed.emit()

    def _set(self, k, v):
        self.when[k] = v
        self.changed.emit()


# ── compact text for a rule (collapsed view) and grouping ───────────────────
CONTEXT_LABELS = {"default": "Desktop", "spotlight": "Search / Spotlight", "minecraft": "Minecraft"}
SOURCE_ORDER = ["myo", "lsl", "camera", "keyboard"]


def _pretty(name: str) -> str:
    return str(name).replace(".", " ").replace("_", " ")


def _cmd_label(name) -> str:
    try:
        import labels
        return labels.command_label(str(name))
    except Exception:
        return str(name)


def action_text(a: dict) -> str:
    t = a.get("type")
    return {
        "key_tap": lambda: f"press {a.get('key')}",
        "key_hold": lambda: f"hold {a.get('key')}",
        "shortcut": lambda: f"press {a.get('keys')}",
        "type_text": lambda: f"type \u201c{a.get('text')}\u201d",
        "mouse_click": lambda: f"{a.get('button')} click",
        "mouse_hold": lambda: f"hold {a.get('button')} button",
        "mouse_move": lambda: f"move mouse \u00d7{a.get('gain')}"
                              + (" (flipped " + "/".join(k for k, f in (("left-right", "flip_x"), ("up-down", "flip_y"))
                                                         if a.get(f)) + ")" if a.get("flip_x") or a.get("flip_y") else ""),
        "scroll": lambda: f"scroll {a.get('amount')}",
        "command": lambda: _cmd_label(a.get("name")),
        "command_hold": lambda: f"hold {_cmd_label(a.get('name'))}",
        "open": lambda: f"open {a.get('target')}",
        "set_mode": lambda: f"mode \u2192 {a.get('value')}",
        "toggle_mode": lambda: f"toggle mode {a.get('a')}/{a.get('b')}",
        "wait": lambda: f"wait {a.get('seconds')} s",
        "sound": lambda: f"sound {a.get('name')}",
        "say": lambda: f"say \u201c{a.get('text')}\u201d",
        "claw_step": lambda: f"claw step ({a.get('button')} click)",
        "claw_cancel": lambda: "cancel claw",
        "claw_drive": lambda: f"glide {'left-right' if a.get('axis') == 'x' else 'up-down'} (tap = click)",
    }.get(t, lambda: str(t))()


def rule_context(rule: dict):
    return next((c.get("value") for c in rule.get("conditions", []) if c.get("type") == "context"), None)


def group_key(rule: dict):
    src = rule.get("when", {}).get("source", "myo")
    return (src, rule_context(rule) if src == "camera" else None)


def group_title(key) -> str:
    src, ctx = key
    base = {"myo": "Myo bands", "lsl": "LSL markers (EEG / BCI / other apps)", "camera": "Camera poses",
            "keyboard": "Test keys"}.get(src, src)
    return f"{base}  \u00b7  {CONTEXT_LABELS.get(ctx, ctx)}" if ctx else (
        f"{base}  \u00b7  any app" if src == "camera" else base)


def group_sort(key):
    src, ctx = key
    order = ["default", "spotlight", "minecraft"]
    return (SOURCE_ORDER.index(src) if src in SOURCE_ORDER else 9,
            -1 if ctx is None else (order.index(ctx) if ctx in order else 5), str(ctx))


def rule_summary(rule: dict, hide_context=True):
    """(when text, do text) for the collapsed one-line view."""
    w = rule.get("when", {})
    when = f"{_pretty(w.get('name', ''))}  <span style='color:{theme.TEXT_FAINT}'>" \
           f"{TRIGGER_SHORT.get(w.get('trigger', 'press'), '')}</span>"
    conds = [c for c in rule.get("conditions", []) if not (hide_context and c.get("type") == "context")]
    if conds:
        when += "  <span style='color:%s'>if %s</span>" % (
            C_COND, ", ".join(f"{R.CONDITIONS.get(c['type'], {}).get('label', c['type'])} {c.get('value')}"
                              for c in conds))
    acts = ", ".join(action_text(a) for a in rule.get("actions", [])) or \
        f"<span style='color:{theme.TEXT_FAINT}'>nothing yet</span>"
    return when, acts


class GroupHeader(QPushButton):
    def __init__(self, title: str, count: int, open_: bool):
        super().__init__()
        self.title_text, self.count = title, count
        self.setCheckable(True)
        self.setChecked(open_)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"QPushButton {{ background: transparent; border: none; text-align: left;"
                           f" color: {theme.TEXT_DIM}; font-size: 13px; font-weight: 700; padding: 6px 2px; }}"
                           f"QPushButton:hover {{ color: {theme.TEXT}; }}")
        self.toggled.connect(lambda _: self._text())
        self._text()

    def _text(self):
        arrow = "\u25be" if self.isChecked() else "\u25b8"
        self.setText(f"{arrow}  {self.title_text}   ({self.count})")


# ── one rule ─────────────────────────────────────────────────────────────────
class RuleCard(QFrame):
    changed = Signal()
    selected = Signal(object)
    delete = Signal(object)
    try_it = Signal(object)

    def __init__(self, rule: dict, parent=None):
        super().__init__(parent)
        self.rule = rule
        self.setAcceptDrops(True)
        self._is_selected = False
        self._flash = 0
        self.setObjectName("rulecard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 6, 10, 6)
        lay.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.enabled = QCheckBox()
        self.enabled.setChecked(rule.get("enabled", True))
        self.enabled.setToolTip("rule on / off")
        self.enabled.toggled.connect(self._toggle)
        head.addWidget(self.enabled)
        self.title = QLineEdit(rule.get("title", ""))
        self.title.setPlaceholderText("name this rule (optional)")
        self.title.textChanged.connect(lambda v: (self.rule.__setitem__("title", v), self.changed.emit()))
        head.addWidget(self.title, stretch=1)
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setStyleSheet(f"color: {theme.TEXT}; font-size: 13px;")
        head.addWidget(self.summary, stretch=1)
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setProperty("class", "ghost")
        self.edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_btn.clicked.connect(lambda: self.set_expanded(not self.expanded))
        head.addWidget(self.edit_btn)
        tr = QPushButton("▶ Try")
        tr.setToolTip("fire this rule's event now, as if the gesture happened")
        tr.setCursor(Qt.CursorShape.PointingHandCursor)
        tr.clicked.connect(lambda: self.try_it.emit(self))
        head.addWidget(tr)
        rm = QPushButton("Delete")
        rm.setProperty("class", "danger")
        rm.setCursor(Qt.CursorShape.PointingHandCursor)
        rm.clicked.connect(lambda: self.delete.emit(self))
        head.addWidget(rm)
        lay.addLayout(head)

        self.editor = QWidget()
        ed = QVBoxLayout(self.editor)
        ed.setContentsMargins(0, 2, 0, 0)
        ed.setSpacing(5)
        lay.addWidget(self.editor)
        self.hat = HatBlock(rule.setdefault("when", {"source": "myo", "name": "right.fist", "trigger": "press"}))
        self.hat.changed.connect(self.changed)
        self.hat.changed.connect(self._refresh_summary)
        ed.addWidget(self.hat)
        spine = QFrame()
        spine.setObjectName("spine")
        spine.setStyleSheet(f"QFrame#spine {{ border-left: 2px solid {_mix(theme.CARD_BORDER, C_EVENT, 0.5).name()}; "
                            f"margin-left: 10px; }}")
        self.body = QVBoxLayout(spine)
        self.body.setContentsMargins(10, 2, 0, 2)     # blocks hang under the hat, Scratch-style
        self.body.setSpacing(3)
        ed.addWidget(spine)
        self.hint = make_label("drop or click blocks to add them here", "faint")
        self.hint.setContentsMargins(20, 2, 0, 0)
        ed.addWidget(self.hint)
        self.changed.connect(self._refresh_summary)
        self.expanded = True
        self._rebuild()
        self._style()
        self.set_expanded(False)

    def set_expanded(self, on: bool):
        self.expanded = on
        self.editor.setVisible(on)
        self.title.setVisible(on)
        self.summary.setVisible(not on)
        self.edit_btn.setText("Done" if on else "Edit")
        self._refresh_summary()

    def _refresh_summary(self):
        when, acts = rule_summary(self.rule)
        title = self.rule.get("title", "")
        self.summary.setText(f"<b>{when}</b> &nbsp;<span style='color:{theme.TEXT_FAINT}'>\u2192</span>&nbsp; "
                             f"<span style='color:{theme.TEXT}'>{acts}</span>")
        self.summary.setToolTip(title)

    def _toggle(self, on):
        self.rule["enabled"] = on
        self._style()
        self.changed.emit()

    def _rebuild(self):
        while self.body.count():
            w = self.body.takeAt(0).widget()
            if w:
                w.deleteLater()
        for c in self.rule.setdefault("conditions", []):
            self._add_widget(BlockWidget(c, "cond", C_COND))
        for a in self.rule.setdefault("actions", []):
            self._add_widget(BlockWidget(a, "act", ACTION_COLOR.get(a.get("type"), C_INPUT)))
        self.hint.setVisible(not self.rule["conditions"] and not self.rule["actions"])

    def _add_widget(self, w):
        w.changed.connect(self.changed)
        w.remove.connect(self._remove_block)
        w.move.connect(self._move_block)
        self.body.addWidget(w)

    def _list_for(self, w):
        return self.rule["conditions"] if w.kind == "cond" else self.rule["actions"]

    def _remove_block(self, w):
        lst = self._list_for(w)
        if w.data in lst:
            lst.remove(w.data)
        self._rebuild()
        self.changed.emit()

    def _move_block(self, w, d):
        lst = self._list_for(w)
        i = next((k for k, x in enumerate(lst) if x is w.data), None)
        if i is None or not 0 <= i + d < len(lst):
            return
        lst[i], lst[i + d] = lst[i + d], lst[i]
        self._rebuild()
        self.changed.emit()

    def add_block(self, key: str) -> bool:
        cat, kind = key.split(":", 1)
        if not self.expanded:
            self.set_expanded(True)
        if cat == "cond":
            self.rule["conditions"].append(R.new_condition(kind))
        elif cat == "act":
            self.rule["actions"].append(R.new_action(kind))
        else:
            return False
        self._rebuild()
        self.changed.emit()
        return True

    # selection + live feedback
    def set_selected(self, on):
        self._is_selected = on
        self._style()

    def flash(self):
        self._flash = 6
        self._style()

    def tick(self):
        if self._flash:
            self._flash -= 1
            if not self._flash:
                self._style()

    def _style(self):
        border = theme.GREEN if self._flash else (theme.ACCENT if self._is_selected else theme.CARD_BORDER)
        width = 2 if (self._flash or self._is_selected) else 1
        bg = theme.CARD if self.rule.get("enabled", True) else theme.FIELD
        self.setStyleSheet(f"QFrame#rulecard {{ background: {bg}; border: {width}px solid {border}; border-radius: 12px; }}")

    def mousePressEvent(self, e):
        self.selected.emit(self)
        super().mousePressEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()

    def dropEvent(self, e):
        key = bytes(e.mimeData().data(MIME)).decode()
        self.selected.emit(self)
        if not self.add_block(key):     # a hat dropped on a rule: start a new rule instead
            self.parent_page.new_rule_from(key)
        e.acceptProposedAction()


# ── palette ──────────────────────────────────────────────────────────────────
class PaletteBlock(QLabel):
    clicked = Signal(str)

    def __init__(self, key, text, color):
        super().__init__(text)
        self.key, self.color = key, color
        self.hat = key.startswith("hat:")
        self.setStyleSheet(f"background: transparent; color: {theme.TEXT}; font-weight: 600; "
                           f"padding: 7px 12px 7px 14px;")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("click to add to the selected rule, or drag onto a rule")
        self._press = None

    def paintEvent(self, e):
        paint_block(self, self.color, self.hat)
        super().paintEvent(e)

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._press is None or (e.position().toPoint() - self._press).manhattanLength() < 8:
            return
        self._press = None
        drag = QDrag(self)
        md = QMimeData()
        md.setData(MIME, self.key.encode())
        drag.setMimeData(md)
        drag.setPixmap(self.grab())
        drag.setHotSpot(QPoint(10, 10))
        drag.exec(Qt.DropAction.CopyAction)

    def mouseReleaseEvent(self, e):
        if self._press is not None:
            self.clicked.emit(self.key)
        self._press = None


# ── the page ─────────────────────────────────────────────────────────────────
class BlocksPage(QWidget):
    def __init__(self, engine=None, parent=None, rules_path=None, executor=None):
        super().__init__(parent)
        self.engine = engine
        self.setObjectName("page")
        self.rules_path = rules_path or R.RULES_PATH
        self.data = R.load(self.rules_path)
        self._executor = executor
        self.runtime = None
        self.cards: list[RuleCard] = []
        self.sel: RuleCard | None = None
        self.bridge = _Bridge()
        self.bridge.fired.connect(self._on_fired)
        self.bridge.event.connect(self._on_event)
        self.bridge.claw.connect(self._on_claw)
        self._claw = None
        self._events = []

        # dark fields for the page's own dropdowns (the blocks style theirs themselves)
        self._bar_qss = (f"QComboBox {{ background: {theme.FIELD}; color: {theme.TEXT}; border: 1px solid "
                         f"{theme.CARD_BORDER}; border-radius: 8px; padding: 5px 10px; }}"
                         f"QComboBox QAbstractItemView {{ background: {theme.CARD}; color: {theme.TEXT}; "
                         f"selection-background-color: {theme.ACCENT}; }}")
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 20)
        root.setSpacing(10)

        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(make_label("Blocks", "h1"))
        sub = make_label("Any input → any action. Click or drag blocks into a rule; "
                         "changes save and apply instantly.", "dim")
        sub.setWordWrap(True)
        col.addWidget(sub)
        head.addLayout(col, stretch=1)
        import presets
        self._presets = presets
        self.preset = QComboBox()
        for k, v in presets.TITLES.items():
            self.preset.addItem(v, k)
        self.preset.setStyleSheet(self._bar_qss)
        load = QPushButton("Load example")
        load.setProperty("class", "ghost")
        load.setCursor(Qt.CursorShape.PointingHandCursor)
        load.clicked.connect(self._load_preset)
        add = QPushButton("  + New rule")
        add.setProperty("class", "primary")
        add.setCursor(Qt.CursorShape.PointingHandCursor)
        add.clicked.connect(lambda: self.new_rule_from("hat:myo"))
        head.addWidget(add)
        root.addLayout(head)

        # sources + run bar
        bar = QHBoxLayout()
        bar.setSpacing(10)
        bar.addWidget(make_label("Input sources:", "dim"))
        self.myo_mode = QComboBox()
        self.myo_mode.addItem("Myo off", "off")
        self.myo_mode.addItem("Myo: built-in poses (no training)", "builtin")
        self.myo_mode.addItem("Myo: trained click", "trained")
        self.myo_mode.addItem("Myo: 5 trained gestures", "gestures")
        self.myo_mode.setStyleSheet(self._bar_qss)
        bar.addWidget(self.myo_mode)
        self.user = QLineEdit()
        self.user.setPlaceholderText("user (trained / gestures)")
        self.user.setMaximumWidth(120)
        bar.addWidget(self.user)
        self.use_lsl = QCheckBox("LSL markers")
        self.use_lsl.setToolTip("Lab Streaming Layer marker streams: EEG / BCI decoders (OpenViBE, BCI2000, "
                                "PsychoPy...) or any app that sends markers. Needs: pip install pylsl")
        bar.addWidget(self.use_lsl)
        self.use_cam = QCheckBox("Camera poses")
        bar.addWidget(self.use_cam)
        bar.addStretch()
        self.mode_lbl = make_label("mode: normal", None, 12, QFont.Weight.DemiBold, C_CONTROL)
        bar.addWidget(self.mode_lbl)
        self.run_btn = QPushButton("▶  Run rules")
        self.run_btn.setProperty("class", "primary")
        self.run_btn.setCheckable(True)
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.toggled.connect(self._run_toggled)
        bar.addWidget(self.run_btn)
        root.addLayout(bar)
        self.status = make_label("Not running. Rules still fire from ▶ Try.", "faint")
        srow = QHBoxLayout()
        srow.addWidget(self.status, stretch=1)
        srow.addWidget(make_label("Examples:", "dim"))
        srow.addWidget(self.preset)
        srow.addWidget(load)
        imp = QPushButton("Import camera mappings")
        imp.setProperty("class", "ghost")
        imp.setToolTip("turn the Mappings page's camera bindings into rules (re-importing replaces them)")
        imp.setCursor(Qt.CursorShape.PointingHandCursor)
        imp.clicked.connect(self.import_camera_mappings)
        srow.addWidget(imp)
        root.addLayout(srow)

        # palette | canvas
        split = QHBoxLayout()
        split.setSpacing(16)
        pal_scroll = QScrollArea()
        pal_scroll.setWidgetResizable(True)
        pal_scroll.setFixedWidth(250)
        pal = QWidget()
        pl = QVBoxLayout(pal)
        pl.setContentsMargins(0, 0, 8, 0)
        pl.setSpacing(4)
        for cat, color, items in PALETTE:
            pl.addWidget(make_label(cat, None, 12, QFont.Weight.Bold, color))
            for key, text in items:
                pb = PaletteBlock(key, text, color)
                pb.clicked.connect(self._palette_click)
                pl.addWidget(pb)
            pl.addSpacing(6)
        pl.addStretch()
        pal_scroll.setWidget(pal)
        self.pal_scroll = pal_scroll
        split.addWidget(pal_scroll)

        canvas_scroll = QScrollArea()
        self.canvas_scroll = canvas_scroll
        canvas_scroll.setWidgetResizable(True)
        self.canvas = QWidget()
        self.canvas.setAcceptDrops(True)
        self.canvas.dragEnterEvent = self._canvas_drag_enter
        self.canvas.dropEvent = self._canvas_drop
        self.cl = QVBoxLayout(self.canvas)
        self.cl.setContentsMargins(0, 0, 8, 0)
        self.cl.setSpacing(6)
        self.empty = make_label("No rules yet. Click a yellow  when …  block to start one, "
                                "or pick an example at the top and press Load example.", "dim")
        self.empty.setWordWrap(True)
        self.empty.setContentsMargins(6, 20, 6, 0)
        self.cl.addWidget(self.empty)
        self.cl.addStretch()
        canvas_scroll.setWidget(self.canvas)
        split.addWidget(canvas_scroll, stretch=1)
        root.addLayout(split, stretch=1)

        self.ticker = make_label("events: —", "faint")
        root.addWidget(self.ticker)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self.save)
        self._tick = QTimer(self)
        self._tick.setInterval(100)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()
        self._ensure_runtime()
        self._rebuild()

    # ── model ──
    def _ensure_runtime(self):
        if self.runtime is None:
            from runtime import Runtime
            try:
                from context import get_active_context
            except Exception:
                get_active_context = None
            self.runtime = Runtime(self._executor, self.data, get_active_context)
            self.runtime.engine.listeners.append(lambda rid, ev: self.bridge.fired.emit(rid))
            self.runtime.event_listeners.append(
                lambda ev: self.bridge.event.emit(f"{ev.source}:{ev.name} {ev.kind}") if ev.kind != "value" else None)
            hooks = getattr(self.runtime.engine.executor, "hooks", None)
            if hooks is not None:
                hooks["claw"] = self.bridge.claw.emit       # worker thread -> GUI thread
            self.runtime.start()

    def _rebuild(self, expand_id=None):
        for w in self.cards + getattr(self, "headers", []):
            w.setParent(None)
            w.deleteLater()
        self.cards, self.headers = [], []
        self._group_open = getattr(self, "_group_open", {})
        self.empty.setVisible(not self.data["rules"])
        groups = {}
        for r in self.data["rules"]:
            groups.setdefault(group_key(r), []).append(r)
        for key in sorted(groups, key=group_sort):
            rules_ = groups[key]
            open_ = self._group_open.get(key, True)
            h = GroupHeader(group_title(key), len(rules_), open_)
            h.toggled.connect(lambda on, k=key: self._toggle_group(k, on))
            self.cl.insertWidget(self.cl.count() - 1, h)
            self.headers.append(h)
            for r in rules_:
                c = self._add_card(r)
                c.group = key
                c.setVisible(open_)
                if expand_id and r.get("id") == expand_id:
                    c.set_expanded(True)
                    self._select(c)
                    self._group_open[key] = True
        if self.cards and self.sel not in self.cards:
            self.sel = None

    def _toggle_group(self, key, on):
        self._group_open[key] = on
        for c in self.cards:
            if getattr(c, "group", None) == key:
                c.setVisible(on)

    def _add_card(self, rule):
        c = RuleCard(rule)
        c.parent_page = self
        c.changed.connect(self._changed)
        c.selected.connect(self._select)
        c.delete.connect(self._delete)
        c.try_it.connect(self._try)
        # a rule moved to another input belongs in another group
        c.hat.src.currentIndexChanged.connect(lambda *_: QTimer.singleShot(0, self.regroup))
        self.cl.insertWidget(self.cl.count() - 1, c)
        self.cards.append(c)
        self.empty.setVisible(False)
        return c

    def _select(self, card):
        self.sel = card
        for c in self.cards:
            c.set_selected(c is card)

    def _changed(self):
        self._save_timer.start()

    def regroup(self):
        """Call after edits that change a rule's source or app (it may belong in another group)."""
        sel_id = self.sel.rule.get("id") if self.sel else None
        self._rebuild(expand_id=sel_id)

    def save(self):
        self.data["mode"] = self.runtime.engine.mode if self.runtime else self.data.get("mode", "normal")
        R.save(self.data, self.rules_path)

    def new_rule_from(self, key: str):
        src = key.split(":", 1)[1] if key.startswith("hat:") else "myo"
        name = R.SOURCE_EVENTS.get(src, [""])[0]
        rule = R.new_rule(src, name, "press")
        self.data["rules"].append(rule)
        self._group_open[group_key(rule)] = True
        self._rebuild(expand_id=rule["id"])
        card = self.sel
        self._changed()
        QTimer.singleShot(50, lambda: self._scroll_to(card))
        return card

    def _scroll_to(self, card):
        try:
            self.canvas_scroll.ensureWidgetVisible(card, 0, 40)
        except RuntimeError:
            pass

    def _palette_click(self, key):
        if key.startswith("hat:") or self.sel is None:
            card = self.new_rule_from(key if key.startswith("hat:") else "hat:myo")
            if not key.startswith("hat:"):
                card.add_block(key)
        else:
            self.sel.add_block(key)

    def _delete(self, card):
        if card.rule in self.data["rules"]:
            self.data["rules"].remove(card.rule)
        self.cards.remove(card)
        card.setParent(None)
        card.deleteLater()
        if self.sel is card:
            self.sel = None
        self._rebuild()
        self._changed()

    def _load_preset(self):
        key = self.preset.currentData()
        if self.data["rules"] and QMessageBox.question(
                self, "Load example", "Replace your current rules with this example?") != QMessageBox.StandardButton.Yes:
            return
        self.load_preset(key)

    def load_preset(self, key):
        self.data.clear()
        self.data.update(self._presets.PRESETS[key]())
        self.runtime.engine.set_rules(self.data)
        self.sel = None
        self._rebuild()
        self.save()

    def import_camera_mappings(self) -> int:
        import import_mappings as IM
        n = IM.import_current(self.data)
        self.runtime.engine.set_rules(self.data)
        self.sel = None
        self._rebuild()
        self.save()
        self.status.setText(f"Imported {n} camera bindings. They run when 'Camera poses' is on here - "
                            "don't also use Home > Start control, or they fire twice.")
        self.status.setToolTip("The camera pointer, pinch-to-click on the pointer and the Minecraft joysticks "
                               "stay in Home > Start control; they aren't rules.")
        return n

    def _try(self, card):
        w = card.rule.get("when", {})
        if w.get("trigger") == "value":
            ev = R.Event(w.get("source", "myo"), w.get("name", ""), "value", (40, 0))
            self.runtime.emit(ev)
            return
        self.runtime.emit(R.Event(w.get("source", "myo"), w.get("name", ""), "press"))
        QTimer.singleShot(400, lambda: self.runtime.emit(R.Event(w.get("source", "myo"), w.get("name", ""), "release")))

    # ── canvas drops: a hat on empty canvas = new rule ──
    def _canvas_drag_enter(self, e):
        if e.mimeData().hasFormat(MIME):
            e.acceptProposedAction()

    def _canvas_drop(self, e):
        key = bytes(e.mimeData().data(MIME)).decode()
        self._palette_click(key)
        e.acceptProposedAction()

    # ── running ──
    def _run_toggled(self, on):
        self.run_btn.setText("■  Stop rules" if on else "▶  Run rules")
        if on:
            mode = self.myo_mode.currentData()
            if mode != "off":
                from myo_source import MyoSource
                user = self.user.text().strip() or None
                if mode in ("trained", "gestures") and not user:
                    self.status.setText("Trained modes need your user name (the one you trained with).")
                    self.run_btn.setChecked(False)
                    return
                self.runtime.add_source(MyoSource(mode, user, gesture_sides=self.data.get("gesture_sides"),
                                                  move=self.data.get("move", "air")))
            if self.use_lsl.isChecked():
                from lsl_source import LslSource
                self.runtime.add_source(LslSource())
            if self.use_cam.isChecked():
                from camera_source import CameraSource
                det = getattr(self.engine, "detector", None)
                self.runtime.add_source(CameraSource(det, own_pump=det is None))
            if not self.runtime.sources:
                self.status.setText("No input sources chosen: pick Myo, LSL markers or camera first.")
                self.run_btn.setChecked(False)
        else:
            for n in list(self.runtime.sources):
                self.runtime.remove_source(n)
            self.runtime.engine.release_all()
            self.status.setText("Stopped. Rules still fire from ▶ Try.")

    def _on_claw(self, a: dict):
        if self._claw is None:
            from claw_overlay import ClawOverlay
            import actions
            self._claw = ClawOverlay(actions.click_at)
        if a.get("type") == "claw_cancel":
            self._claw.cancel()
        elif a.get("type") == "claw_drive":
            self._claw.drive(a)
        else:
            self._claw.step(a)

    def _on_fired(self, rid):
        for c in self.cards:
            if c.rule.get("id") == rid:
                c.flash()

    def _on_event(self, text):
        self._events = (self._events + [text])[-6:]
        self.ticker.setText("events:  " + "   ·   ".join(self._events))

    def _on_tick(self):
        for c in self.cards:
            c.tick()
        if self.runtime:
            self.mode_lbl.setText(f"mode: {self.runtime.engine.mode}")
            if self.runtime.sources:
                self.status.setText("   ".join(f"{n}: {s.status}" for n, s in self.runtime.sources.items()))
            ex = self.runtime.engine.executor
            msgs = getattr(ex, "messages", None)
            if msgs:
                self.status.setText(f"message: {msgs[-1]}")
                msgs.clear()

    def showEvent(self, e):
        super().showEvent(e)
        self.pal_scroll.verticalScrollBar().setValue(0)

    def shutdown(self):
        if self.runtime:
            self.runtime.stop()
        self.save()
