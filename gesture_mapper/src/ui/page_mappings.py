"""
Mappings: per-context editor. Shows what's built into each mode (joysticks,
pointer, clutch) honestly alongside the editable gesture → command bindings.
"""

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

import theme
import labels
import handicon
import commands as commands_mod
from widgets import Card, Chip, SegmentedControl, hline, make_label

_POSE_PINCH_CLASH = {"index_pointer", "peace"}


class BindingRow(QWidget):
    remove_requested = Signal(tuple)   # (hand, gesture)

    def __init__(self, hand: str, gesture: str, cmd, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 10, 8)
        lay.setSpacing(12)

        icon = QLabel()
        icon.setPixmap(handicon.gesture_pixmap(gesture, 44))
        icon.setFixedSize(48, 48)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(icon)

        gcol = QVBoxLayout()
        gcol.setSpacing(1)
        gcol.addWidget(make_label(labels.gesture_label(gesture), None, 13,
                                  QFont.Weight.DemiBold))
        hand_label, hand_color = labels.HANDS.get(hand, (hand, theme.TEXT_DIM))
        gcol.addWidget(make_label(hand_label, None, 11, None, hand_color))
        lay.addLayout(gcol, stretch=2)

        arrow = QLabel()
        arrow.setPixmap(theme.sf_symbol("arrow.right", 12, theme.TEXT_FAINT))
        lay.addWidget(arrow)

        ccol = QVBoxLayout()
        ccol.setSpacing(1)
        ccol.addWidget(make_label(labels.command_label(cmd.name), None, 13,
                                  QFont.Weight.DemiBold))
        ccol.addWidget(make_label(labels.command_detail(cmd.name), "faint"))
        lay.addLayout(ccol, stretch=3)

        mode = Chip("hold" if cmd.mode == "held" else "tap",
                    bg=theme.FIELD,
                    fg=theme.ORANGE if cmd.mode == "held" else theme.TEAL)
        lay.addWidget(mode)

        rm = QPushButton()
        rm.setProperty("class", "ghost")
        rm.setIcon(theme.sf_icon("trash", 12, theme.TEXT_FAINT))
        rm.setFixedSize(26, 26)
        rm.setToolTip("Remove binding")
        rm.setCursor(Qt.CursorShape.PointingHandCursor)
        rm.clicked.connect(lambda: self.remove_requested.emit((hand, gesture)))
        lay.addWidget(rm)


class MappingsPage(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setObjectName("page")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 24)
        root.setSpacing(14)

        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(make_label("Mappings", "h1"))
        col.addWidget(make_label("What each gesture does, per app. Changes apply "
                                 "instantly — even while control is live.", "dim"))
        head.addLayout(col)
        head.addStretch()

        restore = QPushButton("Restore defaults")
        restore.setProperty("class", "ghost")
        restore.setCursor(Qt.CursorShape.PointingHandCursor)
        restore.clicked.connect(self._restore_defaults)
        head.addWidget(restore)

        add = QPushButton("  Add Binding")
        add.setProperty("class", "primary")
        add.setIcon(theme.sf_icon("plus", 13, "white", "semibold"))
        add.setMinimumHeight(36)
        add.setCursor(Qt.CursorShape.PointingHandCursor)
        add.clicked.connect(self._add)
        head.addWidget(add)
        root.addLayout(head)

        self.seg = SegmentedControl([labels.context_label(c)
                                     for c in labels.CONTEXT_ORDER])
        self.seg.changed.connect(lambda _: self._rebuild())
        seg_row = QHBoxLayout()
        seg_row.addWidget(self.seg)
        seg_row.addStretch()
        root.addLayout(seg_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.body_lay = QVBoxLayout(body)
        self.body_lay.setContentsMargins(0, 4, 0, 12)
        self.body_lay.setSpacing(12)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        engine.mappings_changed.connect(self._rebuild)
        engine.library_changed.connect(self._rebuild)
        self._rebuild()

    def _context(self) -> str:
        return labels.CONTEXT_ORDER[self.seg.index()]

    def _rebuild(self):
        while self.body_lay.count():
            item = self.body_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        ctx = self._context()

        builtins = labels.BUILTINS.get(ctx, [])
        if builtins:
            card = Card()
            lay = QVBoxLayout(card)
            lay.setContentsMargins(16, 12, 16, 12)
            lay.setSpacing(8)
            title_row = QHBoxLayout()
            t = make_label("Built into this mode", None, 13, QFont.Weight.DemiBold,
                           theme.TEXT_DIM)
            title_row.addWidget(t)
            title_row.addStretch()
            lay.addLayout(title_row)
            for hand, title, detail, symbol in builtins:
                row = QHBoxLayout()
                row.setSpacing(10)
                ic = QLabel()
                _, hand_color = labels.HANDS.get(hand, ("", theme.TEXT_DIM))
                ic.setPixmap(theme.sf_symbol(symbol, 15, hand_color))
                ic.setFixedWidth(24)
                row.addWidget(ic)
                row.addWidget(make_label(title, None, 13, QFont.Weight.Medium))
                row.addWidget(make_label(detail, "faint"))
                row.addStretch()
                hand_label, _ = labels.HANDS.get(hand, (hand, ""))
                row.addWidget(make_label(hand_label.replace(" hand", ""), None,
                                         11, None, hand_color))
                lay.addLayout(row)
            self.body_lay.addWidget(card)

        _finger_order = {"pinch_index": 0, "pinch_middle": 1,
                         "pinch_ring": 2, "pinch_pinky": 3}

        def row_key(row):
            g, _ = row
            # poses alphabetically, then pinches in finger order
            return (g in _finger_order, _finger_order.get(g, 0), g)

        table = self.engine.mappings().get(ctx, {})
        for hand in ("left", "right", "either"):
            rows = sorted(((g, cmd) for (h, g), cmd in table.items() if h == hand),
                          key=row_key)
            if not rows:
                continue
            hand_label, hand_color = labels.HANDS[hand]
            self.body_lay.addWidget(make_label(hand_label, "h2"))
            card = Card()
            lay = QVBoxLayout(card)
            lay.setContentsMargins(0, 4, 0, 4)
            lay.setSpacing(0)
            for i, (gesture, cmd) in enumerate(rows):
                if i:
                    ln = hline()
                    ln.setStyleSheet(f"background: {theme.CARD_BORDER}; "
                                     "border: none; margin-left: 74px;")
                    lay.addWidget(ln)
                row = BindingRow(hand, gesture, cmd)
                row.remove_requested.connect(self._remove)
                lay.addWidget(row)
            self.body_lay.addWidget(card)

        if not table:
            self.body_lay.addWidget(make_label(
                "No bindings in this mode yet — add one.", "dim"))
        self.body_lay.addStretch()

    def _remove(self, key: tuple):
        ctx = self._context()
        tables = {c: dict(t) for c, t in self.engine.mappings().items()}
        tables[ctx].pop(key, None)
        self.engine.save_mappings(tables)

    def _restore_defaults(self):
        box = QMessageBox(self)
        box.setWindowTitle("Restore defaults")
        box.setText("Replace all bindings in every mode with the factory defaults?")
        box.setStandardButtons(QMessageBox.StandardButton.Cancel |
                               QMessageBox.StandardButton.Yes)
        box.button(QMessageBox.StandardButton.Yes).setText("Restore")
        if box.exec() == QMessageBox.StandardButton.Yes:
            self.engine.restore_default_mappings()

    def _add(self):
        dlg = AddBindingDialog(self.engine, self._context(), self)
        dlg.exec()


class AddBindingDialog(QDialog):
    def __init__(self, engine, context: str, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.context = context
        self.setWindowTitle("Add binding")
        self.setModal(True)
        self.setFixedSize(560, 660)
        self.setStyleSheet(theme.QSS)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)

        lay.addWidget(make_label(f"Add binding — {labels.context_label(context)}",
                                 "h2"))

        lay.addWidget(make_label("HAND", "faint"))
        self.hand_seg = SegmentedControl(["Left", "Right", "Either"])
        row = QHBoxLayout()
        row.addWidget(self.hand_seg)
        row.addStretch()
        lay.addLayout(row)
        self.hand_seg.changed.connect(lambda _: self._refresh_warning())

        lay.addWidget(make_label("GESTURE", "faint"))
        self.gesture_list = QListWidget()
        self.gesture_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.gesture_list.setIconSize(QSize(52, 52))
        self.gesture_list.setGridSize(QSize(114, 92))
        self.gesture_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gesture_list.setMovement(QListWidget.Movement.Static)
        self.gesture_list.setFixedHeight(204)
        self.gesture_list.setWordWrap(True)
        for name in engine.gesture_names() + labels.PINCH_GESTURES:
            if name == "pinch":
                continue   # raw template backing pinch signals, not bindable
            item = QListWidgetItem(handicon.gesture_pixmap(name, 52),
                                   labels.gesture_label(name))
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.gesture_list.addItem(item)
        self.gesture_list.currentItemChanged.connect(
            lambda *_: self._refresh_warning())
        lay.addWidget(self.gesture_list)

        lay.addWidget(make_label("ACTION", "faint"))
        self.cmd_list = QListWidget()
        for name in commands_mod.ALL_COMMANDS:
            label = labels.command_label(name)
            detail = labels.command_detail(name)
            item = QListWidgetItem(f"{label}    ·  {detail}" if detail else label)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.cmd_list.addItem(item)
        self.cmd_list.currentItemChanged.connect(lambda *_: self._refresh_warning())
        lay.addWidget(self.cmd_list, stretch=1)

        self.warning = make_label("", None, 12, None, theme.ORANGE)
        self.warning.setWordWrap(True)
        self.warning.setMinimumHeight(30)
        lay.addWidget(self.warning)

        btns = QHBoxLayout()
        btns.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        self.save_btn = QPushButton("Add Binding")
        self.save_btn.setProperty("class", "primary")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        btns.addWidget(self.save_btn)
        lay.addLayout(btns)

    def _selection(self):
        hand = ("left", "right", "either")[self.hand_seg.index()]
        g_item = self.gesture_list.currentItem()
        c_item = self.cmd_list.currentItem()
        gesture = g_item.data(Qt.ItemDataRole.UserRole) if g_item else None
        cmd = c_item.data(Qt.ItemDataRole.UserRole) if c_item else None
        return hand, gesture, cmd

    def _refresh_warning(self):
        hand, gesture, cmd = self._selection()
        self.save_btn.setEnabled(bool(gesture and cmd))
        msgs = []
        if gesture:
            table = self.engine.mappings().get(self.context, {})
            existing = table.get((hand, gesture))
            if existing is not None:
                msgs.append(f"Replaces the current binding "
                            f"({labels.command_label(existing.name)}).")
            # pose/pinch clash (see src/cv/CLAUDE.md)
            if gesture in _POSE_PINCH_CLASH:
                has_pinch = any(h == hand and g.startswith("pinch_")
                                for (h, g) in table)
                if has_pinch:
                    msgs.append("Heads up: pointing poses disturb pinch detection "
                                "on the same hand — this pairing misfires.")
            if hand == "right" and self.context in ("default", "spotlight") \
                    and not gesture.startswith("pinch_"):
                msgs.append("The right hand is the pointer here — this pose will "
                            "fire while you’re moving the cursor.")
        self.warning.setText("  ".join(msgs))

    def _save(self):
        hand, gesture, cmd_name = self._selection()
        cmd = commands_mod.ALL_COMMANDS[cmd_name]
        tables = {c: dict(t) for c, t in self.engine.mappings().items()}
        tables.setdefault(self.context, {})[(hand, gesture)] = cmd
        self.engine.save_mappings(tables)
        self.accept()
