"""Blocks page, offscreen: presets load, palette clicks build a rule, editing a block changes
the saved rule, Try fires through the real executor (stub OS off Windows/macOS), modes work.
Skips (PASS) if Qt can't start on this machine."""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.join(os.path.dirname(__file__), "..")
for sub in ("src/ui", "src/cv", "src/mapping-engine", "src/os-integration", "src/sources"):
    sys.path.insert(0, os.path.join(ROOT, sub))
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


try:
    from PySide6.QtWidgets import QApplication, QComboBox
    app = QApplication.instance() or QApplication(sys.argv)
except Exception as e:  # no display libs (e.g. headless VM without libEGL)
    print(f"PASS skipped: Qt unavailable here ({type(e).__name__})")
    sys.exit(0)

import rules as R
from page_blocks import BlockWidget, BlocksPage


def pump(n=20):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.03)


class Rec:
    def __init__(self):
        self.log, self.messages = [], []

    def run(self, a, ev, eng):
        self.log.append(a["type"] + ":" + str(a.get("key", a.get("text", ""))))

    def press(self, a):
        self.log.append("press:" + str(a.get("key", a.get("button"))))

    def release(self, a):
        self.log.append("release:" + str(a.get("key", a.get("button"))))


path = os.path.join(tempfile.mkdtemp(), "rules.json")
rec = Rec()
page = BlocksPage(None, rules_path=path, executor=rec)
check(page.empty.isVisibleTo(page), "empty state shown with no rules")
page.load_preset("controller")
check(len(page.cards) == 11, f"controller preset -> 11 rules ({len(page.cards)})")
page._palette_click("hat:lsl")
page._palette_click("act:key_tap")
card = page.sel
bw = [w for w in card.findChildren(BlockWidget) if w.data.get("type") == "key_tap"][0]
bw.findChild(QComboBox).setCurrentText("enter")
check(card.rule["actions"] == [{"type": "key_tap", "key": "enter"}], "palette click + edit -> rule data")
page._try(card)
pump()
check("key_tap:enter" in rec.log, f"Try fires the rule ({rec.log})")
w = [c for c in page.cards if c.rule["when"]["name"] == "left.fist"][0]
rec.log.clear()
page._try(w)
pump(25)
check(rec.log == ["press:w", "release:w"], f"held rule presses then releases ({rec.log})")
page.save()
check(len(R.load(path)["rules"]) == 12, "saved to rules.json")
n = page.import_camera_mappings()
n2 = page.import_camera_mappings()
check(n > 0 and len(page.cards) == 12 + n2, f"import camera mappings twice -> added once ({n}, {len(page.cards)} cards)")
page.shutdown()
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
