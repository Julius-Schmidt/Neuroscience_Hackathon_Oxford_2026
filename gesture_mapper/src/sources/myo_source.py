"""Myo armbands -> events. Two ways to read gestures (a band takes one connection at a time):

  mode "trained"  our per-user click model (myomouse LDA): 'myo:right.fist' (or '.index'),
                  needs `python -m myomouse.train --user <you>` first. Most reliable click.
  mode "builtin"  the band's own classifier, no training: 'myo:<side>.fist', '.wave_in',
                  '.wave_out', '.fingers_spread', '.double_tap'. Do the sync gesture once
                  (bend the wrist back and hold until it vibrates).
  mode "gestures" our 5-gesture model (myomouse.gestures11) on every band that has one:
                  'myo:<side>.ulnar_deviation', '.wrist_extension', '.power_grip',
                  '.chuck_grip', '.pinch_grip'. Bands without a gesture model fall back to
                  the trained click ('<side>.fist'), so one arm can do gestures, the other
                  look + click. Needs `python -m myomouse.gestures11.train --user <you>`.

All modes also give arm pointing as a continuous value 'myo:<side>.pointer' = (dx, dy) pixels,
for 'move mouse' blocks - so two bands = two gesture sets + a pointer: a controller.
"""

import asyncio
import os
import queue
import sys
import time

from base import REPO_ROOT, EdgeTracker, Event, Source

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# mode "gestures" is tuned for games, stricter than gestures11.live (measured on a recording:
# a quarter of the wrong-gesture events, almost no dropouts mid-hold, ~75 ms more delay)
GAME_REJECT = 0.75   # classifier must be this sure
GAME_ON_MS = 100     # a gesture must last this long to count (live default 75; 150 felt laggy)
GAME_OFF_MS = 150    # ... and neutral this long to let go (live default 50): a short dropout keeps W held

POSE_NAMES = {"fist": "fist", "wave in": "wave_in", "wave out": "wave_out",
              "fingers spread": "fingers_spread", "double tap": "double_tap"}


class MyoSource(Source):
    name = "myo"

    def __init__(self, mode="builtin", user=None, sides=None, pointer_sides=("right",),
                 move="air", click="fist", fake=False, reject=GAME_REJECT, gesture_sides=None,
                 on_ms=GAME_ON_MS, off_ms=GAME_OFF_MS):
        super().__init__()
        self.mode, self.user, self.sides, self.reject = mode, user, sides, reject
        self.gesture_sides = gesture_sides   # mode "gestures": None = every band with a gesture model
        self.on_ms, self.off_ms = on_ms, off_ms
        self.pointer_sides, self.move, self.click, self.fake = tuple(pointer_sides), move, click, fake
        self.synced = {}

    def _sides(self):
        if self.sides:
            return list(self.sides)
        from myomouse.device import default_sides
        return [s for s in default_sides(self.fake).split(",") if s]

    def _start(self):
        if self.mode == "trained":
            self._thread(self._trained, self._sides())
        elif self.mode == "gestures":
            self._start_gestures()
        else:
            if self.fake:
                self.status = "built-in mode needs real bands (use mode 'trained' with fake)"
                return
            from myomouse.device import load_bands
            bands = load_bands()
            for side in self._sides():
                if side in bands:
                    self._thread(self._builtin, side, bands[side])
            self.status = "connecting..."

    # ── our trained click model, reusing myomouse's band processes ──
    def _trained(self, sides):
        from myomouse import config as C
        from myomouse.run import start_bands
        q, procs = start_bands(self.user, sides, "lda", self.fake, False, False, self.click, self.move, "both")
        self.status = "connecting..."
        try:
            while not self._stop.is_set():
                try:
                    kind, a, b, _t = q.get(timeout=0.2)
                except Exception:
                    continue
                if kind == "ready":
                    self.status = f"running ({', '.join(sides)})"
                elif kind == "btn":
                    self.emit(Event("myo", f"{a}.{self.click}", "press" if b == "press" else "release"))
                elif kind == "move":
                    self.emit(Event("myo", f"{C.POINTER_SIDE}.pointer", "value", (a, b)))
        finally:
            for p in procs:
                p.terminate()

    # ── our 5-gesture model (gestures11), one band connection per gesture side ──
    def _start_gestures(self):
        from myomouse.gestures11 import model_path
        sides = self._sides()
        g_sides = [s for s in sides if model_path(self.user, s).exists()
                   and (self.gesture_sides is None or s in self.gesture_sides)]
        if not g_sides:
            self.status = (f"no gesture model for {self.user} ({', '.join(sides)}): "
                           f"run python -m myomouse.gestures11.train --user {self.user} --side <side>")
            return
        for side in g_sides:
            self._thread(self._gestures, side)
        rest = [s for s in sides if s not in g_sides]
        if rest:   # the other band: trained click + pointer, like mode "trained"
            self._thread(self._trained, rest)
        self.status = "connecting..."

    def _gestures(self, side):
        import joblib
        from myomouse.device import open_band
        from myomouse.gestures11 import STEP, model_path
        from myomouse.gestures11.live import GestureClassifier
        from myomouse.pointer import make_pointer
        decision_ms = STEP * 1000 / 200   # EMG at 200 Hz
        clf = GestureClassifier(joblib.load(model_path(self.user, side)), self.reject,
                                max(1, round(self.on_ms / decision_ms)), max(1, round(self.off_ms / decision_ms)))
        edges = EdgeTracker("myo", self.emit)
        band = open_band(side, fake=self.fake)

        def on_emg(sample):
            g = clf.push(sample)
            if g is not None:
                self.on_gesture(side, edges, g)
        band.add_emg_handler(on_emg)
        if side in self.pointer_sides:
            ptr = make_pointer(self.move)

            def on_imu(quat, acc, gyro):
                dx, dy = ptr.update(quat, acc, gyro)
                if dx or dy:
                    self.emit(Event("myo", f"{side}.pointer", "value", (dx, dy)))
            band.add_imu_handler(on_imu)
        band.start()
        self.status = f"{side}: gestures running"
        try:
            while not self._stop.is_set() and band.alive:
                time.sleep(0.1)
            if not band.alive:
                self.status = f"{side} band disconnected: {getattr(band, 'error', '')}"
        finally:
            edges.clear()
            band.stop()

    @staticmethod
    def on_gesture(side, edges, gesture):
        """The shown gesture changed -> press / release. neutral / rest = no gesture."""
        from myomouse.gestures11 import IDLE_KEYS
        edges.set(side, None if gesture in IDLE_KEYS else gesture)

    # ── the band's built-in classifier + IMU pointer, one BLE thread per band ──
    def _builtin(self, side, address):
        from myomouse import onboard as OB
        from myomouse.device import COMMAND_UUID, IMU_UUID, cmd_vibrate
        from myomouse.pointer import make_pointer
        edges = EdgeTracker("myo", self.emit)
        ptr = make_pointer(self.move) if side in self.pointer_sides else None
        events = queue.Queue()

        async def main():
            from bleak import BleakClient, BleakScanner
            dev = await BleakScanner.find_device_by_address(address, timeout=15.0)
            if dev is None:
                self.status = f"{side} band not found (wake it / close other scripts)"
                return
            async with BleakClient(dev) as client:
                await client.write_gatt_char(COMMAND_UUID, OB.CMD_NEVER_SLEEP, response=True)
                await client.write_gatt_char(COMMAND_UUID, OB.CMD_CLASSIFIER_ON, response=True)
                await client.start_notify(OB.CLASSIFIER_UUID, lambda _c, d: events.put(OB.parse_event(d)))
                if ptr is not None:
                    import struct

                    def on_imu(_c, d):
                        v = struct.unpack("<10h", d[:20])
                        dx, dy = ptr.update(v[:4], v[4:7], v[7:10])
                        if dx or dy:
                            self.emit(Event("myo", f"{side}.pointer", "value", (dx, dy)))
                    await client.start_notify(IMU_UUID, on_imu)
                await client.write_gatt_char(COMMAND_UUID, OB.CMD_UNLOCK_HOLD, response=True)
                await client.write_gatt_char(COMMAND_UUID, cmd_vibrate(1), response=True)
                self.status = f"{side}: connected - do the sync gesture"
                while not self._stop.is_set():
                    while not events.empty():
                        kind, detail = events.get()
                        if self.on_classifier(side, edges, kind, detail):
                            await client.write_gatt_char(COMMAND_UUID, OB.CMD_UNLOCK_HOLD, response=True)
                    await asyncio.sleep(0.02)
                edges.clear()
                try:
                    await client.write_gatt_char(COMMAND_UUID, OB.CMD_ALL_OFF, response=True)
                except Exception:
                    pass
        asyncio.run(main())

    def on_classifier(self, side, edges, kind, detail) -> bool:
        """One classifier event from a band -> events. Returns True when the band needs
        re-unlocking (just synced)."""
        if kind == "synced":
            self.synced[side] = True
            self.status = f"{side}: synced ({detail})"
            return True
        if kind == "unsynced":
            self.synced[side] = False
            edges.clear()
            self.status = f"{side}: unsynced - sync again"
        elif kind == "pose":
            edges.set(side, POSE_NAMES.get(detail))
        return False
