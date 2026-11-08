"""Myo armbands -> events. Two ways to read gestures (a band takes one connection at a time):

  mode "trained"  our per-user click model (myomouse LDA): 'myo:right.fist' (or '.index'),
                  needs `python -m myomouse.train --user <you>` first. Most reliable click.
  mode "builtin"  the band's own classifier, no training: 'myo:<side>.fist', '.wave_in',
                  '.wave_out', '.fingers_spread', '.double_tap'. Do the sync gesture once
                  (bend the wrist back and hold until it vibrates).

Both also give arm pointing as a continuous value 'myo:<side>.pointer' = (dx, dy) pixels,
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

POSE_NAMES = {"fist": "fist", "wave in": "wave_in", "wave out": "wave_out",
              "fingers spread": "fingers_spread", "double tap": "double_tap"}


class MyoSource(Source):
    name = "myo"

    def __init__(self, mode="builtin", user=None, sides=None, pointer_sides=("right",),
                 move="air", click="fist", fake=False):
        super().__init__()
        self.mode, self.user, self.sides = mode, user, sides
        self.pointer_sides, self.move, self.click, self.fake = tuple(pointer_sides), move, click, fake
        self.synced = {}

    def _sides(self):
        if self.sides:
            return list(self.sides)
        from myomouse.device import default_sides
        return [s for s in default_sides(self.fake).split(",") if s]

    def _start(self):
        if self.mode == "trained":
            self._thread(self._trained)
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
    def _trained(self):
        from myomouse import config as C
        from myomouse.run import start_bands
        sides = self._sides()
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
                        if kind == "synced":
                            self.synced[side] = True
                            self.status = f"{side}: synced ({detail})"
                            await client.write_gatt_char(COMMAND_UUID, OB.CMD_UNLOCK_HOLD, response=True)
                        elif kind == "unsynced":
                            self.synced[side] = False
                            edges.clear()
                            self.status = f"{side}: unsynced - sync again"
                        elif kind == "pose":
                            edges.set(side, POSE_NAMES.get(detail))
                    await asyncio.sleep(0.02)
                edges.clear()
                try:
                    await client.write_gatt_char(COMMAND_UUID, OB.CMD_ALL_OFF, response=True)
                except Exception:
                    pass
        asyncio.run(main())
