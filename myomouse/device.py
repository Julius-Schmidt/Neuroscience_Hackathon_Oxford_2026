"""Talking to Myo bands over the computer's built-in Bluetooth (bleak), no dongle needed.

Each band object runs its own asyncio loop in a background thread and calls the
registered handlers from there:
    emg handler: (sample: tuple of 8 ints)            at 200 Hz
    imu handler: (quat, accel, gyro) int16 tuples     at 50 Hz
                 quat (w, x, y, z) / 16384, accel (x, y, z) / 2048 = g, gyro (x, y, z) / 16 = deg/s
"""

import asyncio
import json
import math
import struct
import threading
import time

import numpy as np

from . import config as C


def _uuid(short: int) -> str:
    # Myo BLE protocol (Thalmic Labs myohw.h, docs/REFERENCES.md [9])
    return f"d506{short:04x}-a904-deb9-4748-2c7f4a124842"


SERVICE_UUID = _uuid(0x0001)      # advertised by every Myo
COMMAND_UUID = _uuid(0x0401)
IMU_UUID = _uuid(0x0402)
# The Myo spreads raw EMG over 4 characteristics, 2 samples each, sent round-robin.
# You must subscribe to all 4 to get the full 200 Hz.
EMG_UUIDS = [_uuid(0x0105), _uuid(0x0205), _uuid(0x0305), _uuid(0x0405)]

CMD_NEVER_SLEEP = bytes([0x09, 0x01, 0x01])
CMD_NORMAL_SLEEP = bytes([0x09, 0x01, 0x00])
CMD_DEEP_SLEEP = bytes([0x04, 0x00])  # powers the band off; wake it by plugging in USB
CMD_RAW_EMG_AND_IMU = bytes([0x01, 0x03, 0x03, 0x01, 0x00])  # emg=raw, imu=on, classifier=off
CMD_STREAMS_OFF = bytes([0x01, 0x03, 0x00, 0x00, 0x00])


def cmd_vibrate(length=1):  # 1 short, 2 medium, 3 long
    return bytes([0x03, 0x01, length])


def cmd_leds(rgb):
    return bytes([0x06, 0x06, *rgb, *rgb])


SIDE_COLOR = {"right": (0, 0, 255), "left": (255, 0, 0)}


async def discover_myos(seconds=6.0):
    """List (address, name, rssi) of nearby Myos. On macOS the address is a UUID, not a MAC."""
    from bleak import BleakScanner

    found = await BleakScanner.discover(timeout=seconds, return_adv=True)
    out = []
    for addr, (dev, adv) in found.items():
        uuids = [u.lower() for u in (adv.service_uuids or [])]
        name = adv.local_name or dev.name or ""
        if SERVICE_UUID in uuids or "myo" in name.lower():
            out.append((addr, name, adv.rssi))
    return sorted(out, key=lambda x: -x[2])


def default_sides(fake=False) -> str:
    """The bands in bands.json (e.g. just 'right' when only one band was identified).
    Fake mode or no bands.json: both. So with one band you don't need --sides."""
    if fake or not C.BANDS_FILE.exists():
        return "right,left"
    try:
        found = json.loads(C.BANDS_FILE.read_text())
    except Exception:
        return "right,left"
    sides = [s for s in ("right", "left") if s in found]
    return ",".join(sides) or "right,left"


def load_bands():
    if not C.BANDS_FILE.exists():
        raise FileNotFoundError("bands.json missing. Run `python -m myomouse.identify` first.")
    return json.loads(C.BANDS_FILE.read_text())


class BleBand:
    """One Myo over Bluetooth. Call start() (blocks until streaming), then stop()."""

    def __init__(self, side: str, address: str | None = None):
        self.side = side
        if not address:
            bands = load_bands()
            if side not in bands:
                raise SystemExit(f"No {side} band in bands.json (only: {', '.join(bands) or 'none'}). "
                                 "Wear both bands and run `python -m myomouse.identify`.")
            address = bands[side]
        self.address = address
        self.emg_handlers, self.imu_handlers = [], []
        self._thread = None
        self._loop = None
        self._stop = None
        self._ready = threading.Event()
        self.error = None
        self.n_emg = 0

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def add_imu_handler(self, h):
        self.imu_handlers.append(h)

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _on_emg(self, _char, data: bytearray):
        s1 = struct.unpack("<8b", data[:8])
        s2 = struct.unpack("<8b", data[8:16])
        for s in (s1, s2):
            self.n_emg += 1
            for h in self.emg_handlers:
                h(s)

    def _on_imu(self, _char, data: bytearray):
        vals = struct.unpack("<10h", data[:20])
        for h in self.imu_handlers:
            h(vals[:4], vals[4:7], vals[7:10])

    async def _main(self):
        from bleak import BleakClient, BleakScanner

        self._stop = asyncio.Event()
        print(f"[{self.side}] looking for {self.address} ...", flush=True)
        dev = await BleakScanner.find_device_by_address(self.address, timeout=15.0)
        if dev is None:
            raise RuntimeError(f"[{self.side}] band {self.address} not found - is it awake and charged? "
                               "(tap the logo; close Myo Connect / other scripts holding it)")
        async with BleakClient(dev) as client:
            await client.write_gatt_char(COMMAND_UUID, CMD_NEVER_SLEEP, response=True)
            await client.write_gatt_char(COMMAND_UUID, CMD_RAW_EMG_AND_IMU, response=True)
            await client.write_gatt_char(COMMAND_UUID, cmd_leds(SIDE_COLOR[self.side]), response=True)
            for u in EMG_UUIDS:
                await client.start_notify(u, self._on_emg)
            if self.imu_handlers:
                await client.start_notify(IMU_UUID, self._on_imu)
            await client.write_gatt_char(COMMAND_UUID, cmd_vibrate(1), response=True)
            print(f"[{self.side}] streaming", flush=True)
            self._ready.set()
            await self._stop.wait()
            try:
                await client.write_gatt_char(COMMAND_UUID, CMD_STREAMS_OFF, response=True)
                await client.write_gatt_char(COMMAND_UUID, CMD_NORMAL_SLEEP, response=True)
            except Exception:
                pass

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:  # surfaced to the caller via start() / alive
            self.error = e
        finally:
            self._ready.set()
            self._loop.close()

    def start(self, timeout=30.0):
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        if self.error:
            raise self.error
        if not self._ready.is_set():
            raise TimeoutError(f"[{self.side}] connecting took longer than {timeout} s")
        return self

    def stop(self):
        if self._loop and self._stop and self.alive:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(3.0)


class FakeBand:
    """Simulated band for testing without hardware (same interface as BleBand).

    200 Hz EMG with a fist burst every few seconds, 50 Hz IMU sweeping the forearm left/right.
    """

    def __init__(self, side: str, fist_period=3.0, fist_len=0.8, seed=0):
        self.side = side
        self.emg_handlers, self.imu_handlers = [], []
        self.rng = np.random.default_rng(seed)
        self.fist_period, self.fist_len = fist_period, fist_len
        self._running = False
        self._thread = None
        self.error = None
        self.n_emg = 0

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def add_imu_handler(self, h):
        self.imu_handlers.append(h)

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    def fist_at(self, t):
        return (t % self.fist_period) < self.fist_len and t > 1.0

    def _loop(self):
        t0 = time.perf_counter()
        n = 0
        while self._running:
            t = n / C.EMG_HZ
            delay = t0 + t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            amp = 40.0 if self.fist_at(t) else 3.0
            emg = tuple(int(v) for v in np.clip(self.rng.normal(0, amp, C.N_CHANNELS), -128, 127))
            self.n_emg += 1
            for h in self.emg_handlers:
                h(emg)
            if n % 4 == 0:
                # forearm flat, turning left/right; band x-axis is forward, gravity on z
                w_t = 2 * math.pi * t / 6.0
                yaw = 0.3 * math.sin(w_t)
                q = (math.cos(yaw / 2) * 16384, 0, 0, math.sin(yaw / 2) * 16384)
                acc = (0, 0, 2048)  # gravity only (yaw doesn't change it)
                gyro = (0, 0, int(0.3 * math.cos(w_t) * 2 * math.pi / 6.0 * 180 / math.pi * 16))
                for h in self.imu_handlers:
                    h(q, acc, gyro)
            n += 1

    def start(self, timeout=None):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(1.0)


def open_band(side: str, fake=False, seed=0):
    return FakeBand(side, seed=seed) if fake else BleBand(side)
