"""Band connections for the dev tool: both bands on ONE asyncio loop in one background thread.

Every EMG notification carries 2 samples. Each sample gets an index; its band-clock time is
index x 5 ms. Myo sends the 4 EMG characteristics round-robin (0,1,2,3,0,...), so a skip in
that order means a dropped packet: we count it and advance the index by 2 per lost packet,
so later timestamps stay right. Those positions are marked valid=False. A skip is only filled
if host time confirms the time passed; otherwise it's counted as out_of_order and ignored.

Host time of a sample is estimated as t0 + index x 5 ms, where t0 is the minimum over recent
packets of (arrival - index x 5 ms), i.e. the mapping implied by the fastest-arriving packets.
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time
from collections import deque

import numpy as np

from .emg import N_CH, SAMPLE_MS, BandCal, ClickDetector

BATTERY_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
DT = SAMPLE_MS / 1000.0


class BandState:
    """Everything known about one band. Written by the BLE thread, read by the UI."""

    def __init__(self, side: str, cal: BandCal, threshold: float, refractory_ms: float, hours=1.0):
        self.side = side
        self.status = "not started"
        self.connected = False
        self.battery: int | None = None
        cap = int(hours * 3600 * 200)
        self.emg = np.zeros((cap, N_CH), dtype=np.int8)
        self.valid = np.zeros(cap, dtype=bool)
        self.pkt = np.full(cap, -1, dtype=np.int64)
        self.slot = np.full(cap, -1, dtype=np.int8)
        self.arrival = np.zeros(cap, dtype=np.float64)
        self.act = np.zeros(cap, dtype=np.float32)
        self.fact = np.zeros(cap, dtype=np.float32)
        self.n = 0                    # next sample index
        self.n_packets = 0
        self.drops = 0
        self.out_of_order = 0        # skips in characteristic order that were NOT real drops
        self.last_char = None
        self.last_arrival = None
        self._blocks = deque(maxlen=10)   # minima of (arrival - idx*DT) over 100-packet blocks
        self._cur_min, self._cur_count = float("inf"), 0
        self._seg_pk, self._seg_n0, self._seg_anchor = 0, 0, float("inf")
        self.det = ClickDetector(cal, threshold, refractory_ms)
        self.clicks = deque()        # (side, sample_idx, host_time) appended on each press
        self.click_idx = deque(maxlen=50)  # recent click sample indices, for drawing markers
        self.on_imu = None
        self.lock = threading.Lock()

    # --- time mapping ---------------------------------------------------------------------
    @property
    def t0(self):
        vals = list(self._blocks) + ([self._cur_min] if self._cur_count else [])
        return min(vals) if vals else None

    def host_time_of(self, idx: int):
        t0 = self.t0
        return None if t0 is None else t0 + idx * DT

    def _grow(self, need):
        if need <= len(self.emg):
            return
        with self.lock:
            new = max(need, 2 * len(self.emg))
            for name in ("emg", "valid", "pkt", "slot", "arrival", "act", "fact"):
                old = getattr(self, name)
                arr = np.zeros((new,) + old.shape[1:], dtype=old.dtype)
                if name in ("pkt", "slot"):
                    arr[:] = -1
                arr[:len(old)] = old
                setattr(self, name, arr)

    # --- incoming data ------------------------------------------------------------------
    def new_segment(self):
        """Call on (re)connect: keep indices moving with host time across the gap."""
        if self.last_arrival is not None:
            gap = int(round((time.perf_counter() - self.last_arrival) / DT))
            self.n += max(gap, 0)
        self.last_char = None
        self._seg_pk, self._seg_n0, self._seg_anchor = 0, self.n, float("inf")
        self._blocks.clear()
        self._cur_min, self._cur_count = float("inf"), 0

    def handle_packet(self, char_i: int, data: bytes, arrival: float):
        if self.last_char is not None:
            missed = (char_i - (self.last_char + 1)) % 4
            if missed:
                # Only fill the gap if host time agrees time really passed; otherwise the
                # characteristics just arrived out of order and filling would push timestamps ahead.
                latest_ok = self._seg_n0 + (arrival - self._seg_anchor) / DT + 10  # +50 ms slack
                if self.n + 2 * missed + 1 <= latest_ok:
                    self.drops += missed
                    self.n += 2 * missed
                else:
                    self.out_of_order += 1
        self.last_char = char_i
        # segment anchor from packets *received* (independent of gap filling): earliest
        # arrival minus the time those packets cover
        if self._seg_pk == 0:
            self._seg_n0 = self.n
        self._seg_anchor = min(self._seg_anchor, arrival - 2 * self._seg_pk * DT)
        self._seg_pk += 1
        self._grow(self.n + 2)
        p = self.n_packets
        self.n_packets += 1
        for slot in (0, 1):
            s = struct.unpack_from("<8b", data, 8 * slot)
            i = self.n
            self.emg[i] = s
            self.valid[i] = True
            self.pkt[i], self.slot[i], self.arrival[i] = p, slot, arrival
            fired = self.det.push(i, s)
            self.act[i], self.fact[i] = self.det.activation, self.det.fast_activation
            self.n += 1
            if fired:
                self.clicks.append((self.side, i, time.perf_counter()))
                self.click_idx.append(i)
        # t0 estimate from the last sample of this packet
        v = arrival - (self.n - 1) * DT
        self._cur_min = min(self._cur_min, v)
        self._cur_count += 1
        if self._cur_count >= 100:
            self._blocks.append(self._cur_min)
            self._cur_min, self._cur_count = float("inf"), 0
        self.last_arrival = arrival

    def handle_imu(self, data: bytes):
        if self.on_imu:
            self.on_imu(struct.unpack_from("<4h", data, 0))


class BleManager:
    """Connects every band in `bands` (side -> BandState) and keeps reconnecting."""

    def __init__(self, bands: dict[str, BandState], addresses: dict[str, str], imu_side: str | None):
        self.bands, self.addresses, self.imu_side = bands, addresses, imu_side
        self._loop = None
        self._stop = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(5.0)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop = asyncio.Event()
        self._connect_lock = asyncio.Lock()   # connect one band at a time (more reliable)
        tasks = [self._band_task(s) for s in self.bands]
        try:
            self._loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            self._loop.close()

    async def _band_task(self, side):
        from bleak import BleakClient, BleakScanner

        from myomouse.device import (CMD_NEVER_SLEEP, CMD_NORMAL_SLEEP, CMD_RAW_EMG_AND_IMU,
                                     CMD_STREAMS_OFF, COMMAND_UUID, EMG_UUIDS, IMU_UUID, SIDE_COLOR,
                                     cmd_leds, cmd_vibrate)
        st = self.bands[side]
        addr = self.addresses[side]
        while not self._stop.is_set():
            disconnected = asyncio.Event()
            try:
                async with self._connect_lock:
                    st.status = "scanning"
                    dev = await BleakScanner.find_device_by_address(addr, timeout=10.0)
                    if dev is None:
                        raise RuntimeError("not found (asleep? tap logo; close other apps)")
                    st.status = "connecting"
                    client = BleakClient(dev, disconnected_callback=lambda _c: disconnected.set())
                    await client.connect()
                try:
                    await client.write_gatt_char(COMMAND_UUID, CMD_NEVER_SLEEP, response=True)
                    await client.write_gatt_char(COMMAND_UUID, CMD_RAW_EMG_AND_IMU, response=True)
                    await client.write_gatt_char(COMMAND_UUID, cmd_leds(SIDE_COLOR[side]), response=True)
                    st.new_segment()
                    for ci, u in enumerate(EMG_UUIDS):
                        await client.start_notify(
                            u, lambda _ch, d, ci=ci: st.handle_packet(ci, d, time.perf_counter()))
                    if side == self.imu_side:
                        await client.start_notify(IMU_UUID, lambda _ch, d: st.handle_imu(d))
                    await client.write_gatt_char(COMMAND_UUID, cmd_vibrate(1), response=True)
                    st.connected, st.status = True, "streaming"
                    while not self._stop.is_set() and not disconnected.is_set():
                        try:
                            st.battery = int((await client.read_gatt_char(BATTERY_UUID))[0])
                        except Exception:
                            pass
                        try:  # wake every 30 s for battery, immediately on stop/disconnect
                            await asyncio.wait_for(self._either(disconnected), 30.0)
                        except asyncio.TimeoutError:
                            pass
                    if not disconnected.is_set():
                        try:
                            await client.write_gatt_char(COMMAND_UUID, CMD_STREAMS_OFF, response=True)
                            await client.write_gatt_char(COMMAND_UUID, CMD_NORMAL_SLEEP, response=True)
                        except Exception:
                            pass
                finally:
                    st.connected = False
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                if not self._stop.is_set():
                    st.status = "disconnected - retrying"
            except Exception as e:
                st.connected = False
                st.status = f"error: {e}"[:70]
            if not self._stop.is_set():
                await asyncio.sleep(2.0)
        st.status = "stopped"

    async def _either(self, ev):
        a = asyncio.create_task(ev.wait())
        b = asyncio.create_task(self._stop.wait())
        _, pending = await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()


class FakeManager:
    """Simulated bands: rest noise, a contraction while `squeeze[side]` is True.

    Sends 2-sample packets every 10 ms, cycling the 4 characteristic indices like a real Myo.
    drop_rate > 0 skips packets to exercise the drop counter.
    """

    def __init__(self, bands: dict[str, BandState], drop_rate=0.0, seed=0):
        self.bands = bands
        self.squeeze = {s: False for s in bands}
        self.drop_rate = drop_rate
        self.rng = np.random.default_rng(seed)
        self._running = False
        self._threads = []

    def start(self):
        self._running = True
        for side in self.bands:
            t = threading.Thread(target=self._loop, args=(side,), daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(1.0)

    def _loop(self, side):
        st = self.bands[side]
        st.status, st.connected, st.battery = "streaming (fake)", True, 87
        gain = np.array([1.0, 1.4, 1.8, 1.6, 1.2, 0.8, 0.6, 0.7])
        level = 0.0
        t_start = time.perf_counter()
        k = 0
        while self._running:
            due = t_start + k * 0.010
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            target = 1.0 if self.squeeze[side] else 0.0
            level += (target - level) * 0.35  # ~30 ms rise
            samples = []
            for _ in range(2):
                amp = 2.0 + 28.0 * level * gain
                samples.extend(int(v) for v in np.clip(self.rng.normal(0, amp), -128, 127))
            if self.rng.random() >= self.drop_rate:
                st.handle_packet(k % 4, struct.pack("<16b", *samples), time.perf_counter())
            k += 1
        st.connected, st.status = False, "stopped"
