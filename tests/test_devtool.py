"""Hardware-free tests for devtool/ (detector, sample indexing, calibration, session CSV + summary)."""

import struct
import time

import numpy as np

from devtool.ble import BandState
from devtool.emg import BandCal, compute_calibration, detect_offline
from devtool.session import Session, load_events, summarize


def burst(n_rest, n_on, amp_on=40, amp_rest=2, seed=0):
    rng = np.random.default_rng(seed)
    amp = np.r_[np.full(n_rest, amp_rest), np.full(n_on, amp_on)][:, None]
    return np.clip(rng.normal(0, 1, (n_rest + n_on, 8)) * amp, -128, 127).astype(np.int8)


def test_calibration_scales_rest_to_0_and_squeeze_to_1():
    rest, sq = burst(1000, 0), burst(0, 1000, amp_on=30)
    cal = compute_calibration(rest, sq)
    assert cal.calibrated and cal.rest_act_p99 < 0.15 and 0.6 < cal.peak_act_median < 1.1


def test_refractory_and_release():
    cal = compute_calibration(burst(1000, 0), burst(0, 1000, amp_on=30))
    # two 200 ms squeezes with presses ~340 ms apart
    x = np.concatenate([burst(100, 40, seed=1), burst(30, 40, seed=2), burst(200, 0, seed=3)])
    idx = np.arange(len(x))
    assert len(detect_offline(x, idx, cal, 0.35, 250)) == 2
    # refractory 400 ends while the 2nd squeeze is still on: it must be swallowed, not fire late
    assert len(detect_offline(x, idx, cal, 0.35, 400)) == 1
    # one long squeeze never double-fires, however short the refractory
    y = np.concatenate([burst(100, 400, seed=4), burst(100, 0, seed=5)])
    assert len(detect_offline(y, np.arange(len(y)), cal, 0.35, 50)) == 1


def test_dropped_packet_advances_sample_index():
    b = BandState("right", BandCal(), 0.35, 250, hours=0.01)
    pkt = struct.pack("<16b", *range(16))
    for ci in (0, 1, 3, 0):  # characteristic 2 missing
        b.handle_packet(ci, pkt, time.perf_counter())
    assert b.drops == 1 and b.n == 10 and b.n_packets == 4
    assert list(b.valid[:10]) == [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    assert list(b.slot[:4]) == [0, 1, 0, 1] and list(b.emg[1]) == list(range(8, 16))


def test_session_csv_roundtrip(tmp_path):
    b = BandState("right", BandCal(), 0.35, 250, hours=0.01)
    pkt = struct.pack("<16b", *([1] * 16))
    for k in range(200):
        b.handle_packet(k % 4, pkt, time.perf_counter())
    s = Session(tmp_path, "t", {"right": b}, {"threshold": 0.35, "refractory_ms": 250, "smoothing": 0.5,
                                               "cursor_source": "mouse"}, 10, 10, pre_ms=50, post_ms=20)
    s.mark_phase("rest_still")
    s.add_click("right", 100, time.perf_counter(), "rest_still", "false_rest", (0, 0), lambda t: (0, 0))
    s.mark_phase("targets")
    t0 = time.perf_counter() - 1.5
    s.add_click("right", 300, time.perf_counter(), "targets", "hit", (5, 5), lambda t: (5, 5),
                (5, 5, 30, 1), "right", t0, t0 + 0.5)
    s.add_timeout((50, 50, 30, 2), "right", t0, None)
    summary, text = s.close({"right": BandCal()})
    ev = load_events(s.csv_path)
    assert sorted(e["event"] for e in ev) == ["click", "click", "timeout"]  # clicks land after post window
    s2 = summarize(ev)
    assert s2["targets"]["hits"] == 1 and s2["targets"]["targets_timed_out"] == 1
    assert s2["rest"]["rest_still"]["false_per_min"] == 6.0
    assert abs(s2["time_to_target_s"]["median"] - 1.5) < 0.05
    assert "hit rate 100%" in text
    n_rows = sum(1 for _ in open(s.csv_path)) - 1
    assert n_rows == 2 * (10 + 4 + 1) + 1  # 2 clicks x (pre+post+1 samples) + 1 timeout row


def test_out_of_order_packets_do_not_inflate_index():
    b = BandState("right", BandCal(), 0.35, 250, hours=0.01)
    b.new_segment()
    pkt = struct.pack("<16b", *range(16))
    order = [0, 2, 1, 3]  # every group of 4 arrives swapped, one packet per 10 ms
    for k in range(400):
        b.handle_packet(order[k % 4], pkt, 1000.0 + 0.010 * k)
    assert b.out_of_order > 0
    assert b.n <= 2 * 400 + 12  # never more than ~50 ms ahead of real time
