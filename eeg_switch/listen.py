"""Debug helper.

    python -m eeg_switch.listen                    print every intent event (EEG, CV, EMG)
    python -m eeg_switch.listen --list             list all LSL streams (check the Unicorn's name/type)
    python -m eeg_switch.listen --peek Unicorn     channel labels + live values of one stream (3 s)
"""

import argparse
import time

import numpy as np

from . import config as C


def peek(name: str, seconds: float = 3.0):
    """Show what a stream actually carries, so you can check channel order and units."""
    import pylsl
    found = pylsl.resolve_byprop("name", name, minimum=1, timeout=5.0)
    if not found:
        raise SystemExit(f"no stream named {name!r}. Run --list to see what's there.")
    inlet = pylsl.StreamInlet(found[0])
    info = inlet.info()                                   # full info, including channel labels
    n = info.channel_count()
    labels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label") or "?")
        ch = ch.next_sibling()
    labels += [f"ch{i}" for i in range(len(labels), n)]

    print(f"{info.name()!r} type={info.type()!r} {n} channels @ {info.nominal_srate():.0f} Hz")
    print(f"collecting {seconds:.0f} s ...")
    rows, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        chunk, _ = inlet.pull_chunk(timeout=0.2)
        rows += chunk
    if not rows:
        raise SystemExit("no samples arrived: is the stream started?")
    x = np.asarray(rows, float)
    print(f"got {len(x)} samples ({len(x) / seconds:.0f} per s)\n")
    print(f"{'#':>3} {'label':<12} {'last value':>12} {'mean':>12} {'SD':>10}  looks like")
    for i in range(n):
        sd, mean = x[:, i].std(), x[:, i].mean()
        if sd < 1e-9:
            kind = "constant (battery/flag?)"
        elif np.all(np.diff(x[:, i]) >= 0) and sd > 1:
            kind = "counter"
        elif 1 <= sd <= 200:
            kind = "EEG in µV" if i < 8 else "signal"
        elif sd < 1e-3:
            kind = "too small: volts, not µV?"
        else:
            kind = "very large / noisy"
        print(f"{i:>3} {labels[i]:<12} {x[-1, i]:>12.2f} {mean:>12.2f} {sd:>10.2f}  {kind}")
    print("\nFirst 8 rows should be EEG in µV (SD roughly 5-50, Fz C3 Cz C4 Pz PO7 Oz PO8).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--peek", metavar="STREAM_NAME")
    a = ap.parse_args()
    import pylsl
    if a.list:
        for i in pylsl.resolve_streams(wait_time=3.0):
            print(f"name={i.name()!r} type={i.type()!r} channels={i.channel_count()} "
                  f"rate={i.nominal_srate()} format={i.channel_format()} source_id={i.source_id()!r}")
        return
    if a.peek:
        peek(a.peek)
        return
    from .events import EventInlet
    print(f"waiting for '{C.EVENT_STREAM['name']}' producers ...")
    inlet = EventInlet(timeout=30.0)
    last_refresh = time.monotonic()
    while True:
        for ev in inlet.poll():
            print(ev)
        if time.monotonic() - last_refresh > 5:          # pick up producers started later
            inlet.refresh(timeout=0.2)
            last_refresh = time.monotonic()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
