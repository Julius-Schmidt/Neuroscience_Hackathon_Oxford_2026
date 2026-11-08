"""Record the 5 gestures plus rest for one arm (5 repetitions still + 5 while moving the arm, ~6 min).

    python -m myomouse.gestures11.record --user julius --side right
    python -m myomouse.gestures11.record --user julius --side right --reps 3       # shorter
    python -m myomouse.gestures11.record --user julius --side right --move-reps 0  # arm still only (old way)
    python -m myomouse.gestures11.record --user julius --side right --no-camera   # cue timing only

Setup: forearm resting on the table the whole time (elbow down), camera low in front of the
hand. Holding the arm up keeps the forearm muscles working, which drowns out the gestures.

Blocks: first REST (hand completely limp, then limp while moving the arm), then one block
per repetition (every gesture once, in random order, held for HOLD_S with a neutral pause in
between), then the same again while MOVING the arm (first neutral while moving, then the
gestures while moving: like the fist protocol's "relax / squeeze while moving", so arm
movement while you play isn't new to the model), then a short rest again. Before each block you press SPACE (camera window) or
Enter; nothing is recorded while you read.

By default the webcam watches your hand like in myomouse.record: the labels then follow
when your hand actually moved into and out of each gesture, and repetitions where you did
a different gesture are left out (see gestures11/vision.py). Put the camera so it sees the
whole hand of the arm you record, from the front/side. --no-camera uses the cue timing, with
the start and end of every hold left out.

Tip: record 2-3 times, taking the band off and on in between (same position: mark it on
your skin). train then reports how well it works on a session it has never seen.
"""

import argparse
import sys
import time

import numpy as np

from .. import config as C
from ..device import open_band
from ..record import BOLD, CLEAR_LINE, DIM, GREEN, RED, RESET, bar, countdown, tick, wait_start, window
from ..users import add_user_arg, data_dir
from . import (END_TRIM_S, GESTURES, HOLD_S, IDLE, IGNORE, KEYS, MOVE_NEUTRAL_S, MOVE_REPS, NEUTRAL, RELAX_S,
               REPS, REST, REST_END_S, REST_MOVE_S, REST_STILL_S, SETTLE_S, recording_prefix)

REST_STILL = "hand completely limp, arm still"
REST_MOVE = "hand limp, slide / turn the forearm around"
MOVING = "keep moving the arm"   # cue note of every step done while moving the arm
MOVE_HOW = ("Keep moving your forearm the WHOLE time, the way you will while playing: turn it left/right, "
            "tilt the hand, lift it a little.")
from .vision import picture


def protocol(reps=REPS, seed=None, gestures=None, rest=True, move_reps=MOVE_REPS):
    """List of blocks (title, instructions, steps); steps = [(label, seconds, cue text), ...].
    Rest block, one block per repetition with every gesture once in random order (arm still),
    neutral while moving + the same repetitions while moving the arm, then rest."""
    rng = np.random.default_rng(seed)
    active = [g for g in (gestures or range(len(GESTURES))) if g not in IDLE]
    blocks = []
    if rest:
        blocks.append(("Rest", "Forearm on the table, let the hand go completely limp - no effort at all. "
                               "Halfway through: keep it limp but slide and turn the forearm around.",
                       [(REST, REST_STILL_S, REST_STILL), (REST, REST_MOVE_S, REST_MOVE)]))
    def rep(note):
        steps = [(NEUTRAL, RELAX_S + 1.0, note)]  # lead-in, so the first gesture also gets a "next" warning
        for g in rng.permutation(active):
            steps += [(int(g), HOLD_S, note), (NEUTRAL, RELAX_S, note)]
        return steps
    for r in range(reps):
        blocks.append((f"Repetition {r + 1}", "Hold each gesture while the banner is red, "
                                              "back to neutral when it's green.", rep("")))
    if move_reps:
        blocks.append(("Neutral while moving", f"Hand in NEUTRAL (not limp, not a gesture). {MOVE_HOW}",
                       [(NEUTRAL, MOVE_NEUTRAL_S, MOVING)]))
    for r in range(move_reps):
        blocks.append((f"Moving repetition {r + 1}", f"{MOVE_HOW} Hold each gesture while red, neutral when "
                                                     "green - and never stop moving the arm.", rep(MOVING)))
    if rest:
        blocks.append(("Rest again", "Hand completely limp again, arm still.", [(REST, REST_END_S, REST_STILL)]))
    return blocks


def step_label(label, dur, t_in):
    """Training label for a sample t_in seconds into a step: IGNORE near its start and end."""
    return label if SETTLE_S <= t_in <= dur - END_TRIM_S else IGNORE


def status_line(steps, i, t_in):
    label, dur, note = steps[i]
    left = dur - t_in
    if label == REST:
        return f"{GREEN}     {'REST':^20}     {RESET}  {bar(1 - t_in / dur)}  {left:4.1f} s   {note}"
    if label != NEUTRAL:
        g = GESTURES[label]
        return (f"{RED}{BOLD} >>> {g.name.upper():^20} <<< {RESET}  {bar(left / dur)}  {left:3.1f} s   "
                + (f"{BOLD}KEEP MOVING{RESET}  " if note == MOVING else "") + f"{DIM}{g.how}{RESET}")
    nxt = steps[i + 1][0] if i + 1 < len(steps) else None
    moving = f"  {BOLD}KEEP MOVING{RESET}" if note == MOVING else ""
    if nxt is None:
        return f"{GREEN}     {'relax (neutral)':^20}     {RESET}  {bar(1 - t_in / dur)}  {left:3.1f} s{moving}"
    g = GESTURES[nxt]
    when = f"{BOLD}{g.name} in {left:3.1f} s{RESET}" if left < 1.5 else f"next: {g.name} in {left:3.1f} s"
    return f"{GREEN}     {'relax (neutral)':^20}     {RESET}  {when}{moving}   {DIM}{g.how}{RESET}"


def cue_ui(steps, i, t_in):
    """Same as status_line, for the camera window."""
    label, dur, note = steps[i]
    left = dur - t_in
    if label == REST:
        return {"cue": "REST", "color": "green", "sub": f"{note}   {left:4.1f} s", "progress": 1 - t_in / dur}
    if label != NEUTRAL:
        g = GESTURES[label]
        return {"cue": g.name.upper(), "color": "red", "progress": left / dur,
                "sub": ("KEEP MOVING - " if note == MOVING else "") + f"{g.how}   {left:3.1f} s",
                "image": picture(g.key), "image_label": f"NOW: {g.name}"}
    nxt = steps[i + 1][0] if i + 1 < len(steps) else None
    moving = "KEEP MOVING - " if note == MOVING else ""
    if nxt is None:
        return {"cue": "neutral", "color": "green", "sub": f"{moving}{left:3.1f} s", "progress": 1 - t_in / dur}
    g = GESTURES[nxt]
    return {"cue": "neutral", "color": "green", "sub": f"{moving}next: {g.name} in {left:3.1f} s  ({g.how})",
            "image": picture(g.key), "image_label": f"NEXT: {g.name}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argv = sys.argv[1:] if argv is None else argv
    add_user_arg(ap, required="--list-cameras" not in argv)
    ap.add_argument("--side", choices=["left", "right"], required="--list-cameras" not in argv)
    ap.add_argument("--reps", type=int, default=REPS, help=f"repetitions with the arm still (default {REPS})")
    ap.add_argument("--move-reps", type=int, default=MOVE_REPS,
                    help=f"repetitions while moving the arm, like when playing (default {MOVE_REPS}; 0 = none)")
    ap.add_argument("--fake", action="store_true", help="simulated band, for testing")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for Enter between blocks")
    ap.add_argument("--no-camera", action="store_true", help="label by cue timing only, no webcam")
    ap.add_argument("--camera", default=None,
                    help="webcam: number or part of its name (default: VISION_CAMERA in config.py)")
    ap.add_argument("--list-cameras", action="store_true", help="show the connected cameras and exit")
    ap.add_argument("--swap-hands", action="store_true",
                    help="if the camera tracks your other hand (e.g. an already-mirrored webcam)")
    ap.add_argument("--no-preview", action="store_true", help="use the camera but don't show its window")
    args = ap.parse_args(argv)
    if args.list_cameras:
        from ..vision import list_cameras, pick_camera_index
        try:
            chosen = pick_camera_index(args.camera)
        except SystemExit as e:  # configured camera not plugged in: still list the others
            chosen = None
            print(e)
        for i, n in list_cameras():
            print(f"  {i}: {n}{'   <- used' if i == chosen else ''}")
        return

    blocks = protocol(args.reps, move_reps=args.move_reps)
    seg_label = [st[0] for _, _, steps in blocks for st in steps]  # step id -> gesture
    seg_moving = [st[2] in (REST_MOVE, MOVING) for _, _, steps in blocks for st in steps]  # arm moving
    total = sum(st[1] for _, _, steps in blocks for st in steps)
    emg, labels, segs, times = [], [], [], []
    t_ref = time.perf_counter()  # shared clock for EMG and camera
    state = {"steps": None, "bounds": None, "t0": None, "seg0": 0}

    def on_emg(sample):
        steps, bounds, t0 = state["steps"], state["bounds"], state["t0"]
        if steps is None:
            return
        t = time.perf_counter() - t0
        i = int(np.searchsorted(bounds, t, side="right") - 1)
        if not 0 <= i < len(steps):
            return
        emg.append(sample)
        labels.append(step_label(steps[i][0], steps[i][1], t - bounds[i]))
        segs.append(state["seg0"] + i)
        times.append(time.perf_counter() - t_ref)

    print(f"\n{BOLD}The gestures{RESET} (try each once now, before you start):")
    for g in GESTURES:
        print(f"  {g.name:17s} {DIM}{g.how}{RESET}")
    print(f"\n{BOLD}Rest your forearm on the table{RESET} (elbow down) for the still parts; only the hand/wrist "
          f"moves. Between gestures go back to {BOLD}neutral{RESET}; {BOLD}rest{RESET} = completely limp.")
    if args.move_reps:
        print(f"Then come {BOLD}moving parts{RESET}: keep moving the arm the whole time, the way you will "
              "while playing.\nThat's what makes the gestures work while you move.")

    cam = None
    if not args.no_camera:
        from .vision import GestureCamera
        cam = GestureCamera(args.side, t_ref, index=args.camera, swap_hands=args.swap_hands,
                            preview=not args.no_preview)
        cam.start()

    band = open_band(args.side, fake=args.fake)
    band.add_emg_handler(on_emg)
    band.start()
    print(f"\n{BOLD}Recording {args.side} arm{RESET}: {len(blocks)} blocks, {total:.0f} s of recording.")

    seg0 = 0
    try:
        if window(cam):
            print(f"\n{BOLD}Follow the camera window{RESET} (SPACE there starts each part, ESC quits).")
        if cam is not None:
            check = {"title": f"Camera check - {args.side} arm",
                     "image": picture("neutral"), "image_label": "NEUTRAL",
                     "lines": [f"Rest your {args.side.upper()} forearm on the table. Put the camera low in "
                               "front so it sees the whole hand, also when you turn the palm up/down.",
                               "Then hold your hand in NEUTRAL and press SPACE.",
                               f"{len(blocks)} parts, {total:.0f} s of recording in total."]}
            while True:
                wait_start(cam, f"  {DIM}camera check: hand in neutral, then Enter{RESET} ", check)
                if cam.calibrate_neutral():
                    break
                check["lines"][1] = f"Couldn't see your {args.side} hand - move it into view, then SPACE."
                if not window(cam):
                    print(f"  camera doesn't see your {args.side} hand yet")
        for b, (name, text, steps) in enumerate(blocks, 1):
            dur = sum(st[1] for st in steps)
            title = f"[{b}/{len(blocks)}] {name}  ({dur:.0f} s)"
            print(f"\n{BOLD}{title}{RESET}: {text}")
            if not args.no_wait:
                wait_start(cam, f"  {DIM}press Enter when ready{RESET} ", {"title": title, "lines": [text]})
            countdown(cam, {"title": title})
            bounds = np.cumsum([0.0] + [st[1] for st in steps])
            state.update(steps=steps, bounds=bounds, seg0=seg0, t0=time.perf_counter())
            while True:
                if not band.alive:
                    raise SystemExit(f"\nband disconnected: {band.error}")
                t = time.perf_counter() - state["t0"]
                i = int(np.searchsorted(bounds, t, side="right") - 1)
                if i >= len(steps):
                    break
                line = status_line(steps, i, t - bounds[i])
                if cam is not None:
                    line += f"  {DIM}{cam.status()}{RESET}"
                print(CLEAR_LINE + line, end="", flush=True)
                ui = cue_ui(steps, i, t - bounds[i]) if window(cam) else None
                tick(cam, ui and dict(ui, title=f"[{b}/{len(blocks)}] {name}"))
            state["steps"] = None
            seg0 += len(steps)
            print(f"{CLEAR_LINE}  {GREEN}done{RESET}")
    except KeyboardInterrupt:
        if cam is not None:
            cam.stop()
        raise SystemExit("\naborted, nothing saved")
    finally:
        band.stop()

    cue = np.asarray(labels)
    final, extra = cue, {}
    if cam is not None:
        from .vision import gesture_vision_labels
        cam_t, cam_pts = np.asarray(cam.times), np.asarray(cam.points, dtype=np.float32)
        final, report = gesture_vision_labels(np.asarray(times), cue, np.asarray(segs), seg_label, cam_t, cam_pts)
        # moving parts: turning the hand looks like a pose change to the camera, so it can't time
        # the gestures there - keep the cue timing (start and end of every hold already left out)
        moving = np.isin(np.asarray(segs), np.flatnonzero(seg_moving))
        if moving.any():
            final = np.where(moving, cue, final)
            report.append("camera: moving parts use the cue timing (arm movement fools the camera)")
        print()
        for line in report:
            print(" ", line)
        extra = {"cam_t": cam_t, "cam_pts": cam_pts}
        if window(cam):
            ui = {"title": "Recording done - saved", "footer": "any key = close",
                  "lines": [ln.removeprefix("camera: ") for ln in report]}
            end = time.perf_counter() + 10
            while time.perf_counter() < end and cam.show(ui, 30) == -1:
                pass
        cam.stop()

    out_dir = data_dir(args.user)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{recording_prefix(args.side)}{time.strftime('%Y%m%d-%H%M%S')}.npz"
    # label = what train uses (camera-refined if available), label_cue = the plain cue timing
    np.savez(out, emg=np.asarray(emg, dtype=np.int8), label=final, label_cue=cue, seg=np.asarray(segs),
             seg_label=np.asarray(seg_label), seg_moving=np.asarray(seg_moving), t=np.asarray(times), gestures=np.asarray(KEYS),
             side=args.side, user=args.user, fs=C.EMG_HZ, **extra)
    n = len(emg)
    print(f"\nsaved {out}: {n} samples ({n / C.EMG_HZ:.1f} s, expected ~{total:.0f} s)")
    if n < 0.8 * C.EMG_HZ * total:
        print("WARNING: fewer samples than expected - Bluetooth may be dropping packets.")
    print(f"next: python -m myomouse.gestures11.train --user {args.user} --side {args.side}")


if __name__ == "__main__":
    sys.exit(main())
