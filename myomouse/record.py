"""Record calibration data for one arm (about 2 minutes of recording).

    python -m myomouse.record --user julius --side right                 # fist (default)
    python -m myomouse.record --user julius --side right --move air      # the original 3D setup
    python -m myomouse.record --user julius --side right --click index   # index lift (experimental)

Saved to data/<user>/index_<side>_<timestamp>.npz (index) or data/<user>/<side>_<timestamp>.npz (fist).

The recording is split into short blocks. Before each block you read what to do and
press Enter; then a 3-2-1 countdown starts it. During a block a live line shows what to
do now and counts down to the next squeeze. Nothing is recorded while you're reading.

The "relax while moving" parts matter most: they teach the classifier that moving the
cursor is NOT a click. --move says how you'll move (table / air) so you practise the right one.

By default a webcam watches your hand while you record (see vision.py): the labels then
follow when your hand actually closed / your index actually lifted, not when the cue said so. Put the laptop so
the camera sees the hand of the arm you record. --no-camera records with cue timing only.
The camera is only used here; the live demo runs on the Myo bands alone.
"""

import argparse
import sys
import time

import numpy as np

from . import config as C
from .gestures import DEFAULT_CLICK, add_style_args, recording_prefix
from .users import add_user_arg, data_dir
from .device import open_band

REST, FIST, IGNORE = 0, 1, -1  # FIST = "click gesture active" (a fist or a lifted index)
TRANSITION_S = 0.25  # samples this soon after a cue change are left out of training

# terminal colours
BOLD, RED, GREEN, DIM, RESET = "\033[1m", "\033[41;97m", "\033[32m", "\033[2m", "\033[0m"
CLEAR_LINE = "\r\033[K"


def protocol(side, click=DEFAULT_CLICK, move=C.MOVE):
    """List of blocks: (title, instructions, [(label, seconds, short cue), ...]).

    click: "fist" or "index" (what to train);  move: "air" or "table" (how you'll move the cursor).
    """
    if move in ("table", "slide"):
        hand = "hand flat on the table" if click == "index" else "forearm on the table"
        if side == C.POINTER_SIDE:
            move_txt = ("slide and turn your hand around on the table like a mouse" if move == "slide"
                        else "turn your forearm left/right on the table and tilt your hand to the side")
        else:
            move_txt = "move your hand around on the table naturally"
    else:
        hand = "arm still"
        move_txt = ("move your forearm around like you're steering a cursor" if side == C.POINTER_SIDE
                    else "move your arm around naturally")

    def clicks(n, active, relax, cue_on, cue_off):
        steps = [(REST, 2.0, cue_off)]  # lead-in so the first one also gets a countdown
        for _ in range(n):
            steps += [(FIST, active, cue_on), (REST, relax, cue_off)]
        return steps

    if click == "index":
        return [
            ("Rest, hand still", f"Relax your {hand}, fingers resting on the table. Keep still.",
             [(REST, 8.0, "relax, hand still")]),
            ("Rest, hand moving", f"Keep the index finger down and relaxed, and {move_txt}.",
             [(REST, 15.0, "relax index, keep moving")]),
            ("Other finger moves", "Tap with your middle finger, press down, spread the fingers, wiggle the\n"
                                   "  thumb - anything EXCEPT lifting the index finger.",
             [(REST, 8.0, "anything but the index")]),
            ("Index lifts", f"{hand[0].upper() + hand[1:]}, still. LIFT your index finger off the table when "
                            "the line turns red,\n  put it back down when it turns green. "
                            "A countdown tells you when the next one comes.",
             clicks(8, 1.5, 2.5, "LIFT", "down")),
            ("Lifts while moving", f"Keep moving the whole time ({move_txt}).\n"
                                   "  LIFT the index when red, put it down (but keep moving) when green.",
             clicks(5, 1.5, 2.5, "LIFT (keep moving)", "down, keep moving")),
            ("Quick clicks", "Hand still. Short, quick index lifts like a mouse click.",
             clicks(6, 0.6, 1.8, "CLICK", "down")),
        ]

    return [
        ("Rest, arm still", f"Relax your hand completely and keep your {'arm' if move == 'air' else hand} still.",
         [(REST, 8.0, "relax, arm still")]),
        ("Rest, arm moving", f"Keep your hand relaxed and {move_txt}.",
         [(REST, 15.0, "relax hand, keep moving")]),
        ("Other hand shapes", "Spread your fingers, point, wave - anything EXCEPT a fist.",
         [(REST, 8.0, "anything but a fist")]),
        ("Fists", "Arm still. SQUEEZE a firm fist when the line turns red, relax when it turns green.\n"
                  "  A countdown tells you when the next squeeze comes.",
         clicks(8, 1.5, 2.5, "SQUEEZE", "relax")),
        ("Fists while moving", f"Keep moving the whole time ({move_txt}).\n"
                               "  SQUEEZE when red, relax your hand (but keep moving) when green.",
         clicks(5, 1.5, 2.5, "SQUEEZE (keep moving)", "relax, keep moving")),
        ("Quick clicks", "Arm still. Short, quick squeezes like a mouse click.",
         clicks(6, 0.6, 1.8, "CLICK", "relax")),
    ]


def bar(frac, width=20):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "·" * (width - n)


def status_line(steps, i, t_in):
    """One live line: what to do now + countdown to the next squeeze."""
    label, dur, cue = steps[i]
    left = dur - t_in
    if label == FIST:
        return f"{RED}{BOLD} >>> {cue:^22} <<< {RESET}  {bar(left / dur)}  {left:4.1f} s"
    nxt = steps[i + 1] if i + 1 < len(steps) else None
    if nxt and nxt[0] == FIST:
        warn = f"{BOLD}{nxt[2]} in {left:3.1f} s{RESET}" if left < 1.5 else f"next: {nxt[2]} in {left:3.1f} s"
        return f"{GREEN}     {cue:^22}     {RESET}  {warn}"
    return f"{GREEN}     {cue:^22}     {RESET}  {bar(1 - t_in / dur)}  {left:4.1f} s left"


def cue_ui(steps, i, t_in):
    """Same as status_line, for the camera window: big red/green banner + countdown."""
    label, dur, cue = steps[i]
    left = dur - t_in
    if label == FIST:
        return {"cue": cue, "color": "red", "sub": f"{left:3.1f} s", "progress": left / dur}
    nxt = steps[i + 1] if i + 1 < len(steps) else None
    if nxt and nxt[0] == FIST:
        return {"cue": cue, "color": "green", "sub": f"{nxt[2]} in {left:3.1f} s"}
    return {"cue": cue, "color": "green", "sub": f"{left:4.1f} s left", "progress": 1 - t_in / dur}


def window(cam):
    """True if instructions go to the camera window instead of the terminal."""
    return cam is not None and cam.preview


def tick(cam, ui=None, seconds=0.03):
    """Wait a moment; keeps the camera window alive (it has to be drawn from the main thread)."""
    if cam is None:
        time.sleep(seconds)
    elif cam.show(ui, max(1, int(seconds * 1000))) in (27, ord("q")):
        raise KeyboardInterrupt


def wait_start(cam, prompt, ui):
    """Wait for SPACE/Enter in the camera window (or Enter in the terminal without one)."""
    if not window(cam):
        input(prompt)
        return
    ui = dict(ui, footer="SPACE = start    ESC = quit")
    while True:
        key = cam.show(ui, 30)
        if key in (13, 10, 32):
            return
        if key in (27, ord("q")):
            raise KeyboardInterrupt


def countdown(cam=None, ui=None):
    for k in (3, 2, 1):
        print(f"{CLEAR_LINE}  starting in {BOLD}{k}{RESET} ...", end="", flush=True)
        end = time.perf_counter() + 1.0
        while time.perf_counter() < end:
            tick(cam, dict(ui or {}, cue=f"starting in {k}", color=None))
    print(CLEAR_LINE, end="")


def main():
    ap = argparse.ArgumentParser()
    add_user_arg(ap, required="--list-cameras" not in sys.argv)
    ap.add_argument("--side", choices=["left", "right"], required="--list-cameras" not in sys.argv)
    add_style_args(ap)
    ap.add_argument("--fake", action="store_true", help="simulated band, for testing")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for Enter between blocks")
    ap.add_argument("--no-camera", action="store_true", help="label by cue timing only, no webcam")
    ap.add_argument("--camera", "--camera-index", dest="camera", default=None,
                    help="webcam: number or part of its name, e.g. --camera 'web camera' "
                         "(default: VISION_CAMERA in config.py, built-in if empty)")
    ap.add_argument("--list-cameras", action="store_true", help="show the connected cameras and exit")
    ap.add_argument("--swap-hands", action="store_true",
                    help="if the camera tracks your other hand (e.g. an already-mirrored webcam)")
    ap.add_argument("--no-preview", action="store_true", help="use the camera but don't show its window")
    args = ap.parse_args()
    if args.list_cameras:
        from .vision import list_cameras, pick_camera_index
        cams = list_cameras()
        chosen = pick_camera_index(args.camera)
        for i, n in cams:
            print(f"  {i}: {n}{'   <- used' if i == chosen else ''}")
        if not cams:
            print("  no cameras listed (pip install cv2-enumerate-cameras)")
        return

    blocks = protocol(args.side, args.click, args.move)
    emg, labels, segs, times = [], [], [], []
    t_ref = time.perf_counter()  # shared clock for EMG and camera
    # what the EMG handler should label incoming samples with; None = not recording
    state = {"steps": None, "bounds": None, "t0": None, "seg0": 0}

    def on_emg(sample):
        steps, bounds, t0 = state["steps"], state["bounds"], state["t0"]
        if steps is None:
            return
        t = time.perf_counter() - t0
        i = int(np.searchsorted(bounds, t, side="right") - 1)
        if not 0 <= i < len(steps):
            return
        lab = steps[i][0] if t - bounds[i] >= TRANSITION_S else IGNORE
        emg.append(sample)
        labels.append(lab)
        segs.append(state["seg0"] + i)
        times.append(time.perf_counter() - t_ref)

    cam = None
    if not args.no_camera:
        from .vision import HandCamera
        cam = HandCamera(args.side, t_ref, index=args.camera, swap_hands=args.swap_hands,
                         preview=not args.no_preview, click=args.click)
        cam.start()

    band = open_band(args.side, fake=args.fake)
    band.add_emg_handler(on_emg)
    band.start()
    total = sum(d for _, _, steps in blocks for _, d, _ in steps)
    print(f"\n{BOLD}Recording {args.side} arm{RESET} (click: {args.click}, move: {args.move}): "
          f"{len(blocks)} blocks, {total:.0f} s of recording.")

    seg0 = 0
    try:
        if window(cam):
            print(f"\n{BOLD}Follow the camera window{RESET} (SPACE there starts each part, ESC quits).")
        if cam is not None:
            wait_start(cam, f"  {DIM}camera check: press Enter when it tracks your hand{RESET} ", {
                "title": f"Camera check - {args.side} arm",
                "lines": [f"Place the laptop so the camera sees your {args.side.upper()} hand, "
                          "also while you move your arm.",
                          ("Lift and lower your index finger: the skeleton should follow it."
                           if args.click == "index" else "Open and close the hand: the skeleton should follow it."),
                          f"Then press SPACE. {len(blocks)} parts, {total:.0f} s of recording in total."]})
        for b, (title, text, steps) in enumerate(blocks, 1):
            dur = sum(d for _, d, _ in steps)
            print(f"\n{BOLD}[{b}/{len(blocks)}] {title}{RESET}  ({dur:.0f} s)\n  {text}")
            intro = {"title": f"[{b}/{len(blocks)}] {title}  ({dur:.0f} s)",
                     "lines": [ln.strip() for ln in text.split("\n")]}
            if not args.no_wait:
                wait_start(cam, f"  {DIM}press Enter when ready{RESET} ", intro)
            countdown(cam, {"title": intro["title"]})

            bounds = np.cumsum([0.0] + [d for _, d, _ in steps])
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
                tick(cam, dict(cue_ui(steps, i, t - bounds[i]), title=f"[{b}/{len(blocks)}] {title}"))
            state["steps"] = None  # stop recording while the next instructions are read
            seg0 += len(steps)
            print(f"{CLEAR_LINE}  {GREEN}done{RESET}")
    except KeyboardInterrupt:
        if cam is not None:
            cam.stop()
        raise SystemExit("\naborted, nothing saved")
    finally:
        band.stop()

    cue = np.asarray(labels)
    extra = {}
    final = cue
    if cam is not None:
        from .vision import vision_labels
        cam_t, cam_r = np.asarray(cam.times), np.asarray(cam.ratios)
        final, report = vision_labels(np.asarray(times), cue, np.asarray(segs), cam_t, cam_r, click=args.click)
        print()
        for line in report:
            print(" ", line)
        extra = {"cam_t": cam_t, "cam_ratio": cam_r}
        if window(cam):  # summary in the window too; closes by itself
            ui = {"title": "Recording done - saved", "footer": "any key = close",
                  "lines": [ln.removeprefix("camera: ") for ln in report]}
            end = time.perf_counter() + 10
            while time.perf_counter() < end and cam.show(ui, 30) == -1:
                pass
        cam.stop()

    out_dir = data_dir(args.user)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{recording_prefix(args.click, args.side)}{time.strftime('%Y%m%d-%H%M%S')}.npz"
    # label = what train uses (camera-refined if available), label_cue = the plain cue timing
    np.savez(out, emg=np.asarray(emg, dtype=np.int8), label=final, label_cue=cue,
             seg=np.asarray(segs), t=np.asarray(times), side=args.side, user=args.user, fs=C.EMG_HZ,
             click=args.click, move=args.move, **extra)
    n = len(emg)
    print(f"\nsaved {out}: {n} samples ({n / C.EMG_HZ:.1f} s, expected ~{total:.0f} s)")
    if n < 0.8 * C.EMG_HZ * total:
        print("WARNING: fewer samples than expected - Bluetooth may be dropping packets.")
    print(f"next: python -m myomouse.train --user {args.user} --side {args.side} --click {args.click}")


if __name__ == "__main__":
    sys.exit(main())
