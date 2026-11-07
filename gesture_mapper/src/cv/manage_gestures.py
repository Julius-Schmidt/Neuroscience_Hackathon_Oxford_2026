"""
CLI gesture manager — fully headless, no GUI window needed.

Usage:
  python3 manage_gestures.py record <name>   — record a new gesture (5 samples, auto-captured)
  python3 manage_gestures.py list            — list saved gestures
  python3 manage_gestures.py delete <name>   — delete a gesture
  python3 manage_gestures.py test            — live recognition, prints to terminal
"""

import sys
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from normalizer import normalize
from gesture_library import add, delete, list_gestures, load as load_library
from recognizer import RECOGNITION_THRESHOLD
from camera import index_for_name, list_cameras
from keypress import KeyListener

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_SAMPLES_REQUIRED = 5


def _make_landmarker():
    base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def _open_cap():
    idx = index_for_name()
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open camera at index {idx}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    return cap


def _get_all_landmarks(cap, landmarker):
    """Returns list of landmark sets, one per detected hand (0, 1, or 2)."""
    ok, frame = cap.read()
    if not ok:
        return []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    return result.hand_landmarks if result.hand_landmarks else []


def cmd_record(name: str):
    cameras = list_cameras()
    print(f"\nAvailable cameras: {[f'{i}: {n}' for i,n in cameras]}")
    print(f"Using: {cameras[index_for_name()][1]}\n")
    print(f"Recording '{name}' — {_SAMPLES_REQUIRED} samples.")
    print("Hold your gesture steady, then press ENTER to capture. 'q' to cancel.\n")

    cap = _open_cap()
    landmarker = _make_landmarker()
    samples = []

    try:
        while len(samples) < _SAMPLES_REQUIRED:
            raw = input(f"  Sample {len(samples)+1}/{_SAMPLES_REQUIRED} — "
                        f"hold gesture and press ENTER (or 'q' to cancel): ")
            if raw.strip().lower() == "q":
                print("Cancelled.")
                cap.release()
                landmarker.close()
                return

            # Drain stale frames, grab a fresh one
            for _ in range(5):
                cap.read()
            hands = _get_all_landmarks(cap, landmarker)

            if not hands:
                print("  No hand detected — try again.")
                continue

            samples.append(normalize(hands[0]))
            print(f"  Captured.")

    except KeyboardInterrupt:
        print("\nCancelled.")
        cap.release()
        landmarker.close()
        return

    cap.release()
    landmarker.close()

    template = np.mean(samples, axis=0)
    add(name, template)
    print(f"\nSaved '{name}'.")


def cmd_list():
    gestures = list_gestures()
    if not gestures:
        print("No gestures saved.")
        return
    for name in gestures:
        print(f"  {name}")


def cmd_delete(name: str):
    try:
        delete(name)
        print(f"Deleted '{name}'.")
    except KeyError as e:
        print(e)


def cmd_test():
    library = load_library()
    if not library:
        sys.exit("No gestures saved. Record some first.")

    print(f"Testing against: {', '.join(library.keys())}. Press 'q' to quit.\n")
    cap = _open_cap()
    landmarker = _make_landmarker()
    last_label = None

    try:
        with KeyListener() as keys:
            while not keys.should_quit:
                hands = _get_all_landmarks(cap, landmarker)

                if not hands:
                    if last_label != "no hand":
                        print("no hand")
                        last_label = "no hand"
                    continue

                labels = []
                for lm in hands:
                    vec = normalize(lm).flatten()
                    dists = {n: float(np.linalg.norm(vec - t.flatten())) for n, t in library.items()}
                    best = min(dists, key=dists.get)
                    best_dist = dists[best]
                    dist_str = "  ".join(f"{n}: {d:.3f}" for n, d in dists.items())
                    if best_dist <= RECOGNITION_THRESHOLD:
                        labels.append(f"{best} ({1 - best_dist/RECOGNITION_THRESHOLD:.0%})  [{dist_str}]")
                    else:
                        labels.append(f"no match  [{dist_str}]")

                label = " | ".join(labels)
                if label != last_label:
                    print(label)
                    last_label = label

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped.")
        cap.release()
        landmarker.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "record":
        if len(sys.argv) < 3:
            sys.exit("Usage: manage_gestures.py record <name>")
        cmd_record(sys.argv[2])
    elif cmd == "list":
        cmd_list()
    elif cmd == "delete":
        if len(sys.argv) < 3:
            sys.exit("Usage: manage_gestures.py delete <name>")
        cmd_delete(sys.argv[2])
    elif cmd == "test":
        cmd_test()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
