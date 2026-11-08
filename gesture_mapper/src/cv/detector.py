"""
Gesture detector: camera frame stream → list[Gesture].

Classifies gestures via template matching (see recognizer.py).
Landmark normalization in normalizer.py.
Supports up to 2 hands simultaneously.
"""

import time
import os
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from .gesture import Gesture
    from .recognizer import recognize
    from .camera import index_for_name
except ImportError:
    from gesture import Gesture
    from recognizer import recognize
    from camera import index_for_name

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


class GestureDetector:
    def __init__(self, camera_index: int | None = None):
        self._camera_index = camera_index if camera_index is not None else index_for_name()
        self._cap: cv2.VideoCapture | None = None

        # Read-only taps for the UI: whoever pumps next_gestures() leaves the
        # latest frame/landmarks/gestures here, so a preview can render off the
        # same camera read the control loop is already doing — never a second
        # VideoCapture on the device. Written only by the pumping thread;
        # readers take the whole reference (never mutate in place).
        self.last_frame = None                 # BGR ndarray, or None
        self.last_hands: list[tuple[str, list]] = []   # (hand, 21 landmarks)
        self.last_gestures: list[Gesture] = []

        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> None:
        if self._cap is not None:      # re-entrant: the GUI hands one detector
            return                     # between preview and control threads
        import sys as _sys
        if _sys.platform == "win32":   # match the index order camera.list_cameras() used
            self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_MSMF)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self._camera_index}. "
                + ("Is another program using it (Teams, Zoom, a myomouse record window)? Close it and retry. "
                   "Also check Settings > Privacy & security > Camera > let desktop apps access your camera."
                   if _sys.platform == "win32" else
                   "Check System Settings → Privacy & Security → Camera.")
            )
        # 640x480 (was 320x240): at 320px wide, landmark quantization is ~1px
        # ≈ one inventory slot of cursor travel — GUI precision needs finer
        # steps. MediaPipe resizes to its own input internally, so inference
        # cost is unchanged; capture/convert cost rises slightly. If the
        # detection rate visibly drops (watch gesture-log cadence), revert.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def close(self) -> None:
        """Releases the camera only. The landmarker stays alive so the same
        detector can be reopened (GUI preview ↔ control handover); it dies
        with the process."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def next_gestures(self) -> list[Gesture]:
        """Read one frame and return a Gesture for each detected hand (0, 1, or 2)."""
        if not self._cap:
            raise RuntimeError("Call open() before next_gestures().")

        ok, frame = self._cap.read()
        if not ok:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        self.last_frame = frame

        if not result.hand_landmarks:
            self.last_hands = []
            self.last_gestures = []
            return []

        gestures = []
        hands_out: list[tuple[str, list]] = []
        now = time.time()
        # MediaPipe normalizes x by frame width and y by height, so a hypot
        # over raw normalized coords is direction-dependent (identical
        # physical distances measure 1.33× larger vertically on a 4:3 frame).
        # Multiply x by the aspect ratio first so every distance below is in
        # uniform frame-height units.
        h, w = frame.shape[:2]
        aspect = (w / h) if h else 4 / 3
        for i, landmarks in enumerate(result.hand_landmarks):
            hand = "unknown"
            if result.handedness and i < len(result.handedness):
                hand = result.handedness[i][0].category_name.lower()
            anchor = landmarks[9]  # base of middle finger (MCP) — joystick control point
            wrist = landmarks[0]
            hand_scale = math.hypot((anchor.x - wrist.x) * aspect, anchor.y - wrist.y)
            # Pinch = fingertip proximity as a fraction of hand size — a
            # continuous click signal (run.py applies hysteresis thresholds),
            # deliberately not a pose template: it barely moves the tracked
            # anchor, unlike forming index_pointer/peace.
            # All four fingers, not just index/middle: the thumb-to-ring and
            # thumb-to-pinky pinches carry the hotbar bindings that used to be
            # the index_pointer/peace poses (those poses perturb the pinch
            # ratios on the same hand — see "pose/pinch clash" in CLAUDE.md).
            thumb_tip = landmarks[4]
            safe_scale = max(hand_scale, 1e-6)
            _TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
            pinches = {}
            for finger, tip_i in _TIPS.items():
                tip = landmarks[tip_i]
                pinches[finger] = math.hypot((thumb_tip.x - tip.x) * aspect,
                                             thumb_tip.y - tip.y) / safe_scale

            match = recognize(landmarks)
            if match is not None:
                name, confidence = match
            else:
                name, confidence = "none", 1.0

            gestures.append(Gesture(
                name=name, confidence=confidence, timestamp=now,
                hand=hand, wrist_x=anchor.x, wrist_y=anchor.y,
                hand_scale=hand_scale, aspect=aspect,
                pinch_index=pinches["index"], pinch_middle=pinches["middle"],
                pinch_ring=pinches["ring"], pinch_pinky=pinches["pinky"],
            ))
            hands_out.append((hand, landmarks))

        self.last_hands = hands_out
        self.last_gestures = gestures
        return gestures

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
