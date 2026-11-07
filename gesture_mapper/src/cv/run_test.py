"""
Visual test for fist detection. Press 'q' to quit.
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark

from detector import _is_fist

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Landmark connections for drawing (index pairs)
_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


def _draw_landmarks(frame, landmarks, h, w):
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    for conn in _CONNECTIONS:
        a, b = conn.start, conn.end
        x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
        x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)


def main():
    base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)  # FaceTime HD (index shifts when iPhone Continuity Camera absent)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Check System Settings → Privacy & Security → Camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print("Camera open. Show a fist. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        label = "none"
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            _draw_landmarks(frame, lm, h, w)
            if _is_fist(lm):
                label = "FIST"

        color = (0, 200, 0) if label == "FIST" else (180, 180, 180)
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.imshow("Neuratect — cv/ test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
