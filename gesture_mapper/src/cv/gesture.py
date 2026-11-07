from dataclasses import dataclass
import time


@dataclass
class Gesture:
    name: str         # gesture name, e.g. "fist"
    confidence: float # 0.0–1.0
    timestamp: float  # time.time()
    hand: str = "unknown"        # "left" | "right" | "unknown"
    wrist_x: float = 0.0         # normalised 0–1 wrist x position in frame
    wrist_y: float = 0.0         # normalised 0–1 wrist y position in frame
    hand_scale: float = 0.05     # wrist-to-middle-finger-MCP distance (normalised,
                                  # isotropic frame-height units) — shrinks/grows with
                                  # distance from camera; use as a depth-invariant
                                  # reference (divide movement by this)
    aspect: float = 4 / 3        # frame width/height — multiply x by this to get
                                  # isotropic coords (same units as y / hand_scale)
    pinch_index: float = 10.0    # thumb-tip→index-tip distance / hand_scale
                                  # (small = pinching; default = far apart)
    pinch_middle: float = 10.0   # thumb-tip→middle-tip distance / hand_scale
    pinch_ring: float = 10.0     # thumb-tip→ring-tip distance / hand_scale
    pinch_pinky: float = 10.0    # thumb-tip→pinky-tip distance / hand_scale

    @staticmethod
    def none() -> "Gesture":
        return Gesture(name="none", confidence=1.0, timestamp=time.time())
