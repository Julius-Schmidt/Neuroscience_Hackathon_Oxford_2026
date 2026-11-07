# cv — module context

## Non-obvious constraints

- Camera selection is name-based (cv2-enumerate-cameras). Never hardcode an index —
  AVFoundation and OpenCV sort by uniqueID, so index 0 moves when Continuity Camera appears.
- MediaPipe 1.0.0 dropped `mp.solutions.*`; use `tasks.python.vision.HandLandmarker` with
  `hand_landmarker.task` (in src/cv/). Downgrading to 0.10.x doesn't help on arm64.
- cv2.imshow works here, but **main thread only** (macOS forbids otherwise — that's what the
  old "doesn't work" note actually was). A camera preview for the UI module is viable;
  run.py's cbreak KeyListener predates this and isn't evidence against it.
- The detector emits one Gesture per detected hand even when unrecognised (`name="none"`),
  so joystick wrist position is always available regardless of classification.
- MediaPipe normalizes x by frame width, y by height, so raw hypot over normalized coords
  carries a 1.33× vertical bias on 4:3. hand_scale and the pinch distances are isotropic
  (x pre-multiplied by `Gesture.aspect`) — do the same for any new landmark distance, and
  multiply wrist_x by aspect before mixing it with wrist_y. The joystick paths deliberately
  still use raw coords; retuning them wasn't worth the feel change.
- `hand_scale` (landmark 0 → 9) scales with camera distance exactly as apparent hand movement
  does, so dividing a delta by it cancels depth. Recompute per frame — the cost is nothing
  next to the inference it rides along with.
- `pinch_index/middle/ring/pinky` = thumb-tip→tip distance / hand_scale: a continuous click
  signal, hysteresis applied in run.py. Distance-based on purpose — forming a *pose* to click
  moves the anchor you're aiming with.
- **Pose/pinch clash (open).** Poses and pinches on one hand interfere by construction: a pose
  is defined by where the fingers are, a pinch is measured from where they are. index_pointer
  and peace read as pinches and vice versa (live, 2026-08-08). Avoided, not fixed — both are
  unmapped everywhere and the hotbar moved to ring/pinky pinches. Don't re-map either on a
  hand that also pinches without fixing the measurement (candidates: notes/deferred.md).
- open_palm looks too much like shaaa (caused false inventory triggers); unassigned in
  Minecraft, used as the pointer clutch and the joystick anchor release.
- Joystick curves: power (CURVE=0.4) for discrete WASD, linear raw offset for mouse delta,
  shared DEADZONE=0.04.
