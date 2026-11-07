# Gesture Mapper Project Instructions

## Project Overview  
The Gesture Mapper is a tool enabling users to define custom hand gestures via webcam, mapping them to computer commands (e.g., key presses, analogue controls) with configurable transformation invariance (displacement, scale, rotation). Built on OpenCV and MediaPipe for real-time 21-keypoint hand detection, the project aims for a Scratch-like visual environment for gesture combinations and logic. As of September 4, 2025, the MVP (v1.0.0) is stable, featuring a PyQt5 GUI with a Landmark Inspector for interactive gesture creation. The current focus is integrating immersive Minecraft controls, using two hands for a "touch screen in the air" experience: left hand as a joystick for movement intensity (WASD), right hand for perspective intensity (mouse look/rotation). Palm centroids track position on a virtual plane, with finger-based static gestures (e.g., jump, crouch) and fist toggles to enable/disable joystick modes. Modularity ensures future app-specific profiles (e.g., "Minecraft Controller").

## Core Objectives  
- Deliver fluid Minecraft navigation via two-hand joystick emulation: left hand for movement intensity, right hand for look direction intensity.  
- Implement palm centroid tracking for analogue control, preserving finger freedom for static gestures, with fist-based joystick toggling.  
- Ensure smoothness through user-tunable sensitivity settings for responsive gameplay.  
- Implement a Profile-Based System (e.g., JSON-loaded maps for different apps), prioritizing Minecraft’s core move/look controls as the *first of many* possible profiles.
- Future-proof for gesture sequences and visual programming without immediate implementation.

## MVP Scope (Minecraft profile)  
- **Features**:  
  - Two-hand detection (MediaPipe, assign left/right by x-position).  
  - Joystick Emulation: Palm centroid (avg of wrist + finger bases, points 0,5,9,13,17) tracks position on a virtual plane; displacement from neutral (calibrated at start) controls intensity (e.g., left hand forward = hold W with speed proportional to distance, right hand right = faster mouse rotation).  
  - Joystick Toggle: Fist gesture (e.g., thumb-to-index distance < 0.1) on respective hand enables/disables joystick mode to prevent unintended inputs.  
  - Static Gestures: Non-conflicting and primarily finger-based actions. For example, a 'pinch' (space/jump) can be performed while a joystick is active, but the 'fist' gesture will be reserved exclusively for toggling the joystick inacse the user wants to exclusively access static gestures.
  - Sensitivity Controls: Configurable scalars (e.g., UI sliders, default 1.0) for tuning movement/look intensity.  
  - Analogue Mapping: Centroid deltas to pyautogui key holds (WASD, pulsed for precision) or mouse deltas (look); ensure smooth bursts for small movements.  
- **Tech Stack**: Python, OpenCV, MediaPipe, PyQt5 (GUI/threading), pyautogui; extend HandCapture for two hands.  
- **Success Criteria**: Smooth Minecraft navigation (walk/look without jitter, >80% gesture accuracy, <100ms latency); Fist toggle must have a >95% success rate in preventing misfires during a 5-minute gameplay test." This makes the success condition less subjective; tested with varied hand sizes/distances.  
- **Out of Scope**: Full menu access, complex sequences, multi-app profile loading (plan structurally, implement later).

## Next Steps (Post-MVP)  
1. **Joystick Emulation**: Implement centroid tracking for left (movement) and right (look) hands; add fist toggle to enable/disable joystick mode per hand.  
2. **Dynamic Gestures**: Track centroids over frames for intensity-based control (e.g., swipe speed); adapt normalization to preserve displacement for joystick.  
3. **Interim UI Enhancements**: Add PyQt5 dropdowns for mapping (e.g., select key_press, key) and sliders for sensitivity; include calibration button.  
4. **Robustness**: Handle hand swaps (reassign via x-position) or temporary loss (fallback to single-hand mode).  
5. **Future Prep**: Structure configs for app profiles (e.g., "minecraft.json") and sequences (e.g., hold crouch + move).

## Development Guidelines  
- **Keep It Lean**: Prioritize Minecraft move/look with fist toggle; reuse MVP modules (e.g., HandCapture, GestureDetector) for two-hand support.  
- **Modular Design**:  
  - **Input Capture**: Extend to process two hands (list of landmarks); compute centroids for joystick input.  
  - **Gesture Detection**: Separate left/right logic; add fist detection to toggle joystick mode; normalize per-hand (disable displacement for intensity). This will require a state machine or boolean flags (e.g., is_left_joystick_active) within the detection logic to track toggle states. 
  - **Command Execution**: Map centroid deltas to key holds (WASD) or mouse moves; pulse keys for precise bursts.  
  - **Configuration**: Per-hand JSON sections; support toggle states and sensitivity.  
  - **GUI**: Add fist toggle feedback, sensitivity sliders, and calibration in PyQt5. "visual dead zone representation", Showing the user the calibrated neutral point and the dead zone circle on the video feed is extremely helpful for debugging and usability.
- **User Focus**: At startup, guide the user through a calibration step where they hold their hands steady. Visualize the captured neutral point and the joystick dead zone directly on the video feed; provide real-time centroid/joystick visuals; use dropdowns for intuitive mapping.  
- **Testing**: Playtest in Minecraft for smoothness (navigate terrain, turn head); validate fist toggle reliability; test across webcam distances.  
- **Performance**: Average centroids over 3-5 frames for smoothness; use dead zones for small movements; keep <100ms latency.  
- **Extensibility**: Use list-based structures for hands (e.g., `hands_landmarks`) for multi-hand scalability; design configs as loadable profiles.

## Sample Config (Minecraft-Specific)  
```json  
{  
  "transformations": {  
    "displacement_invariant": false,  
    "scale_invariant": true,  
    "rotation_invariant": true,  
    "per_hand_overrides": {  
      "left": {"displacement_invariant": false},  
      "right": {"displacement_invariant": false}  
    }  
  },  
  "hands": {  
    "left": {  
      "type": "joystick",  
      "centroid_points": [0,5,9,13,17],  
      "toggle_gesture": "fist",  
      "mappings": {  
        "forward": {"key": "w", "sensitivity": 1.2},  
        "backward": {"key": "s", "sensitivity": 1.2},  
        "left": {"key": "a", "sensitivity": 1.2},  
        "right": {"key": "d", "sensitivity": 1.2}  
      }  
    },  
    "right": {  
      "type": "joystick",  
      "centroid_points": [0,5,9,13,17],  
      "toggle_gesture": "fist",  
      "mappings": {  
        "look_up": {"mouse_delta": "y_negative", "sensitivity": 2.0},  
        "look_down": {"mouse_delta": "y_positive", "sensitivity": 2.0},  
        "look_left": {"mouse_delta": "x_negative", "sensitivity": 2.0},  
        "look_right": {"mouse_delta": "x_positive", "sensitivity": 2.0}  
      }  
    }  
  },  
  "gestures": [  
    {"name": "fist", "hand": "any", "conditions": [{"type": "distance", "points": [4,8], "max": 0.1}], "on_perform_mapping": {"key": "space"}},  
    {"name": "open_palm", "hand": "any", "conditions": [{"type": "distance", "points": [4,8], "min": 0.2}], "on_perform_mapping": {"key": "shift"}},  
    {"name": "pinch", "hand": "any", "conditions": [{"type": "distance", "points": [4,8], "max": 0.05}], "on_perform_mapping": {"key": "e"}}  
  ]  
}  
```

## Pseudo Code Foundation  
```python  
# Main Loop (Minecraft Control)  
config = load_config('minecraft.json')  
video_thread = VideoThread(hand_capture=HandCapture(max_hands=2), gesture_detector=GestureDetector(config))  
video_thread.start()  # Signals: update_frame, detected_gesture, landmarks_data  

# In VideoThread (gesture_detection.py)  
def process_frame(landmarks_list):  # [left_landmarks, right_landmarks]  
    if len(landmarks_list) >= 1:  
        left, right = assign_hands(landmarks_list)  # By x-position (leftmost = left)  
        left_active = not detect_fist(left, config['gestures'])  # Toggle joystick off if fist  
        right_active = not detect_fist(right, config['gestures'])  
        if left_active:  
            left_centroid = calculate_centroid(left, config['hands']['left']['centroid_points'])  
            left_delta = delta_from_neutral(left_centroid, calibration_neutral_left)  
            execute_analogue('left', left_delta, config['hands']['left']['mappings'])  # e.g., pulse W if delta_y > dead_zone  
        if right_active:  
            right_centroid = calculate_centroid(right, config['hands']['right']['centroid_points'])  
            right_delta = delta_from_neutral(right_centroid, calibration_neutral_right)  
            execute_analogue('right', right_delta, config['hands']['right']['mappings'])  # e.g., mouse moveRel(dx, dy)  
        finger_gesture = detect_static([left, right], config['gestures'])  # Check both hands for fist, pinch, etc.  
        if finger_gesture: execute_command(finger_gesture, config['gestures'])  
    emit_signals(frame, finger_gesture, landmarks_list)  # Update GUI  

# In GUI (main.py) - Calibration and Sensitivity  
def calibrate_neutral():  
    # Prompt user to hold hands steady for 2s, average centroids  
    self.neutral_left = avg_centroid(left_frames)  
    self.neutral_right = avg_centroid(right_frames)  
def update_sensitivity(self, hand, value):  
    config['hands'][hand]['sensitivity'] = value  # Update via slider  
```

## Roadmap (4 Weeks, ~20 Hours)  
- **Weeks 1-2**: Two-hand detection, centroid tracking, fist toggle for joystick; map to Minecraft move/look.  
- **Week 3**: Add sensitivity sliders and static gestures (fist, open palm, pinch); refine burst pulsing for precision.  
- **Week 4**: Calibration routine, playtest in Minecraft, polish smoothness (e.g., dead zones, frame averaging).  

## Risks & Mitigations  
- **Jittery Joystick**: Use 3-5 frame averaging, dead zones (e.g., ignore |delta| < 0.05); test sensitivity ranges.  
- **Hand Misassignment**: Reassign left/right dynamically by x-position; allow manual override in UI.  
- **Toggle Misfires**: Strict fist detection (tight thresholds); visual feedback in GUI for toggle state.  
- **Profile Complexity**: Keep JSON modular with per-hand sections; test single-app loading