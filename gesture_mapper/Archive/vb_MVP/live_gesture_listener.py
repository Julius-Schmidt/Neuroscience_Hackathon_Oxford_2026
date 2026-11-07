# live_gesture_listener.py
# Live loop that detects gestures and maps them to actions

import cv2
import mediapipe as mp
import time
from gesture_recorder import load_templates, normalize_landmarks, select_landmarks_subset # Import normalize_landmarks and select_landmarks_subset
# MAPPINGS will be passed as an argument now
from action_mapper import map_and_perform # map_and_perform will use passed mappings
from gesture_matcher import match_gesture, compare_landmarks # Ensure compare_landmarks is imported

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# TEMPLATES = load_templates() # Loaded inside start_listener now
# MAPPINGS = load_mappings() # Passed as argument

COOLDOWN_TIME = 1.0  # seconds between allowed actions per gesture
last_triggered = {}

def should_trigger(gesture_name):
    now = time.time()
    if gesture_name not in last_triggered or now - last_triggered[gesture_name] > COOLDOWN_TIME:
        last_triggered[gesture_name] = now
        return True
    return False

def start_listener(stop_event, current_mappings, schedule_debug_update_cb, debug_gesture_name_var):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        if schedule_debug_update_cb: # Try to inform UI if possible
            schedule_debug_update_cb("Error", "Webcam Fail")
        return

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    templates = load_templates() # Load fresh templates when listener starts
    last_triggered.clear() # Clear cooldowns from previous runs

    print("Gesture Listener: Listening for gestures... Press ESC in listener window or use Stop button.")
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Gesture Listener: Failed to grab frame.")
            time.sleep(0.1) # Avoid tight loop on error
            continue # Try to re-acquire
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_debug_target_name = None
        if debug_gesture_name_var: # Check if the var was passed
            try:
                current_debug_target_name = debug_gesture_name_var.get()
                if current_debug_target_name == "None": # Treat "None" string as no target
                    current_debug_target_name = None
            except Exception as e: # Handle cases where .get() might fail if var is not a valid StringVar (defensive)
                print(f"[Listener] Error getting debug_gesture_name_var: {e}")
                current_debug_target_name = None
        
        live_debug_score = None # Score for the specific debug target gesture

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                raw_coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                
                # Normalize all 21 live landmarks first
                normalized_all_21_live = normalize_landmarks(raw_coords) # Uses default perform_3d_rotation_normalization=True
                
                if normalized_all_21_live:
                    # Select the subset of landmarks for matching
                    landmarks_subset_for_matching = select_landmarks_subset(normalized_all_21_live)

                    if landmarks_subset_for_matching:
                        # Overall matching for action triggering using the SUBSET
                        overall_match_name, overall_confidence = match_gesture(landmarks_subset_for_matching, templates)
                        
                        if overall_match_name:
                            print(f"[Listener] Matched (Subset): {overall_match_name} (Confidence: {overall_confidence:.2f})")
                        elif overall_confidence > 0.1: 
                            print(f"[Listener] Low overall confidence match (Subset). Highest: {overall_confidence:.2f}")

                        if overall_match_name and overall_confidence >= 0.6 and should_trigger(overall_match_name):
                            print(f"[Listener] Attempting action for (Subset): {overall_match_name}.")
                            map_and_perform(overall_match_name, current_mappings)

                        # Specific debugging score for the UI-selected target gesture
                        if current_debug_target_name and current_debug_target_name in templates:
                            target_template_variations = templates[current_debug_target_name] # These are already subsets
                            best_score_for_target = float('inf')
                            if target_template_variations:
                                for variation_subset in target_template_variations:
                                    # compare_landmarks now receives subset vs subset
                                    score = compare_landmarks(landmarks_subset_for_matching, variation_subset)
                                    if score < best_score_for_target:
                                        best_score_for_target = score
                                live_debug_score = best_score_for_target
                    else:
                        print("[Listener] Could not select subset from normalized live landmarks, skipping match.")
                else:
                    print("[Listener] Could not normalize live landmarks, skipping match.")

                # Standard landmark drawing uses the original hand_landmarks from mediapipe
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # else: no hands detected, live_debug_score remains None or its last value if not reset
        
        # Schedule UI update for the debug score (even if None, to clear it or show N/A via callback logic)
        if schedule_debug_update_cb and current_debug_target_name:
            schedule_debug_update_cb(current_debug_target_name, live_debug_score)
        elif schedule_debug_update_cb and not current_debug_target_name: # No target selected
             schedule_debug_update_cb("None", None) # Signal no target, score is N/A

        cv2.imshow('Gesture Listener Feed', frame)
        # Allow ESC key in the listener window to also signal a stop
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Gesture Listener: ESC pressed, signaling stop.")
            stop_event.set() # Signal the main app that we're stopping
            break
    
    print("Gesture Listener: Loop terminated.")
    if schedule_debug_update_cb: # Final update to clear debug score when listener stops
        schedule_debug_update_cb("None", "N/A")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Gesture Listener: Resources released.")

# if __name__ == "__main__":
#     # This direct execution mode is for standalone testing if needed,
#     # but the app will call start_listener with arguments.
#     class DummyStopEvent:
#         def is_set(self):
#             return False
#     from action_mapper import load_mappings as load_yaml_mappings
#     print("Running gesture_listener.py standalone for testing (no stop signal, ESC to quit window)")
#     start_listener(DummyStopEvent(), load_yaml_mappings()) # Load fresh mappings for standalone 