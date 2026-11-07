# gesture_recorder.py
# Module for capturing and recording hand gestures using MediaPipe + OpenCV

import cv2
import mediapipe as mp
import time
import pickle
import os
import math
from tkinter import messagebox # For showing errors from this module if app_root is provided
import numpy as np # Added import

# Import necessary functions from gesture_matcher
from gesture_matcher import compare_landmarks, euclidean_distance

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

PREPARATION_TIME = 5

# Constants for interactive editor
INTERACTIVE_MAX_AVG_DISTANCE = 0.5  # For converting score to percentage (lower score = higher %)
# Max distance for an individual landmark connection to be considered "good" (green)
# Tune this value based on observation. Normalized distances are small.
INTERACTIVE_LANDMARK_MAX_DIST_FOR_COLOR = 0.15 

# Define which landmark indices to keep after normalization for matching and template saving
# Keeping: Wrist, Thumb (CMC, MCP, TIP), Index (MCP, TIP), Middle (MCP, TIP), Ring (MCP, TIP), Pinky (MCP, TIP)
# Indices:   0,     1,    2,    4,        5,    8,          9,    12,        13,    16,        17,    20
LANDMARK_INDICES_TO_KEEP = [0, 1, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]
# Ensure landmarks crucial for normalization (0, 5, 9, 17) are in this list. They are.

def select_landmarks_subset(landmarks_full_list):
    """
    Selects a subset of landmarks based on LANDMARK_INDICES_TO_KEEP.
    Assumes landmarks_full_list is a list of 21 landmark tuples/lists.
    """
    if not landmarks_full_list or len(landmarks_full_list) != 21:
        # print("[Select Subset] Input landmarks list is not valid (must be 21).")
        return None # Or return landmarks_full_list to bypass selection if error
    
    subset = [landmarks_full_list[i] for i in LANDMARK_INDICES_TO_KEEP if i < len(landmarks_full_list)]
    
    # Sanity check: did we get the expected number of landmarks?
    if len(subset) != len(LANDMARK_INDICES_TO_KEEP):
        # This should not happen if LANDMARK_INDICES_TO_KEEP contains valid indices for a 21-landmark list.
        # print(f"[Select Subset] Error: Expected {len(LANDMARK_INDICES_TO_KEEP)} landmarks, got {len(subset)}.")
        return None # Or return original list to be safe
    return subset

def normalize_landmarks(landmarks_input, perform_3d_rotation_normalization=True, scale_normalize=True):
    """
    Normalize hand landmarks for translation, 3D rotation, and scale.
    Based on the logic provided by the user, adapted to fit existing parameters.
    Outputs 21 normalized landmarks.
    """
    # Default indices from MediaPipe
    wrist_idx = 0
    mcp_middle_idx = 9
    mcp_index_idx = 5

    if not landmarks_input or len(landmarks_input) < 21: # Check against 21 initially
        print("[NormalizeV2] Not enough landmarks provided.")
        return None

    # Ensure numpy is imported (should be at module level)
    try:
        import numpy as np
    except ImportError:
        print("NumPy not found in normalize_landmarks")
        return None

    # Convert landmarks to numpy array of floats for manipulation
    landmarks = []
    for lm in landmarks_input:
        if isinstance(lm, (list, tuple)) and len(lm) == 3:
            try:
                landmarks.append(np.array(lm, dtype=float))
            except ValueError:
                print(f"[NormalizeV2] Landmark contains non-numeric data: {lm}")
                return None
        else:
            print(f"[NormalizeV2] Unexpected landmark format: {lm}")
            return None
    landmarks = np.array(landmarks)

    if landmarks.shape[0] <= max(wrist_idx, mcp_middle_idx, mcp_index_idx):
        print("[NormalizeV2] Landmark array too small after conversion.")
        return None
        
    # Reference points from the original input (now as numpy arrays)
    wrist_point_orig = landmarks[wrist_idx].copy() # Original location of wrist
    
    # 1. Translation normalization: Move wrist to origin
    # All landmarks are shifted relative to the wrist's original position.
    translated_landmarks = landmarks - wrist_point_orig

    # After translation, the wrist landmark (at wrist_idx) should be at [0,0,0].
    # Points needed for scale and rotation are now relative to this new origin.
    mcp_middle_translated = translated_landmarks[mcp_middle_idx]
    mcp_index_translated = translated_landmarks[mcp_index_idx]

    # 2. Scale normalization (optional)
    current_landmarks_for_rotation = translated_landmarks # Start with translated
    if scale_normalize:
        # Calculate scale as the distance from (translated) wrist to (translated) middle MCP
        # Since wrist is now origin, this is just the norm of mcp_middle_translated
        scale = np.linalg.norm(mcp_middle_translated)
        if scale > 1e-9:  # Avoid division by zero or extreme scaling
            current_landmarks_for_rotation = translated_landmarks / scale
            # Update points used for rotation to their scaled versions
            mcp_middle_scaled = current_landmarks_for_rotation[mcp_middle_idx]
            mcp_index_scaled = current_landmarks_for_rotation[mcp_index_idx]
        else:
            print("[NormalizeV2] Scale is too small, skipping scale normalization.")
            # If scale is too small, mcp_middle_translated is near zero.
            # Rotation might be unstable, but proceed with unscaled translated for rotation.
            mcp_middle_scaled = mcp_middle_translated # Use unscaled for rotation basis if scale failed
            mcp_index_scaled = mcp_index_translated
    else:
        mcp_middle_scaled = mcp_middle_translated # Use translated for rotation basis if no scaling
        mcp_index_scaled = mcp_index_translated

    # 3. Rotation normalization (optional)
    final_output_landmarks = current_landmarks_for_rotation # Default output

    if perform_3d_rotation_normalization:
        # Define a local coordinate system for the hand using scaled (or translated if no scale) landmarks
        # Primary axis (let's call it hand_x_axis internally for this logic):
        # Vector from (new origin) wrist to (scaled/translated) middle MCP.
        hand_x_axis_vec = mcp_middle_scaled # Since wrist is at origin (0,0,0)
        norm_hand_x = np.linalg.norm(hand_x_axis_vec)

        if norm_hand_x < 1e-9:
            print("[NormalizeV2] Hand X-axis (wrist-middleMCP) is near zero. Skipping rotation.")
        else:
            hand_x_axis = hand_x_axis_vec / norm_hand_x

            # Secondary vector for defining the plane: from (new origin) wrist to (scaled/translated) index MCP.
            vec_to_index_for_plane = mcp_index_scaled # Since wrist is at origin

            # Temporary Hand Z-axis: cross product of hand_x_axis and vec_to_index_for_plane
            hand_z_axis_temp = np.cross(hand_x_axis, vec_to_index_for_plane)
            norm_hand_z_temp = np.linalg.norm(hand_z_axis_temp)

            if norm_hand_z_temp < 1e-9: # Hand X-axis and vec_to_index_for_plane are collinear
                # print("[NormalizeV2] Hand Z-axis (temp) is near zero. Points might be collinear. Trying fallback for Z.")
                # Fallback: If index finger is collinear with middle finger line (X-axis),
                # try to cross with a global axis (e.g., global Z) to form a Z, unless hand_X is global Z.
                global_z_for_fallback = np.array([0.0, 0.0, 1.0])
                if np.linalg.norm(np.cross(hand_x_axis, global_z_for_fallback)) > 1e-9:
                    hand_z_axis_temp = np.cross(hand_x_axis, global_z_for_fallback)
                else: # hand_x_axis was aligned with global_z, try global_y
                    global_y_for_fallback = np.array([0.0, 1.0, 0.0])
                    hand_z_axis_temp = np.cross(hand_x_axis, global_y_for_fallback)
                
                norm_hand_z_temp = np.linalg.norm(hand_z_axis_temp) # Recalculate norm for fallback
                if norm_hand_z_temp < 1e-9:
                    print("[NormalizeV2] Fallback for Hand Z-axis also failed. Skipping rotation.")
                    # final_output_landmarks remains current_landmarks_for_rotation
                else:
                    hand_z_axis = hand_z_axis_temp / norm_hand_z_temp
                    valid_z = True
            else:
                hand_z_axis = hand_z_axis_temp / norm_hand_z_temp
                valid_z = True
            
            if valid_z:
                # Hand Y-axis: cross product of Z and X to ensure orthogonality and right-handed system.
                hand_y_axis = np.cross(hand_z_axis, hand_x_axis)
                # This should be normalized if hand_z_axis and hand_x_axis are ortho-normal.
                # For safety, explicitly normalize.
                norm_hand_y = np.linalg.norm(hand_y_axis)
                if norm_hand_y < 1e-9:
                     print("[NormalizeV2] Hand Y-axis is near zero. Skipping rotation.")
                else:
                    hand_y_axis = hand_y_axis / norm_hand_y
                    
                    # Construct the rotation matrix. The rows are the new basis vectors (for transforming points)
                    # Or, columns are new basis vectors if we think of transforming the basis itself.
                    # To transform points FROM original TO new basis: P_new = R * P_old
                    # R = [x_new_basis_in_terms_of_old.T, y_new_basis_in_terms_of_old.T, ...]
                    # Here, hand_x_axis, hand_y_axis, hand_z_axis ARE the new basis vectors in the old coord system.
                    # To project P_old onto these: P_new_x = dot(P_old, hand_x_axis), etc.
                    # So, the matrix columns should be these vectors if P_old is a column vector.
                    # If P_old is a row vector: P_new = P_old * R_transpose
                    # Where R_transpose has hand_x_axis, hand_y_axis, hand_z_axis as columns.
                    # Equivalent to: P_new_coord_i = dot(P_old_vector, new_basis_vector_i)
                    
                    # Rotation matrix: columns are the new basis vectors X, Y, Z
                    # This matrix transforms vectors from the new hand coord system TO the original system.
                    # To go FROM original system TO hand system, we need its inverse, which is its transpose (since it's orthonormal).
                    # rotation_matrix_to_hand_system = np.array([hand_x_axis, hand_y_axis, hand_z_axis]) # Rows are new axes
                    # final_output_landmarks = np.dot(current_landmarks_for_rotation, rotation_matrix_to_hand_system.T)

                    # Simpler projection: For each point, its new x is its projection on hand_x_axis, etc.
                    rotated_landmarks_list = []
                    for lm_point in current_landmarks_for_rotation:
                        new_x = np.dot(lm_point, hand_x_axis)
                        new_y = np.dot(lm_point, hand_y_axis)
                        new_z = np.dot(lm_point, hand_z_axis)
                        rotated_landmarks_list.append(np.array([new_x, new_y, new_z]))
                    final_output_landmarks = np.array(rotated_landmarks_list)
                    # print("[NormalizeV2] Applied 3D rotation.")

    # Convert back to list of (Python) tuples, ensuring all 21 are returned
    return [tuple(pt) for pt in final_output_landmarks]

def interactive_gesture_editor(app_root_for_dialogs, gesture_name_to_process, is_new_gesture):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        if app_root_for_dialogs:
            messagebox.showerror("Webcam Error", "Could not open webcam.", parent=app_root_for_dialogs)
        else:
            print("Error: Could not open webcam.")
        return False

    hands_solution = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
    
    window_title = f"Edit: {gesture_name_to_process}" if not is_new_gesture else f"Record New: {gesture_name_to_process}"
    cv2.namedWindow(window_title)

    current_gesture_variations = [] # Stores subsets
    gesture_file_path = os.path.join(TEMPLATES_DIR, f"{gesture_name_to_process}.pkl")

    if not is_new_gesture and os.path.exists(gesture_file_path):
        try:
            with open(gesture_file_path, "rb") as f:
                loaded_data = pickle.load(f)
                # Ensure loaded data is a list of variations, and each variation is a list (of 12 landmarks)
                if isinstance(loaded_data, list) and \
                   (not loaded_data or (isinstance(loaded_data[0], list) and \
                                         (not loaded_data[0] or isinstance(loaded_data[0][0], tuple)))):
                    current_gesture_variations = loaded_data
                else:
                    print(f"Warning: Data for '{gesture_name_to_process}' has old/unexpected format. Starting fresh.")
        except Exception as e:
            print(f"Error loading existing variations for '{gesture_name_to_process}': {e}")
            if app_root_for_dialogs:
                messagebox.showwarning("Load Error", f"Could not load existing data for {gesture_name_to_process}. Proceeding as if new.", parent=app_root_for_dialogs)

    reference_landmarks_normalized_subset = None # This will store the 12-landmark subset for reference
    display_save_message_until = 0
    saved_successfully = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_solution.process(rgb_frame)

        display_text_y = 30
        cv2.putText(frame, "Q: Quit | SPACE: Set Ref | S: Save Ref", (10, display_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        display_text_y += 25

        if reference_landmarks_normalized_subset:
            cv2.putText(frame, f"Ref Set for '{gesture_name_to_process}'. Press 'S' to save.", (10, display_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            display_text_y += 25
        else:
            cv2.putText(frame, "No Ref. Position hand & press SPACE.", (10, display_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            display_text_y += 25

        live_coords_subset_for_comparison = None

        if results.multi_hand_landmarks:
            for hand_landmarks_proto in results.multi_hand_landmarks:
                raw_coords_frame = [(lm.x, lm.y, lm.z) for lm in hand_landmarks_proto.landmark]
                
                # Normalize all 21 landmarks first using the new logic
                # perform_3d_rotation_normalization=True and scale_normalize=True by default in this call
                normalized_all_21 = normalize_landmarks(raw_coords_frame) 
                
                if normalized_all_21:
                    live_coords_subset_for_comparison = select_landmarks_subset(normalized_all_21)

                    if live_coords_subset_for_comparison and reference_landmarks_normalized_subset:
                        mp_drawing.draw_landmarks(frame, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS)
                        score = compare_landmarks(live_coords_subset_for_comparison, reference_landmarks_normalized_subset)
                        similarity_perc = max(0.0, (1.0 - (score / INTERACTIVE_MAX_AVG_DISTANCE))) * 100
                        cv2.putText(frame, f"Similarity: {similarity_perc:.1f}% (Subset)", (10, display_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 50), 2)
                        display_text_y += 25
                    elif live_coords_subset_for_comparison: # Hand detected and processed, but no reference yet
                        mp_drawing.draw_landmarks(frame, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS)
                else:
                    # print("Normalization of live landmarks failed.")
                    pass # No hand or normalization failed, draw nothing or only frame
                break # Process only the first detected hand
        
        if time.time() < display_save_message_until:
             cv2.putText(frame, "SAVED!", (frame_width // 2 - 70, frame_height // 2), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)

        cv2.imshow(window_title, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '): # SPACE
            if live_coords_subset_for_comparison:
                reference_landmarks_normalized_subset = live_coords_subset_for_comparison
                print(f"Reference pose (subset) updated for '{gesture_name_to_process}'. Press 'S' to save this new reference.")
            else:
                print("No hand detected or processed to set as reference.")
        elif key == ord('s'):
            if reference_landmarks_normalized_subset:
                if not current_gesture_variations or compare_landmarks(reference_landmarks_normalized_subset, current_gesture_variations[-1]) > 1e-5:
                    current_gesture_variations.append(reference_landmarks_normalized_subset)
                    print(f"Saved/Appended gesture variation (subset) for '{gesture_name_to_process}'. Total variations: {len(current_gesture_variations)}")
                    saved_successfully = True
                    display_save_message_until = time.time() + 1.0
                else:
                    print("This reference (subset) is identical to the last saved variation. Not re-adding.")
                
                # Save all variations (which are subsets) to file
                try:
                    os.makedirs(TEMPLATES_DIR, exist_ok=True)
                    with open(gesture_file_path, "wb") as f:
                        pickle.dump(current_gesture_variations, f)
                except Exception as e:
                    error_msg = f"Error saving gesture '{gesture_name_to_process}': {e}"
                    print(error_msg)
                    if app_root_for_dialogs:
                        messagebox.showerror("Save Error", error_msg, parent=app_root_for_dialogs)
            else:
                print("No reference pose (subset) to save. Press SPACE to capture one first.")

    cap.release()
    cv2.destroyAllWindows()
    hands_solution.close()
    return saved_successfully

# The old record_gesture function - can be kept for other uses or eventually removed
# For now, the UI calls interactive_gesture_editor
def record_gesture(label, duration=1.0):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    print(f"Preparing to record gesture '{label}'. Get ready...")
    prep_start_time = time.time()
    while time.time() - prep_start_time < PREPARATION_TIME:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame during preparation.")
            cap.release(); hands.close(); return
        frame = cv2.flip(frame, 1)
        remaining_prep_time = PREPARATION_TIME - (time.time() - prep_start_time)
        if remaining_prep_time < 0: remaining_prep_time = 0
        cv2.putText(frame, f"Recording in {remaining_prep_time:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        rgb_prep = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_prep = hands.process(rgb_prep)
        if result_prep.multi_hand_landmarks:
            for hand_landmarks_prep in result_prep.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks_prep, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Recording Gesture', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Recording cancelled during preparation.")
            cap.release(); cv2.destroyAllWindows(); hands.close(); return
            
    print(f"Recording gesture '{label}' for {duration} seconds...")
    collected_raw_coords = []
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                raw_coords_frame = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                if len(raw_coords_frame) == 21:
                    collected_raw_coords.append(raw_coords_frame)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, "RECORDING...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Recording Gesture', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Recording cancelled."); break
    cap.release(); cv2.destroyAllWindows(); hands.close()

    if collected_raw_coords:
        avg_raw_coords = []
        num_frames = len(collected_raw_coords)
        if num_frames == 0:
            print("No frames with landmarks were collected.")
            return

        for i in range(21):
            x_sum, y_sum, z_sum = 0, 0, 0
            for frame_coords in collected_raw_coords:
                if i < len(frame_coords):
                    x_sum += frame_coords[i][0]; y_sum += frame_coords[i][1]; z_sum += frame_coords[i][2]
            avg_raw_coords.append((x_sum / num_frames, y_sum / num_frames, z_sum / num_frames))
        
        new_normalized_variation = normalize_landmarks(avg_raw_coords)

        if new_normalized_variation:
            filepath = os.path.join(TEMPLATES_DIR, f"{label}.pkl")
            all_variations_for_label = []
            if os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as f:
                        loaded_data = pickle.load(f)
                        # Expected format: list of lists of tuples (list of variations)
                        if isinstance(loaded_data, list) and (not loaded_data or (isinstance(loaded_data[0], list) and (not loaded_data[0] or isinstance(loaded_data[0][0], tuple)))):
                            all_variations_for_label = loaded_data
                        # Old format: list of tuples (single template)
                        elif isinstance(loaded_data, list) and (not loaded_data or isinstance(loaded_data[0], tuple)):
                            all_variations_for_label = [loaded_data] # Wrap it
                            print(f"Note: Existing template for '{label}' was old format. Converted to multi-variation.")
                        else:
                            print(f"Warning: File for '{label}' ({filepath}) exists but has unexpected format ({type(loaded_data)}). Will overwrite with new variation.")
                            all_variations_for_label = [] # Start fresh if format is unknown
                except Exception as e:
                    print(f"Error loading existing variations for '{label}' from {filepath}: {e}. Starting fresh.")
                    all_variations_for_label = [] # Start fresh on error
            
            all_variations_for_label.append(new_normalized_variation)
            try:
                with open(filepath, "wb") as f:
                    pickle.dump(all_variations_for_label, f)
                print(f"Saved/Appended gesture variation for '{label}'. Total variations: {len(all_variations_for_label)}")
            except Exception as e:
                print(f"Error saving variations for '{label}' to {filepath}: {e}")
        else:
            print(f"Could not normalize current recording for '{label}'. Not saved.")
    else:
        print("No valid landmarks collected for averaging.")

def load_templates():
    templates = {}
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)
    for file_name in os.listdir(TEMPLATES_DIR):
        if file_name.endswith(".pkl"):
            label = file_name[:-4]
            filepath = os.path.join(TEMPLATES_DIR, file_name)
            try:
                with open(filepath, "rb") as f:
                    loaded_data = pickle.load(f)
                    # Expected format: list of lists of tuples
                    if isinstance(loaded_data, list) and (not loaded_data or (isinstance(loaded_data[0], list) and (not loaded_data[0] or isinstance(loaded_data[0][0], tuple)))):
                        templates[label] = loaded_data
                    # Old format: list of tuples (single template) - wrap it
                    elif isinstance(loaded_data, list) and (not loaded_data or isinstance(loaded_data[0], tuple)):
                        templates[label] = [loaded_data]
                        print(f"Note: Template for '{label}' from {file_name} was old format, wrapped into list of variations.")
                    else:
                        print(f"Warning: Template file {file_name} for '{label}' has unexpected data format: {type(loaded_data)}. Skipping.")
            except Exception as e:
                print(f"Error loading template '{label}' from {file_name}: {e}. Skipping.")
    return templates

if __name__ == "__main__":
    # Example of how to test the interactive editor standalone
    # Create a dummy Tk root if you want to test messagebox popups from here
    # import tkinter as tk
    # dummy_root = tk.Tk()
    # dummy_root.withdraw() # Hide it

    test_gesture_name = input("Enter gesture name to test interactive editor: ")
    if test_gesture_name:
        print("Testing NEW interactive_gesture_editor...")
        # Pass None for app_root if not using Tkinter messageboxes from here
        saved = interactive_gesture_editor(None, test_gesture_name, is_new_gesture=True) 
        if saved:
            print(f"Interactive editor saved '{test_gesture_name}'")
        else:
            print(f"Interactive editor closed for '{test_gesture_name}' without saving.")
    else:
        print("No gesture name provided.")
    
    # Old test code for record_gesture:
    # gesture_name_to_record = input("Enter the name for the new gesture (e.g., 'fist', 'open_hand'): ")
    # if gesture_name_to_record:
    #     record_gesture(gesture_name_to_record, duration=3.0)
    # else:
    #     print("No gesture name provided. Exiting.")
   