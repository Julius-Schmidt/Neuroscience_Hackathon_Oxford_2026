# gesture_matcher.py
# Module to match live gesture landmarks to saved templates using Euclidean distance

import math
# Removed: import pickle, os, from gesture_capture import load_templates
# load_templates is now handled by the main app or listener directly.

# This should ideally come from a shared constants file or be passed, 
# but for now, we know the subset length.
EXPECTED_SUBSET_LANDMARK_COUNT = 12 

def euclidean_distance(a, b):
    # Ensure points are tuples/lists of numbers and have same dimension (e.g., 3 for x,y,z)
    if not (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b)):
        # print(f"Error: euclidean_distance received incompatible points: {a}, {b}")
        return float('inf') # Or raise an error
    try:
        return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a, b)))
    except TypeError:
        # print(f"Error: non-numeric data in euclidean_distance for points: {a}, {b}")
        return float('inf')

def compare_landmarks(input_landmarks, template_variation):
    # input_landmarks and template_variation should now both be lists of EXPECTED_SUBSET_LANDMARK_COUNT (x,y,z) tuples
    if not (input_landmarks and template_variation and 
            isinstance(input_landmarks, list) and isinstance(template_variation, list) and
            len(input_landmarks) == EXPECTED_SUBSET_LANDMARK_COUNT and 
            len(template_variation) == EXPECTED_SUBSET_LANDMARK_COUNT):
        print(f"[compare_landmarks] Error: Invalid landmark sets. Expected {EXPECTED_SUBSET_LANDMARK_COUNT} landmarks for both.")
        print(f"Received: input_landmarks len = {len(input_landmarks) if input_landmarks else 'None'}, template_variation len = {len(template_variation) if template_variation else 'None'}")
        return float('inf')

    distances = []
    for i in range(len(input_landmarks)):
        dist = euclidean_distance(input_landmarks[i], template_variation[i])
        if dist == float('inf'): # Propagate error from euclidean_distance
            # print(f"Error: Bad distance for landmark {i}")
            return float('inf') 
        distances.append(dist)
    
    return sum(distances) / len(distances) if distances else float('inf')

def match_gesture(input_landmarks, templates_dict, confidence_threshold=0.5):
    """
    Matches input_landmarks against a dictionary of templates, where each template can have multiple variations.

    Args:
        input_landmarks: A list of EXPECTED_SUBSET_LANDMARK_COUNT normalized (x,y,z) tuples for the live gesture.
        templates_dict: A dictionary where keys are gesture names (str) and 
                        values are lists of variations. Each variation is a list of EXPECTED_SUBSET_LANDMARK_COUNT normalized (x,y,z) tuples.
        confidence_threshold: The divisor used in confidence calculation (1 - score / threshold).
                            A smaller value makes confidence drop faster with increasing score.

    Returns:
        (best_match_name, confidence_score) or (None, confidence_score_of_closest_non_match)
    """
    if not input_landmarks or not isinstance(input_landmarks, list) or len(input_landmarks) != EXPECTED_SUBSET_LANDMARK_COUNT:
        print(f"[match_gesture] Error: Invalid input_landmarks. Expected {EXPECTED_SUBSET_LANDMARK_COUNT} landmarks.")
        print(f"Received: input_landmarks len = {len(input_landmarks) if input_landmarks else 'None'}")
        return None, 0.0

    overall_best_score = float('inf')
    best_match_name = None

    for name, variations_list in templates_dict.items():
        if not variations_list or not isinstance(variations_list, list):
            # print(f"[Matcher] Skipping gesture '{name}': no variations or invalid format.")
            continue

        best_score_for_this_gesture_name = float('inf')

        for template_variation in variations_list:
            if not (isinstance(template_variation, list) and len(template_variation) == EXPECTED_SUBSET_LANDMARK_COUNT):
                print(f"[match_gesture] Skipping invalid variation for gesture '{name}'. Expected {EXPECTED_SUBSET_LANDMARK_COUNT} landmarks.")
                print(f"Received: template_variation len = {len(template_variation) if template_variation else 'None'}")
                continue # Skip malformed variations
            
            score = compare_landmarks(input_landmarks, template_variation)
            if score < best_score_for_this_gesture_name:
                best_score_for_this_gesture_name = score
        
        # Check if this gesture (considering its best variation) is the new overall best
        if best_score_for_this_gesture_name < overall_best_score:
            overall_best_score = best_score_for_this_gesture_name
            best_match_name = name

    # Calculate confidence based on the overall best score found across all gestures and their variations
    # The confidence_threshold parameter here is crucial for how scores translate to confidence.
    # It is NOT the direct action trigger threshold (which is usually 0.6 or similar on the *output* confidence).
    if overall_best_score == float('inf'): # No valid templates or variations were processed
        return None, 0.0
        
    confidence = max(0.0, 1.0 - (overall_best_score / confidence_threshold))
    
    # The decision to return best_match_name or None is typically based on another threshold 
    # applied to this calculated `confidence` (e.g., in gesture_listener.py: if confidence >= 0.6)
    # So, this function returns the best found name and its calculated confidence.
    return best_match_name, confidence

if __name__ == "__main__":
    # Example Usage (requires dummy data or actual templates)
    print("Gesture Matcher - Standalone Test")

    # 1. Create some dummy template variations
    dummy_variation1_g1 = [(i*0.01, i*0.01, 0.0) for i in range(21)]
    dummy_variation2_g1 = [(i*0.011, i*0.011, 0.0) for i in range(21)] # Slightly different
    dummy_variation1_g2 = [(i*0.02, i*0.02, 0.0) for i in range(21)]

    test_templates = {
        "gesture1": [dummy_variation1_g1, dummy_variation2_g1],
        "gesture2": [dummy_variation1_g2]
    }

    # 2. Create dummy input landmarks (normalized)
    live_gesture_similar_to_g1v1 = [(i*0.0105, i*0.0105, 0.0) for i in range(21)]
    live_gesture_similar_to_g2 = [(i*0.0205, i*0.0205, 0.0) for i in range(21)]
    live_gesture_very_different = [(i*0.5, i*0.5, 0.0) for i in range(21)]

    # 3. Test matching (using default confidence_threshold=0.5)
    print("\nMatching gesture similar to gesture1's first variation:")
    name, conf = match_gesture(live_gesture_similar_to_g1v1, test_templates)
    print(f"Matched: {name}, Confidence: {conf:.2f}")

    print("\nMatching gesture similar to gesture2:")
    name, conf = match_gesture(live_gesture_similar_to_g2, test_templates)
    print(f"Matched: {name}, Confidence: {conf:.2f}")

    print("\nMatching very different gesture:")
    name, conf = match_gesture(live_gesture_very_different, test_templates)
    print(f"Matched: {name}, Confidence: {conf:.2f} (expect low or None for name)")

    # Test with an empty templates dictionary
    print("\nMatching with empty templates dict:")
    name, conf = match_gesture(live_gesture_similar_to_g1v1, {})
    print(f"Matched: {name}, Confidence: {conf:.2f}")

    # Test with invalid input landmarks
    print("\nMatching with invalid live landmarks (None):")
    name, conf = match_gesture(None, test_templates)
    print(f"Matched: {name}, Confidence: {conf:.2f}")
    print("\nMatching with invalid live landmarks (empty list):")
    name, conf = match_gesture([], test_templates)
    print(f"Matched: {name}, Confidence: {conf:.2f}")
    print("\nMatching with invalid live landmarks (too few points):")
    name, conf = match_gesture([(0,0,0)]*5, test_templates)
    print(f"Matched: {name}, Confidence: {conf:.2f}")

    # Test with template dict having some malformed entries
    malformed_templates = {
        "good_gesture": [dummy_variation1_g1],
        "bad_gesture_empty_variations": [],
        "bad_gesture_not_a_list": dummy_variation1_g2, # Should be list of lists
        "bad_gesture_variation_not_list": [[(0,0,0)], None, "string_instead_of_list"],
        "bad_gesture_variation_wrong_len": [[(0,0,0)]*5] 
    }
    print("\nMatching with some malformed templates:")
    name, conf = match_gesture(live_gesture_similar_to_g1v1, malformed_templates)
    print(f"Matched: {name}, Confidence: {conf:.2f} (should match good_gesture)")
