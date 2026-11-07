import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # Video stream so not static
    max_num_hands=1,               # Detect one hand at a time
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Open video capture (webcam).
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# CSV filename to store the landmark data.
csv_filename = "hand_landmarks.csv"

# Define header for the CSV (63 values: x, y, z for each of 21 landmarks).
landmark_names = [
    "Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"
]
header = []
for name in landmark_names:
    header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

# Write the header to the CSV file.
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)

print(f"Press ENTER to capture the current hand landmark data into {csv_filename}.")
print("Press ESC to exit.")

# This variable will store the landmark values from the current frame (if any).
current_landmark_values = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Flip the frame horizontally for a mirror view.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hand landmarks.
    results = hands.process(frame_rgb)

    # Draw crosshair lines.
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 1)
    cv2.line(frame, (0, center_y), (width, center_y), (0, 255, 0), 1)

    # Reset current landmarks for this frame.
    current_landmark_values = None

    # If hand landmarks are detected, draw them and update current_landmark_values.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Gather 63 values: x, y, z for each of the 21 landmarks.
            landmark_values = []
            for landmark in hand_landmarks.landmark:
                landmark_values.extend([landmark.x, landmark.y, landmark.z])
            current_landmark_values = landmark_values

    # Show the processed frame.
    cv2.imshow("Hand Landmarks", frame)

    # Capture key presses.
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit.
        break
    elif key == 13 or key == 32:  # Enter key pressed.
        if current_landmark_values is not None:
            # Append the current landmark values to the CSV file.
            with open(csv_filename, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(current_landmark_values)
            csv_line = ",".join(f"{val:.4f}" for val in current_landmark_values)
            print(csv_line)
            print("Data captured.")
        else:
            print("No hand detected to capture.")

cap.release()
cv2.destroyAllWindows()
