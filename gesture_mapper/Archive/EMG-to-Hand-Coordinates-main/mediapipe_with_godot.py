import cv2
import mediapipe as mp
import socket

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # For video stream.
    max_num_hands=1,               # Detect one hand at a time.
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Setup UDP client socket.
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.settimeout(1)

# Open the webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Flip the frame for a mirror view and convert BGR to RGB.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands.
    results = hands.process(frame_rgb)

    # If hand landmarks are detected, draw them and send coordinates.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract 63 values: x, y, z for each of 21 landmarks.
            landmark_values = []
            for landmark in hand_landmarks.landmark:
                landmark_values.extend([landmark.x, landmark.y, landmark.z])
            # Convert list to a comma-separated string.
            landmark_string = ", ".join(f"{val:.4f}" for val in landmark_values)
            
            # Sends string to Godot (make sure the IP matches your Godot machine).
            client_socket.sendto(landmark_string.encode(), ("127.0.0.1", 5051))
            print(landmark_string)

    # Display the annotated frame.
    cv2.imshow("Hand Landmarks", frame)

    # Exit on pressing ESC.
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
