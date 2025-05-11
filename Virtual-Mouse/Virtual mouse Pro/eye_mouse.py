import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Set pyautogui to be more responsive
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Initialize camera and MediaPipe Face Mesh
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Get screen size for mapping coordinates
screen_width, screen_height = pyautogui.size()

# Smoothing factor for cursor movement
prev_x, prev_y = 0, 0
smoothing = 0.3

# Variables for click detection
eyes_closed_start = None
eyes_closed_threshold = 1.5  # 1.5 seconds

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for natural movement
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Iris landmarks for right eye (left in flipped frame) and left eye (right in flipped frame)
        # Right eye iris center: landmark 468, Left eye iris center: landmark 473
        right_iris = face_landmarks.landmark[468]  # Right eye iris (left in flipped frame)
        left_iris = face_landmarks.landmark[473]   # Left eye iris (right in flipped frame)

        # Average the iris positions for more stable cursor movement
        iris_x = (right_iris.x + left_iris.x) / 2 * frame_width
        iris_y = (right_iris.y + left_iris.y) / 2 * frame_height

        # Map iris position to screen coordinates with very high sensitivity
        screen_x = np.interp(iris_x, [frame_width * 0.4, frame_width * 0.6], [0, screen_width])
        screen_y = np.interp(iris_y, [frame_height * 0.4, frame_height * 0.6], [0, screen_height])

        # Apply smoothing to cursor movement
        smoothed_x = prev_x + (screen_x - prev_x) * smoothing
        smoothed_y = prev_y + (screen_y - prev_y) * smoothing
        pyautogui.moveTo(smoothed_x, smoothed_y)
        prev_x, prev_y = smoothed_x, smoothed_y

        # Detect eye closure for left click
        # Right eye (left in flipped frame) landmarks: 145 (lower), 159 (upper)
        # Left eye (right in flipped frame) landmarks: 374 (lower), 386 (upper)
        right_eye_upper = face_landmarks.landmark[159].y
        right_eye_lower = face_landmarks.landmark[145].y
        left_eye_upper = face_landmarks.landmark[386].y
        left_eye_lower = face_landmarks.landmark[374].y

        # Calculate vertical distance for each eye (small distance = eye closed)
        right_eye_dist = abs(right_eye_lower - right_eye_upper)
        left_eye_dist = abs(left_eye_lower - left_eye_upper)

        # Slightly adjust threshold for more reliable detection
        eye_closed_threshold = 0.012  # Adjusted for better sensitivity

        # Check if both eyes are closed
        if right_eye_dist < eye_closed_threshold and left_eye_dist < eye_closed_threshold:
            if eyes_closed_start is None:
                eyes_closed_start = time.time()
            elif time.time() - eyes_closed_start >= eyes_closed_threshold:
                pyautogui.click()
                eyes_closed_start = None  # Reset timer after click
                pyautogui.sleep(0.3)  # Increased debounce to prevent multiple clicks
        else:
            eyes_closed_start = None  # Reset timer if eyes open

    # Display the frame (optional, can be removed for better performance)
    cv2.imshow('Eye-Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()