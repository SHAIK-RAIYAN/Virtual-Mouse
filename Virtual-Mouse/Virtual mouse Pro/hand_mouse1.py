import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from math import sqrt

# Set pyautogui to be more responsive
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Initialize camera and MediaPipe Hands
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size for mapping coordinates
screen_width, screen_height = pyautogui.size()

# Smoothing factor for cursor movement
prev_x, prev_y = 0, 0
smoothing = 0.3

# Function to calculate distance between two points
def get_distance(point1, point2):
    return sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for natural movement
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract key landmarks
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Use ring finger to move cursor
        ring_x = int(ring_tip.x * frame_width)
        ring_y = int(ring_tip.y * frame_height)
        screen_x = np.interp(ring_x, [0, frame_width], [0, screen_width])
        screen_y = np.interp(ring_y, [0, frame_height], [0, screen_height])

        # Apply smoothing to cursor movement
        smoothed_x = prev_x + (screen_x - prev_x) * smoothing
        smoothed_y = prev_y + (screen_y - prev_y) * smoothing
        pyautogui.moveTo(smoothed_x, smoothed_y)
        prev_x, prev_y = smoothed_x, smoothed_y

        # Pinch detection for clicks (thumb-index for left, thumb-middle for right)
        thumb_index_dist = get_distance(thumb_tip, index_tip)
        thumb_middle_dist = get_distance(thumb_tip, middle_tip)

        if thumb_index_dist < 0.05:  # Thumb-index pinch for left click
            pyautogui.click()
            pyautogui.sleep(0.2)  # Debounce to prevent multiple clicks

        if thumb_middle_dist < 0.05:  # Thumb-middle pinch for right click
            pyautogui.click(button='right')
            pyautogui.sleep(0.2)  # Debounce

        # Detect fist for scrolling (all fingers close to wrist)
        fingers = [index_tip, middle_tip, ring_tip, pinky_tip]
        fist_detected = all(get_distance(finger, wrist) < 0.15 for finger in fingers)
        
        if fist_detected:
            pyautogui.scroll(10)  # Scroll up when fist is detected
            pyautogui.sleep(0.1)  # Control scroll speed

    # Display the frame (optional, can be removed for better performance)
    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()